import torch
from torch.utils.data import DataLoader, Subset
import os
import time
import math
from pathlib import Path
# import hydra # Hydra is not directly used for running, but its config objects are loaded
from omegaconf import DictConfig, OmegaConf, MISSING
import numpy as np
from transformers import GPT2Config
from loguru import logger
from sklearn.metrics import roc_curve, auc
import argparse
from contextlib import nullcontext
from collections import defaultdict
import json # For saving summary

# Assuming your project structure allows these imports
# Ensure this script is run from a location where 'src' is discoverable
# or adjust python path accordingly.
from src.tokenizer.constants import SpecialToken as ST
from src.tokenizer.datasets import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary
from src.model import GPT2LMNoBiasModel, ModelOutput
from src.metrics import top_k_accuracy, top_k_pr, top_k_acc_special, top_k_pr_special

def get_memory_usage(device):
    """Gets current and peak GPU memory usage."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
        # Reset peak stats for next measurement if needed, though here it's per-experiment
        # torch.cuda.reset_peak_memory_stats(device) 
        return allocated, reserved, max_allocated, max_reserved
    return 0, 0, 0, 0

def calculate_roc_auc(y_true_binary, y_probs_positive_class, token_name_for_log):
    """Calculates ROC curve and AUC for a specific token."""
    try:
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs_positive_class)
        roc_auc_value = auc(fpr, tpr)
        # Logger info for this is now inside the main reporting loop for better context
        # logger.info(f"ROC AUC for {token_name_for_log}: {roc_auc_value:.4f}") 
        return roc_auc_value
    except ValueError as e: # Happens if only one class present in y_true_binary
        logger.warning(f"Could not compute ROC AUC for {token_name_for_log}: {e}. Ensure y_true contains both classes for this token.")
        return float('nan')


@torch.no_grad()
def evaluate_model_core(model, val_dataloader, device, ctx, tokens_of_interest, cfg_eval_params, vocab_size_param):
    """
    Core evaluation function for a single model.
    Calculates loss, accuracy, precision, recall, and ROC AUC for specified tokens.
    """
    model.eval()
    # Initialize results dictionary. Loss will be stored here.
    all_metrics_results = {"loss": float('nan')} 
    
    # Accumulators for metrics over all batches
    all_tokens_metrics_accum = {
        "accuracy": defaultdict(list), "precision": defaultdict(list), "recall": defaultdict(list)
    }
    special_tokens_metrics_accum = {
        token_name: {
            "accuracy": defaultdict(list), "precision": defaultdict(list), "recall": defaultdict(list),
            "y_true_binary_list": [], "y_probs_positive_class_list": [] # For ROC AUC
        } for token_name in tokens_of_interest.keys()
    }

    # Determine block threshold for metric calculation from model or eval config
    block_size = model.base_config.n_positions
    block_thresh = min(block_size // 2, cfg_eval_params.get("block_thresh", 1024))
    
    total_loss_val = 0.0
    num_batches_processed = 0

    logger.info(f"Starting evaluation loop for {cfg_eval_params.eval_batches} batches using block_thresh: {block_thresh}...")
    for i, (X_batch, Y_batch) in enumerate(val_dataloader):
        if i >= cfg_eval_params.eval_batches:
            break # Stop if max eval batches reached
        
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device).long()

        with ctx: # Autocast context
            output_model: ModelOutput = model(X_batch, Y_batch)
            loss_val = output_model.loss
            logits_val = output_model.logits
        
        if loss_val is not None:
            total_loss_val += loss_val.item()
        num_batches_processed += 1

        # --- All Tokens Metrics (predicting the next token) ---
        for k_top_val in cfg_eval_params.get("k_values_all_tokens", [1, 3, 5]):
            acc_val = top_k_accuracy(logits_val, Y_batch, k=k_top_val, threshold=block_thresh)
            precision_val, recall_val = top_k_pr(logits_val, Y_batch, k=k_top_val, threshold=block_thresh)
            all_tokens_metrics_accum["accuracy"][f"top_{k_top_val}"].append(acc_val)
            all_tokens_metrics_accum["precision"][f"top_{k_top_val}"].append(precision_val)
            all_tokens_metrics_accum["recall"][f"top_{k_top_val}"].append(recall_val)

        # --- Special Tokens Metrics (targeted prediction of specific tokens) ---
        for token_name_key, token_id_val in tokens_of_interest.items():
            # Accuracy, Precision, Recall for this special token
            for k_top_val in cfg_eval_params.get("k_values_special_tokens", [1, 3, 10]):
                acc_val_s, _ = top_k_acc_special(logits_val, Y_batch, token_id_val, k=k_top_val, threshold=block_thresh)
                precision_val_s, recall_val_s, count_s = top_k_pr_special(logits_val, Y_batch, token_id_val, k=k_top_val, threshold=block_thresh)
                
                special_tokens_metrics_accum[token_name_key]["accuracy"][f"top_{k_top_val}"].append(acc_val_s)
                if count_s > 0: # Only add precision/recall if the token was present and predicted/actual
                    special_tokens_metrics_accum[token_name_key]["precision"][f"top_{k_top_val}"].append((precision_val_s, count_s))
                    special_tokens_metrics_accum[token_name_key]["recall"][f"top_{k_top_val}"].append((recall_val_s, count_s))

            # Collect data for ROC AUC calculation for this special token
            # Logits and Y are sliced from block_thresh onwards
            relevant_logits_val = logits_val[:, block_thresh:, :]
            relevant_y_val = Y_batch[:, block_thresh:]

            if relevant_y_val.numel() > 0: # Ensure there are tokens to evaluate
                y_true_flat_val = relevant_y_val.reshape(-1)
                # Get probabilities for all tokens from logits
                probs_flat_val = torch.softmax(relevant_logits_val.reshape(-1, vocab_size_param), dim=-1)
                
                # For the current special token:
                # True binary labels (1 if it's the token_id_val, 0 otherwise)
                y_true_binary_token_val = (y_true_flat_val == token_id_val).cpu().numpy()
                # Predicted probabilities for this specific token_id_val
                y_probs_positive_class_token_val = probs_flat_val[:, token_id_val].cpu().numpy()
                
                special_tokens_metrics_accum[token_name_key]["y_true_binary_list"].extend(y_true_binary_token_val)
                special_tokens_metrics_accum[token_name_key]["y_probs_positive_class_list"].extend(y_probs_positive_class_token_val)
        
        # Log progress periodically
        if (i + 1) % (cfg_eval_params.eval_batches // 10 or 1) == 0: # Log roughly 10 times or every batch if fewer than 10
             logger.info(f"Processed batch {i+1}/{cfg_eval_params.eval_batches}")

    logger.info("Evaluation loop finished. Finalizing metrics...")
    # Calculate average loss if batches were processed
    if num_batches_processed > 0:
        all_metrics_results["loss"] = total_loss_val / num_batches_processed
    
    # Finalize "All Tokens" metrics by averaging
    all_metrics_results["all_tokens"] = {}
    for metric_name, k_results in all_tokens_metrics_accum.items():
        all_metrics_results["all_tokens"][metric_name] = {
            k_val: np.mean(scores) if scores else float('nan') for k_val, scores in k_results.items()
        }

    # Finalize "Special Tokens" metrics
    all_metrics_results["special_tokens"] = {}
    for token_name_key, metrics_dict_val in special_tokens_metrics_accum.items():
        all_metrics_results["special_tokens"][token_name_key] = {"roc_auc": float('nan')} # Initialize ROC AUC
        for metric_name, k_results in metrics_dict_val.items():
            if metric_name == "accuracy": # Simple average for accuracy
                all_metrics_results["special_tokens"][token_name_key][metric_name] = {
                    k_val: np.mean(scores) if scores else float('nan') for k_val, scores in k_results.items()
                }
            elif metric_name in ["precision", "recall"]: # Weighted average for precision/recall based on counts
                 all_metrics_results["special_tokens"][token_name_key][metric_name] = {}
                 for k_val, weighted_scores in k_results.items():
                    total_weight = sum(s[1] for s in weighted_scores) if weighted_scores else 0
                    all_metrics_results["special_tokens"][token_name_key][metric_name][k_val] = \
                        sum(s[0] * s[1] for s in weighted_scores) / total_weight if total_weight > 0 else float('nan')
            # y_true_binary_list and y_probs_positive_class_list are handled next for ROC AUC
            
        # Calculate ROC AUC for this special token using all collected probabilities and true labels
        y_true_b_all = np.array(metrics_dict_val["y_true_binary_list"])
        y_probs_p_all = np.array(metrics_dict_val["y_probs_positive_class_list"])
        
        if len(y_true_b_all) > 0 and len(np.unique(y_true_b_all)) > 1: # Check for sufficient data
            roc_auc_val = calculate_roc_auc(y_true_b_all, y_probs_p_all, token_name_key)
            all_metrics_results["special_tokens"][token_name_key]["roc_auc"] = roc_auc_val
        else:
            logger.warning(f"Skipping ROC AUC for {token_name_key}: insufficient data or only one class present in y_true.")
            
    return all_metrics_results

def find_experiment_directories(base_outputs_dir_path: Path):
    """Scans for valid Hydra experiment directories under the base_outputs_dir_path."""
    experiment_dirs_list = []
    logger.info(f"Scanning for experiments in: {base_outputs_dir_path}")
    
    for potential_exp_dir_dot_hydra in base_outputs_dir_path.rglob(".hydra"):
        exp_root_dir = potential_exp_dir_dot_hydra.parent # The directory containing .hydra
        
        # Check for presence of a checkpoint file and Hydra config
        has_checkpoint = (exp_root_dir / "best_model.pt").exists() or \
                         (exp_root_dir / "final_model.pt").exists() or \
                         any(exp_root_dir.glob("ckpt_*.pt"))
        has_hydra_config = (exp_root_dir / ".hydra" / "config.yaml").exists()

        if has_checkpoint and has_hydra_config:
            experiment_dirs_list.append(exp_root_dir)
            logger.debug(f"Found potential experiment: {exp_root_dir}")
        else:
            if not has_checkpoint:
                logger.debug(f"Skipping {exp_root_dir}, no checkpoint file found.")
            if not has_hydra_config:
                 logger.debug(f"Skipping {exp_root_dir}, .hydra/config.yaml not found.")

    unique_experiment_dirs = sorted(list(set(experiment_dirs_list)))
    logger.info(f"Found {len(unique_experiment_dirs)} unique experiment directories.")
    return unique_experiment_dirs


def main_multi_experiment_evaluation(cli_args_parsed):
    """Main function to discover and evaluate multiple experiments."""
    logger.remove() # Remove default handler to customize
    log_format_multi = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
    )
    logger.add(lambda msg: print(msg, end=""), colorize=True, format=log_format_multi, level="INFO")

    base_output_dir = Path(cli_args_parsed.outputs_dir)
    experiment_paths_list = find_experiment_directories(base_output_dir)

    if not experiment_paths_list:
        logger.error(f"No valid experiment directories found in {base_output_dir}. "
                     "Ensure directories contain a '.hydra' subdirectory and a model checkpoint "
                     " (e.g., 'best_model.pt', 'final_model.pt', or 'ckpt_ITER.pt').")
        return

    logger.info(f"Discovered {len(experiment_paths_list)} experiments for evaluation.")
    all_experiments_summary = [] # To store summary of all experiments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Global device for evaluation: {device}")

    for i, current_exp_path in enumerate(experiment_paths_list):
        logger.info(f"\n{'='*90}")
        logger.info(f"PROCESSING EXPERIMENT {i+1}/{len(experiment_paths_list)}: {current_exp_path}")
        logger.info(f"{'='*90}")
        
        # Determine checkpoint file to use
        checkpoint_target_file = "best_model.pt"
        chkpt_path = current_exp_path / checkpoint_target_file
        if not chkpt_path.exists():
            checkpoint_target_file = "final_model.pt"
            chkpt_path = current_exp_path / checkpoint_target_file
            if not chkpt_path.exists():
                 found_ckpts = sorted(
                    list(current_exp_path.glob("ckpt_*.pt")),
                    key=lambda p: int(p.stem.split('_')[-1]), reverse=True # Sort by iteration number
                 )
                 if found_ckpts:
                     chkpt_path = found_ckpts[0]
                     checkpoint_target_file = chkpt_path.name
                     logger.info(f"Using latest iteration checkpoint: {checkpoint_target_file}")
                 else:
                    logger.warning(f"No 'best_model.pt', 'final_model.pt', or 'ckpt_*.pt' found in {current_exp_path}. Skipping this experiment.")
                    continue
        
        logger.info(f"Attempting to load checkpoint: {chkpt_path}")
        
        try:
            map_loc = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Load checkpoint, allowing unpickling of custom classes from trusted source
            checkpoint_data = torch.load(chkpt_path, map_location=map_loc, weights_only=False) 

            # --- Load Hydra Config from Checkpoint ---
            if 'hydra_config_full' not in checkpoint_data:
                logger.error(f"Hydra config ('hydra_config_full') not found in {chkpt_path}. Skipping.")
                continue
            exp_cfg = OmegaConf.create(checkpoint_data['hydra_config_full'])
            
            # --- Setup Evaluation Parameters ---
            # Use CLI args if provided, else fallback to training config values
            train_batch_size = exp_cfg.train.batch_size
            world_size_train = exp_cfg.get('world_size', 1) # world_size might not be in older configs
            grad_accum_steps = exp_cfg.train.gradient_accumulation_steps
            
            # Calculate effective batch size per GPU during training for a sensible default eval batch size
            # Handle potential division by zero if world_size or grad_accum_steps is 0 or missing
            eff_train_batch_per_gpu = train_batch_size
            if world_size_train > 0 and grad_accum_steps > 0:
                 eff_train_batch_per_gpu = train_batch_size // (world_size_train * grad_accum_steps)

            eval_cfg_params = OmegaConf.create({
                "eval_batches": cli_args_parsed.eval_batches,
                "batch_size": cli_args_parsed.batch_size or eff_train_batch_per_gpu,
                "num_workers": cli_args_parsed.num_workers if cli_args_parsed.num_workers is not None else exp_cfg.train.num_workers,
                "k_values_all_tokens": [1, 3, 5], # Top-k values for 'all tokens' metrics
                "k_values_special_tokens": [1, 3, 10], # Top-k values for 'special tokens' metrics
                "block_thresh": cli_args_parsed.block_thresh 
            })
            if eval_cfg_params.batch_size < 1: eval_cfg_params.batch_size = 1 # Ensure batch size is at least 1
            logger.info(f"Effective evaluation batch size for this run: {eval_cfg_params.batch_size}")

            # --- Vocabulary Setup ---
            if 'vocab_stoi' not in checkpoint_data:
                logger.error(f"Vocabulary ('vocab_stoi') not found in {chkpt_path}. Skipping.")
                continue
            # Reconstruct vocabulary list from stoi mapping
            sorted_vocab_stoi = sorted(checkpoint_data['vocab_stoi'].items(), key=lambda item: item[1])
            exp_vocab_list = [item[0] for item in sorted_vocab_stoi]
            current_vocab = Vocabulary(vocab=exp_vocab_list) # Assumes Vocabulary can be init with list of tokens
            runtime_vocab_size = (len(current_vocab) // 64 + 1) * 64 if len(current_vocab) % 64 != 0 else len(current_vocab)
            logger.info(f"Vocabulary loaded for {current_exp_path.name}. Size: {len(current_vocab)} (Padded: {runtime_vocab_size})")
            
            # Define special tokens of interest for targeted metrics
            exp_tokens_of_interest = {
                st.value: current_vocab.encode(st.value) 
                for st in [ST.DEATH, ST.ADMISSION, ST.DISCHARGE] if st.value in current_vocab.stoi
            }
            logger.info(f"Tokens of Interest for this run: {exp_tokens_of_interest}")

            # --- Dataset Setup ---
            data_tokenized_dir_str = exp_cfg.data.tokenized_dir
            # Resolve data_tokenized_dir if it's relative to the original experiment output dir
            # This assumes the original config stored it relative to where train.py was run (the hydra output dir)
            original_exp_output_dir = Path(exp_cfg.train.out_dir) # This is the key
            
            data_tokenized_dir = Path(data_tokenized_dir_str)
            if not data_tokenized_dir.is_absolute():
                # This logic might need adjustment based on how paths were stored.
                # If tokenized_dir was stored relative to the project root, this is different.
                # Assuming it was stored relative to the *output directory* of the original run.
                # However, often paths in hydra configs are made absolute.
                # The most robust way is if `exp_cfg.data.tokenized_dir` is already absolute.
                # If not, we might need a more complex way to find it or assume it's relative
                # to the project root from where eval.py is run.
                # For now, let's assume it's either absolute or resolvable as is.
                logger.warning(f"Data tokenized_dir '{data_tokenized_dir_str}' is relative. Assuming it's resolvable from current context or project root.")


            if not data_tokenized_dir.exists():
                logger.error(f"Tokenized data directory '{data_tokenized_dir}' from experiment config does not exist. Skipping.")
                continue
            
            exp_full_dataset = TimelineDataset(
                input_dir=data_tokenized_dir,
                n_positions=exp_cfg.model.n_positions, # From the specific experiment's model config
                is_encoder_decoder=False, # Assuming decoder-only
            )
            val_set_fraction = exp_cfg.data.val_split_fraction
            val_set_count = int(val_set_fraction * len(exp_full_dataset))
            train_set_count = len(exp_full_dataset) - val_set_count
            
            data_gen = torch.Generator().manual_seed(exp_cfg.seed) # Use seed from this experiment's config
            _, val_idx = torch.utils.data.random_split(
                range(len(exp_full_dataset)), [train_set_count, val_set_count], generator=data_gen
            )
            exp_val_dataset = Subset(exp_full_dataset, val_idx)
            logger.info(f"Validation dataset size for this experiment: {len(exp_val_dataset)}")

            exp_val_dataloader = DataLoader(
                exp_val_dataset, batch_size=eval_cfg_params.batch_size, shuffle=False, # No shuffle for eval
                num_workers=eval_cfg_params.num_workers, pin_memory=True
            )

            # --- Model Setup ---
            if 'model_configs' not in checkpoint_data: 
                logger.error(f"Model config ('model_configs') not found in {chkpt_path}. Skipping.")
                continue
            
            exp_model_hydra_cfg = OmegaConf.create(checkpoint_data['model_configs']) # This was cfg.model
            # Create the base GPT2Config for the model shell
            exp_base_model_cfg_obj = GPT2Config(
                vocab_size=runtime_vocab_size, 
                n_positions=exp_model_hydra_cfg.n_positions, 
                n_embd=exp_model_hydra_cfg.n_embd,
                n_layer=exp_model_hydra_cfg.n_layer, 
                n_head=exp_model_hydra_cfg.n_head,
                activation_function=exp_model_hydra_cfg.activation_function,
                resid_pdrop=exp_model_hydra_cfg.resid_pdrop, 
                embd_pdrop=exp_model_hydra_cfg.embd_pdrop,
                attn_pdrop=exp_model_hydra_cfg.attn_pdrop, 
                bias=exp_model_hydra_cfg.bias,
                n_inner=getattr(exp_model_hydra_cfg, 'n_inner', None), # For MLP, if specified
            )
            
            # Instantiate the model
            current_model = GPT2LMNoBiasModel(
                base_gpt_config=exp_base_model_cfg_obj, 
                moe_hydra_config=exp_model_hydra_cfg # Pass the specific MoE config for this experiment
            ).to(device)
            
            # Load model state dictionary
            model_sd = checkpoint_data['model']
            curr_model_keys = set(current_model.state_dict().keys())
            filtered_sd = {k: v for k, v in model_sd.items() if k in curr_model_keys} # Filter to current model's keys
            m_keys, u_keys = current_model.load_state_dict(filtered_sd, strict=False) # Load non-strictly
            if m_keys: logger.warning(f"Missing keys when loading model state for {current_exp_path.name}: {m_keys}")
            if u_keys: logger.warning(f"Unexpected keys in checkpoint model state for {current_exp_path.name}: {u_keys}")
            logger.info(f"Model for {current_exp_path.name} loaded onto {device}.")

            alloc_b, res_b, max_alloc_b_seen, max_res_b_seen = get_memory_usage(device)
            logger.info(f"GPU Mem before eval ({current_exp_path.name}): {alloc_b:.1f}MB alloc, {res_b:.1f}MB reserved. Max seen so far: {max_alloc_b_seen:.1f} (A), {max_res_b_seen:.1f} (R)")

            # --- Autocast Context Setup ---
            # Use dtype from the specific experiment's training config
            ptdtype_val = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[exp_cfg.train.dtype]
            autocast_on = (exp_cfg.train.dtype != 'float32') and (device.type == 'cuda')
            eval_ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype_val, enabled=autocast_on)
            logger.info(f"Using autocast for {current_exp_path.name} with dtype: {ptdtype_val} (enabled: {autocast_on})")

            # --- Perform Evaluation ---
            exp_metrics = evaluate_model_core(
                current_model, exp_val_dataloader, device, eval_ctx, 
                exp_tokens_of_interest, eval_cfg_params, runtime_vocab_size
            )
            
            # Store results for this experiment
            run_identifier = str(current_exp_path.relative_to(base_output_dir)) # Use relative path as ID
            all_experiments_summary.append({"experiment_id": run_identifier, 
                                            "checkpoint_used": checkpoint_target_file,
                                            "metrics": exp_metrics})

            # --- Log Results for This Experiment ---
            logger.info(f"\n--- Results for Experiment: {run_identifier} (using {checkpoint_target_file}) ---")
            # Only log loss if it's not NaN (i.e., it was computed)
            loss_value = exp_metrics.get('loss', float('nan'))
            if not math.isnan(loss_value):
                 logger.info(f"  Validation Loss: {loss_value:.4f}")
            
            if "all_tokens" in exp_metrics:
                logger.info("  --- All Token Metrics (next token prediction) ---")
                for metric_t, k_met in exp_metrics["all_tokens"].items():
                    logger.info(f"    {metric_t.capitalize()}:")
                    for k_v, scr in k_met.items(): logger.info(f"      {k_v}: {scr:.4f}")
            
            if "special_tokens" in exp_metrics:
                logger.info("  --- Special Token Metrics (targeted prediction) ---")
                for tok_n, tok_met_val in exp_metrics["special_tokens"].items():
                    logger.info(f"    Token: {tok_n}")
                    if "roc_auc" in tok_met_val and not math.isnan(tok_met_val['roc_auc']): 
                        logger.info(f"      ROC AUC: {tok_met_val['roc_auc']:.4f}")
                    for metric_t, k_met in tok_met_val.items():
                        if metric_t != "roc_auc" and metric_t not in ["y_true_binary_list", "y_probs_positive_class_list"] and isinstance(k_met, dict) :
                            logger.info(f"      {metric_t.capitalize()}:")
                            for k_v, scr in k_met.items(): 
                                if not math.isnan(scr): logger.info(f"        {k_v}: {scr:.4f}")
            
            # Model Parameters
            num_p_total = current_model.num_parameters(exclude_embeddings=False)
            num_p_no_emb = current_model.num_parameters(exclude_embeddings=True)
            logger.info("  --- Model Parameters ---")
            logger.info(f"    Total parameters: {num_p_total:,}")
            logger.info(f"    Parameters (excluding token embeddings): {num_p_no_emb:,}")

            # GPU Usage Post-Eval
            alloc_a, res_a, max_alloc_a_run, max_res_a_run = get_memory_usage(device)
            logger.info("  --- GPU Usage (Post-Eval for this run) ---")
            logger.info(f"    Initial GPU Mem for this eval run: {alloc_b:.1f}MB allocated")
            logger.info(f"    Peak GPU Mem during this eval run: {max_alloc_a_run:.1f}MB (Allocated), {max_res_a_run:.1f}MB (Reserved)")
            logger.info(f"    Final GPU Mem after this eval run: {alloc_a:.1f}MB allocated, {res_a:.1f}MB reserved")

            # Cleanup for the current experiment to free GPU memory
            del current_model, exp_val_dataloader, exp_full_dataset, exp_val_dataset 
            del checkpoint_data, exp_cfg, exp_model_hydra_cfg, exp_base_model_cfg_obj
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Try to release cached memory
            logger.info(f"Resources for experiment '{run_identifier}' released from memory.")

        except Exception as e_outer:
            logger.exception(f"CRITICAL ERROR evaluating experiment {current_exp_path}: {e_outer}")
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Attempt to clean up GPU memory on error as well

    logger.info(f"\n{'='*90}")
    logger.info("MULTI-EXPERIMENT EVALUATION COMPLETE.")
    processed_count = len(all_experiments_summary)
    total_discovered = len(experiment_paths_list)
    logger.info(f"Successfully processed and reported results for {processed_count}/{total_discovered} discovered experiments.")

    # Save summary of all results to a JSON file
    summary_file_path = base_output_dir / "summary.json" # Changed filename
    try:
        # Custom default function to handle numpy floats and other non-serializable types
        def json_converter(o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, Path):
                 return str(o)
            # Add more types here if needed
            # raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
            return str(o) # Fallback to string for other complex types

        with open(summary_file_path, 'w') as f:
            json.dump(all_experiments_summary, f, indent=4, default=json_converter) 
        logger.info(f"Summary of all experiment results saved to: {summary_file_path}")
    except Exception as e_json:
        logger.error(f"Could not save summary JSON: {e_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for multiple EHR_FM model experiments.")
    parser.add_argument("outputs_dir", type=str, 
                        help="Path to the base Hydra outputs directory (e.g., ./outputs) containing experiment subdirectories.")
    parser.add_argument("--eval_batches", type=int, default=50, 
                        help="Number of batches from validation set for evaluation per experiment. Default: 50.")
    parser.add_argument("--batch_size", type=int, default=None, 
                        help="Batch size for evaluation. If None, defaults to train config value adjusted for single GPU evaluation.")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of dataloader workers. If None, defaults to train config value from checkpoint.")
    parser.add_argument("--block_thresh", type=int, default=1024, 
                        help="Metrics are calculated for tokens after this position in sequence. Default: 1024.")
    
    parsed_cli_args = parser.parse_args()
    main_multi_experiment_evaluation(parsed_cli_args)