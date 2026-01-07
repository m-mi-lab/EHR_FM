from collections import defaultdict
from collections.abc import Callable
from omegaconf import DictConfig
import numpy as np
import torch
from transformers import PreTrainedModel


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, ctx, get_batch: Callable, eval_iters: int, tokens_of_interest: dict, 
    moe_config: DictConfig = None
):
    """
    Estimate loss on train and validation datasets.
    Fixed version to avoid hangs and memory issues.
    """
    block_size = model.config.n_positions if hasattr(model, 'config') else 2048
    if block_size > 10000:  # Sanity check
        block_size = 2048
    block_thresh = min(block_size // 2, 1024)  # Cap threshold to avoid memory issues
    
    # Initialize results dictionary
    out = {}
    
    # Set model to evaluation mode
    model.eval()



    # Evaluate on training and validation splits
    for split in ["train", "val"]:
        # Store losses across evaluation iterations
        losses = []
        
        # Store results for tokens
        all_tokens_res = defaultdict(list)
        toi_res = defaultdict(list)
        
        # Run evaluation for specified number of iterations
        last_output = None
        for i in range(eval_iters):
            try:
                # Get batch data
                X, Y = get_batch(split)
                
                # Forward pass with context manager (e.g., autocast)
                with ctx:
                    # Handle both encoder-decoder and decoder-only models
                    if isinstance(X, tuple):
                        output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                    else:
                        output = model(input_ids=X, labels=Y)
                    
                    # Store the last successful output for MoE metrics
                    last_output = output
                    
                    # Get loss and logits from model output
                    if hasattr(output, 'loss'):
                        loss = output.loss
                        logits = output.logits
                    else:
                        # Handle different output formats
                        loss = output[0] if isinstance(output, tuple) else output
                        logits = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                
                # Store loss
                losses.append(float(loss.item()))
                
                # Calculate accuracy metrics for validation set
                if split == "val" and logits is not None:
                    # Top-k accuracy, precision, and recall for all tokens
                    for k in [1, 3, 5]:
                        try:
                            acc = top_k_accuracy(logits, Y, k=k, threshold=block_thresh)
                            precision, recall = top_k_pr(logits, Y, k=k, threshold=block_thresh)
                            all_tokens_res[f"acc_top/all/{split}/{block_thresh}/k_{k}"].append(acc)
                            all_tokens_res[f"precision/all/{split}/{block_thresh}/k_{k}"].append(precision)
                            all_tokens_res[f"recall/all/{split}/{block_thresh}/k_{k}"].append(recall)
                        except Exception as e:
                            print(f"Error calculating accuracy for k={k}: {e}")
                    


                    # Top-k accuracy for special tokens
                    for stoken, token in tokens_of_interest.items():
                        for k in [1, 3, 10]:
                            try:
                                acc = top_k_acc_special(logits, Y, token, k=k, threshold=block_thresh)
                                precision, recall, tc = top_k_pr_special(logits, Y, token, k=k, threshold=block_thresh)
                                toi_res[f"acc_top/{stoken}/{split}/{block_thresh}/k_{k}"].append(acc)
                                toi_res[f"precision/{stoken}/{split}/{block_thresh}/k_{k}"].append((precision, tc))
                                toi_res[f"recall/{stoken}/{split}/{block_thresh}/k_{k}"].append((recall, tc))

                            except Exception as e:
                                print(f"Error calculating special token accuracy for {stoken}, k={k}: {e}")
            
            except Exception as e:
                print(f"Error during evaluation iteration {i} for {split}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next iteration instead of breaking
                continue
        
        # Calculate mean loss
        if losses:
            out[f"loss/{split}"] = sum(losses) / len(losses)
        else:
            out[f"loss/{split}"] = float('nan')
        
        # Calculate metrics for validation set
        if split == "val":
            out.update({test_name: np.mean(v) for test_name, v in all_tokens_res.items()})
            out.update({test_name: compute_weighted_mean(v) for test_name, v in toi_res.items()})

        is_moe_eval = hasattr(moe_config, 'n_experts') and getattr(moe_config, 'n_experts', 0) > 1
        if hasattr(model, 'moe_config'):
            is_moe_eval_model = hasattr(model.moe_config, 'n_experts') and getattr(model.moe_config, 'n_experts', 0) > 1
            if is_moe_eval_model and last_output is not None:
                if hasattr(last_output, 'aux_loss') and last_output.aux_loss is not None:
                        out[f'moe/aux_loss_raw/{split}'] = last_output.aux_loss.item() # Or average it if eval_iters > 1
                if hasattr(last_output, 'router_z_loss') and last_output.router_z_loss is not None:
                        out[f'moe/router_z_loss_raw/{split}'] = last_output.router_z_loss.item()
    model.train()
    
    return out



def compute_weighted_mean(v):
    weights = sum(x[1] for x in v)
    if weights == 0:
        return torch.nan
    return sum(x[0] * x[1] for x in v) / weights


@torch.jit.script
def top_k_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int, threshold: int) -> float:
    """Calculate top-k accuracy for all tokens."""
    # Safety checks
    seq_len = min(logits.size(1), y.size(1))
    threshold = min(threshold, seq_len - 1)  # Ensure threshold is valid
    
    # If threshold is negative or sequence is too short, return 0
    if threshold < 0 or seq_len <= threshold:
        return 0.0
    
    # Slice tensors to appropriate sizes
    logits = logits[:, threshold:seq_len, :]
    y_true = y[:, threshold:seq_len]
    
    # Ensure y_true has correct shape for comparison
    y_true = y_true.unsqueeze(-1)
    
    # Cap k to vocab size
    k = min(k, logits.size(-1))
    
    # Get top-k predictions
    _, indices = torch.topk(logits, k, dim=-1)
    
    # Check if true label is in top-k predictions
    correct = (indices == y_true).any(dim=-1)
    
    # Calculate accuracy
    if correct.numel() > 0:
        return float(correct.sum() / correct.numel())
    else:
        return 0.0

@torch.jit.script
def top_k_pr(logits: torch.Tensor, y: torch.Tensor, k: int, threshold: int) -> tuple[float, float]:
    """Calculate top-k precision and recall for all tokens."""
    # Safety checks
    seq_len = min(logits.size(1), y.size(1))
    threshold = min(threshold, seq_len - 1)  # Ensure threshold is valid
    

    if threshold < 0 or seq_len <= threshold:
        return 0.0, 0.0
    

    logits = logits[:, threshold:seq_len, :]
    y_true = y[:, threshold:seq_len]
    k = min(k, logits.size(-1))

    _, top_k_indices = torch.topk(logits, k, dim=-1) # top k
    y_expanded = y_true.unsqueeze(-1).expand(-1, -1, k)
    y_pred_true = (top_k_indices == y_expanded)

    tp = y_pred_true.sum(dim=-1)
    total = torch.ones_like(tp) * k
    precision = tp.sum().float() / total.sum().float()
    total_true = y_true.numel()

    recall = tp.sum().float() / total_true
    return precision, recall


@torch.jit.script
def top_k_acc_special(
    logits: torch.Tensor, y: torch.Tensor, token_of_interest: int, k: int, threshold: int
) -> tuple[float, int]:
    """Calculate top-k accuracy for specific tokens of interest."""
    # Safety checks
    seq_len = min(logits.size(1), y.size(1))
    threshold = min(threshold, seq_len - 1)  # Ensure threshold is valid
    
    # If threshold is negative or sequence is too short, return 0
    if threshold < 0 or seq_len <= threshold:
        return 0.0, 0
    
    # Slice tensors to appropriate sizes
    logits = logits[:, threshold:seq_len, :]
    y_true = y[:, threshold:seq_len]
    
    # Add dimension for comparison
    y_true = y_true.unsqueeze(-1)
    
    # Identify positions where token of interest appears
    interested = (y_true == token_of_interest)
    
    # If token doesn't appear, return 0
    weight = interested.sum()
    if weight == 0:
        return 0.0, 0
    
    # Cap k to vocab size
    k = min(k, logits.size(-1))
    
    # Get top-k predictions
    _, indices = torch.topk(logits, k, dim=-1)
    
    # Check if true label is in top-k predictions
    correct = (indices == y_true).any(dim=-1, keepdim=True)
    
    # Calculate accuracy for token of interest
    score = (correct & interested).sum() / weight
    
    return score.item(), weight.item()



@torch.jit.script
def top_k_pr_special(
    logits: torch.Tensor, y: torch.Tensor, token_of_interest: int, k: int, threshold: int
) -> tuple[float, float, int]:
    """Calculate top-k precision and recall for special tokens."""
    # Safety checks
    seq_len = min(logits.size(1), y.size(1))
    threshold = min(threshold, seq_len - 1)  # Ensure threshold is valid
    

    if threshold < 0 or seq_len <= threshold:
        return 0.0, 0.0, 0
    
    logits = logits[:, threshold:seq_len, :]
    y_true = y[:, threshold:seq_len]
    interested = (y_true == token_of_interest)
    token_count = interested.sum().item()
    if token_count == 0:
        return 0.0, 0.0, 0


    k = min(k, logits.size(-1))


    
    _, top_k_indices = torch.topk(logits, k, dim=-1) # top k
    token_in_topk = (top_k_indices == token_of_interest)
    predicted_token = token_in_topk.any(dim=-1)


    true_positives = (predicted_token & interested).sum().float()
    false_positives = (predicted_token & ~interested).sum().float()
    false_negatives = (~predicted_token & interested).sum().float()
    precision = 0.0
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)


    recall = 0.0
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    return precision, recall, token_count