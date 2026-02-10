#!/usr/bin/env python3
"""
Trajectory-Based Inference Script (ETHOS Style)

Generates trajectories autoregressively by sampling next tokens.
Runs multiple repetitions with different seeds for statistical aggregation.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import resample
import math

sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer.datasets.hospital_mortality import HospitalMortalityDataset
from src.tokenizer.datasets.mimic_icu import ICUMortalityDataset, ICUReadmissionDataset
from src.tokenizer.datasets.readmission import ReadmissionDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST
from src.utils import load_env, resolve_model_path
from omegaconf import OmegaConf
from transformers import GPT2Config
from src.model import GPT2LMNoBiasModel

# Task configurations
TASK_CONFIGS = {
    "hosp_mortality": {
        "dataset_class": HospitalMortalityDataset,
        "positive_token": ST.DEATH,
        "negative_token": ST.DISCHARGE,
        "stop_tokens": [ST.DEATH, ST.DISCHARGE],
        "max_new_tokens": 10000,
    },
    "icu_mortality": {
        "dataset_class": ICUMortalityDataset,
        "positive_token": ST.DEATH,
        "negative_token": ST.ICU_DISCHARGE,
        "stop_tokens": [ST.DEATH, ST.ICU_DISCHARGE],
        "max_new_tokens": 10000,
    },
    "hosp_readmission": {
        "dataset_class": ReadmissionDataset,
        "positive_token": ST.ADMISSION,
        "negative_token": ST.DEATH,
        "stop_tokens": [ST.ADMISSION, ST.DEATH],
        "max_new_tokens": 10000,
        "time_limit_days": 30,
    },
    "icu_readmission": {
        "dataset_class": ICUReadmissionDataset,
        "positive_token": ST.ICU_ADMISSION,
        "negative_token": ST.DISCHARGE,
        "stop_tokens": [ST.ICU_ADMISSION, ST.DISCHARGE, ST.DEATH],
        "max_new_tokens": 10000,
        "time_limit_days": 30,
    },
}


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if "vocab_stoi" in ckpt:
        vocab_items = sorted(ckpt["vocab_stoi"].items(), key=lambda x: x[1])
        vocab = Vocabulary(vocab=[k for k, _ in vocab_items])
        model_cfg = OmegaConf.create(ckpt["model_configs"])
    else:
        raise RuntimeError("Unsupported checkpoint format")

    vocab_size = len(vocab)
    padded_vocab_size = (vocab_size + 63) // 64 * 64

    base_cfg = GPT2Config(
        vocab_size=padded_vocab_size,
        n_positions=model_cfg.n_positions,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        resid_pdrop=model_cfg.resid_pdrop,
        embd_pdrop=model_cfg.embd_pdrop,
        attn_pdrop=model_cfg.attn_pdrop,
        activation_function=model_cfg.activation_function,
        bias=model_cfg.bias,
    )

    model = GPT2LMNoBiasModel(base_cfg, model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return model, vocab, device, model_cfg


@torch.no_grad()
def generate_trajectory(model, context, vocab, device, task_config, temperature=1.0, top_k=50):
    """Generate a single trajectory by sampling tokens autoregressively."""
    stop_token_ids = [vocab.stoi[t] for t in task_config["stop_tokens"]]
    max_new_tokens = task_config["max_new_tokens"]
    max_context = 2048
    
    # Trim context if too long
    if context.size(0) > max_context:
        context = context[-max_context:]
    
    timeline = context.clone().to(device)
    gen_token_num = 0
    
    while gen_token_num < max_new_tokens:
        # Get logits for next token
        if timeline.size(0) > max_context:
            input_seq = timeline[-max_context:]
        else:
            input_seq = timeline
            
        logits = model(input_seq.unsqueeze(0)).logits[0, -1, :len(vocab)]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(0)))
            logits[logits < v[-1]] = -float('Inf')
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to timeline
        timeline = torch.cat([timeline, next_token])
        gen_token_num += 1
        
        # Check stop condition
        if next_token.item() in stop_token_ids:
            return next_token.item(), gen_token_num
    
    # Max tokens reached
    return timeline[-1].item(), gen_token_num


def evaluate_task_trajectory(model, dataset, vocab, device, task_config, num_samples, num_trajectories=20, temperature=1.0, top_k=50):
    """Evaluate using trajectory-based method (ETHOS style)."""
    pos_id = vocab.stoi[task_config["positive_token"]]
    neg_id = vocab.stoi[task_config["negative_token"]]
    
    results = []
    
    for i in tqdm(range(num_samples), desc="Evaluating"):
        ctx, meta = dataset[i]
        ctx = ctx[ctx >= 0]
        
        gt_token = meta["expected"]
        gt_id = pos_id if gt_token == task_config["positive_token"] else vocab.stoi[gt_token]
        
        # Generate multiple trajectories
        outcomes = []
        for _ in range(num_trajectories):
            outcome_token, _ = generate_trajectory(model, ctx, vocab, device, task_config, temperature, top_k)
            outcomes.append(outcome_token)
        
        # Calculate frequency-based probability
        positive_count = sum(1 for t in outcomes if t == pos_id)
        probability = positive_count / num_trajectories
        
        results.append({
            "positive_probability": probability,
            "ground_truth": 1 if gt_id == pos_id else 0,
            "positive_count": positive_count,
            "total_trajectories": num_trajectories,
        })
    
    return results


def bootstrap_ci(y_true, y_score, metric_func, n_bootstraps=1000, ci=95):
    """Calculate bootstrap confidence intervals for a metric."""
    np.random.seed(42)
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        score = metric_func(y_true_boot, y_score_boot)
        scores.append(score)
    
    lower = (100 - ci) / 2
    upper = 100 - lower
    ci_lower = np.percentile(scores, lower)
    ci_upper = np.percentile(scores, upper)
    
    return ci_lower, ci_upper


def calculate_statistics(results):
    y_true = np.array([r["ground_truth"] for r in results])
    y_score = np.array([r["positive_probability"] for r in results])

    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.0, "auprc": 0.0, "accuracy": 0.0}

    auroc = roc_auc_score(y_true, y_score)
    auroc_ci_lower, auroc_ci_upper = bootstrap_ci(y_true, y_score, roc_auc_score, n_bootstraps=1000)
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    
    y_pred = (y_score > 0.5).astype(int)
    accuracy = np.mean(y_true == y_pred)
    
    positive_rate = np.mean(y_true) * 100

    return {
        "auroc": float(auroc),
        "auroc_ci_lower": float(auroc_ci_lower),
        "auroc_ci_upper": float(auroc_ci_upper),
        "auprc": float(auprc),
        "accuracy": float(accuracy),
        "positive_rate": float(positive_rate),
        "n_samples": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Trajectory-Based Inference (ETHOS Style)")
    parser.add_argument("--config", default="scripts/eval_final_config.yaml", help="Config file")
    parser.add_argument("--test", required=True, choices=list(TASK_CONFIGS.keys()),
                       help="Task to evaluate")
    parser.add_argument("--model", help="Model path (overrides config)")
    parser.add_argument("--data-dir", help="Data directory (overrides config)")
    parser.add_argument("--output", default="infer_trajectory_results", help="Output directory")
    parser.add_argument("--suffix", default="", help="Output file suffix (e.g., rep1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-trajectories", type=int, default=20, help="Number of trajectories per sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    load_env(None)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Register OmegaConf resolver
    if not OmegaConf.has_resolver("oc.env"):
        OmegaConf.register_new_resolver(
            "oc.env",
            lambda k, default=None: os.environ.get(k, default)
        )

    # Load config
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Get parameters
    model_path = args.model or cfg.get("model")
    data_dir = args.data_dir or cfg.get("data_dir")
    num_samples = cfg.get("num_samples", 100)

    if model_path is None:
        raise ValueError("No model specified. Use --model or set in config.")
    if data_dir is None:
        raise ValueError("No data directory specified. Use --data-dir or set in config.")

    model_path = resolve_model_path(model_path)

    print(f"\n{'='*70}")
    print(f"TRAJECTORY-BASED INFERENCE (ETHOS STYLE)")
    print(f"{'='*70}")
    print(f"Task: {args.test}")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Samples: {num_samples}")
    print(f"Trajectories per sample: {args.num_trajectories}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Load model
    model, vocab, device, model_cfg = load_model(model_path)
    print(f"✅ Model loaded")

    # Load dataset
    task_config = TASK_CONFIGS[args.test]
    dataset = task_config["dataset_class"](
        input_dir=Path(data_dir),
        n_positions=model_cfg.n_positions,
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # Cap num_samples
    num_samples = min(num_samples, len(dataset))
    print(f"✅ Evaluating {num_samples} samples")

    # Run evaluation
    results = evaluate_task_trajectory(
        model, dataset, vocab, device, task_config, num_samples,
        num_trajectories=args.num_trajectories,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Calculate statistics
    stats = calculate_statistics(results)
    stats["task"] = args.test
    stats["method"] = "trajectory"
    stats["num_samples"] = num_samples
    stats["num_trajectories"] = args.num_trajectories
    stats["temperature"] = args.temperature
    stats["top_k"] = args.top_k
    stats["seed"] = args.seed
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"AUROC:       {stats['auroc']:.3f} [{stats['auroc_ci_lower']:.3f}, {stats['auroc_ci_upper']:.3f}]")
    print(f"AUPRC:       {stats['auprc']:.4f}")
    print(f"Accuracy:    {stats['accuracy']:.4f}")
    print(f"N samples:   {stats['n_samples']} ({stats['positive_rate']:.1f}% positives)")
    print(f"{'='*70}\n")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{args.suffix}" if args.suffix else ""
    output_file = output_dir / f"{args.test}{suffix}.json"
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()

