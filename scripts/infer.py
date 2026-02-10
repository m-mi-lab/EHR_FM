#!/usr/bin/env python3
"""
Direct Probability Inference Script

Single forward pass per patient, extracts softmax probabilities directly.
Supports binary classification and regression tasks.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    mean_absolute_error, mean_squared_error
)
from scipy.stats import pearsonr
from sklearn.utils import resample
import math
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer.datasets.hospital_mortality import HospitalMortalityDataset
from src.tokenizer.datasets.mimic_icu import ICUMortalityDataset, ICUReadmissionDataset
from src.tokenizer.datasets.readmission import ReadmissionDataset
from src.tokenizer.datasets.mimic_icu import ICUMortalityDataset as ICULOSDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST
from src.utils import load_env, resolve_model_path
from omegaconf import OmegaConf
from transformers import GPT2Config
from src.model import GPT2LMNoBiasModel


TASK_CONFIGS = {
    "hosp_mortality": {
        "dataset_class": HospitalMortalityDataset,
        "positive_token": ST.DEATH,
        "negative_tokens": [ST.DISCHARGE],
        "task_type": "binary",
    },
    "icu_mortality": {
        "dataset_class": ICUMortalityDataset,
        "positive_token": ST.DEATH,
        "negative_tokens": [ST.ICU_DISCHARGE],
        "task_type": "binary",
    },
    "hosp_readmission": {
        "dataset_class": ReadmissionDataset,
        "positive_token": ST.ADMISSION,
        "negative_tokens": [ST.DEATH, ST.TIMELINE_END],
        "task_type": "binary",
    },
    "icu_readmission": {
        "dataset_class": ICUReadmissionDataset,
        "positive_token": ST.ICU_ADMISSION,
        "negative_tokens": [ST.DISCHARGE, ST.DEATH],
        "task_type": "binary",
    },
    "icu_los": {
        "dataset_class": ICULOSDataset,
        "positive_token": ST.ICU_DISCHARGE,
        "negative_tokens": [ST.DEATH],
        "task_type": "regression",
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


def get_direct_probability(model, context, vocab_size, device, positive_token_id, negative_token_ids):
    with torch.no_grad():
        if context.size(0) > 2048:
            context = context[-2048:]
        
        logits = model(context.unsqueeze(0).to(device)).logits[0, -1, :vocab_size]
        probs = torch.softmax(logits, dim=-1)
        
        positive_prob = probs[positive_token_id].item()
        negative_prob = sum(probs[tid].item() for tid in negative_token_ids)
        
        total_prob = positive_prob + negative_prob
        probability = positive_prob / total_prob if total_prob > 0 else 0.5
        
        return probability


def predict_los_direct(model, context, vocab, device, discharge_token_id, death_token_id):
    with torch.no_grad():
        if context.size(0) > 2048:
            context = context[-2048:]
        
        logits = model(context.unsqueeze(0).to(device)).logits[0, -1, :len(vocab)]
        probs = torch.softmax(logits, dim=-1)
        
        discharge_prob = probs[discharge_token_id].item()
        death_prob = probs[death_token_id].item()
        
        total_prob = discharge_prob + death_prob
        death_probability = death_prob / total_prob if total_prob > 0 else 0.5
        
        if death_probability > 0.5:
            return None
        
        baseline_los = 2.5
        predicted_los = baseline_los * (1.0 / max(discharge_prob, 0.1))
        predicted_los = max(0.1, min(30.0, predicted_los))
        
        return predicted_los


def evaluate_task(model, dataset, vocab, device, task_config, num_samples, batch_size=32):
    task_type = task_config["task_type"]
    pos_id = vocab.stoi[task_config["positive_token"]]
    neg_ids = [vocab.stoi[t] for t in task_config["negative_tokens"]]
    vocab_size = len(vocab)

    results = []
    
    # Collect all contexts and metadata first
    print("Loading samples...")
    all_contexts = []
    all_metadata = []
    for i in tqdm(range(num_samples), desc="Loading"):
        ctx, meta = dataset[i]
        ctx = ctx[ctx >= 0]
        all_contexts.append(ctx)
        all_metadata.append(meta)
    
    # Process in batches
    print("Running inference...")
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_contexts = all_contexts[batch_start:batch_end]
            batch_metadata = all_metadata[batch_start:batch_end]
            
            # Pad contexts to same length for batching
            max_len = min(max(ctx.size(0) for ctx in batch_contexts), 2048)
            batch_tensor = torch.zeros(len(batch_contexts), max_len, dtype=torch.long, device=device)
            
            for idx, ctx in enumerate(batch_contexts):
                ctx_trimmed = ctx[-max_len:] if ctx.size(0) > max_len else ctx
                batch_tensor[idx, -ctx_trimmed.size(0):] = ctx_trimmed.to(device)
            
            # Single forward pass for entire batch
            logits = model(batch_tensor).logits[:, -1, :vocab_size]
            probs = torch.softmax(logits, dim=-1)
            
            # Extract results for each sample in batch
            for idx, (ctx, meta) in enumerate(zip(batch_contexts, batch_metadata)):
                if task_type == "regression":
                    gt_los_microseconds = meta.get("true_token_time", 0)
                    gt_los_days = gt_los_microseconds / (1e6 * 60 * 60 * 24)
                    
                    discharge_prob = probs[idx, pos_id].item()
                    death_prob = probs[idx, neg_ids[0]].item()
                    
                    total_prob = discharge_prob + death_prob
                    death_probability = death_prob / total_prob if total_prob > 0 else 0.5
                    
                    if death_probability <= 0.5:
                        baseline_los = 2.5
                        predicted_los = baseline_los * (1.0 / max(discharge_prob, 0.1))
                        predicted_los = max(0.1, min(30.0, predicted_los))
                        
                        results.append({
                            "predicted_los_days": predicted_los,
                            "ground_truth_los_days": gt_los_days,
                        })
                else:
                    gt_token = meta["expected"]
                    gt_id = pos_id if gt_token == task_config["positive_token"] else vocab.stoi[gt_token]
                    
                    positive_prob = probs[idx, pos_id].item()
                    negative_prob = sum(probs[idx, tid].item() for tid in neg_ids)
                    
                    total_prob = positive_prob + negative_prob
                    probability = positive_prob / total_prob if total_prob > 0 else 0.5
                    
                    results.append({
                        "positive_probability": probability,
                        "ground_truth": 1 if gt_id == pos_id else 0,
                    })

    return results


def bootstrap_ci(y_true, y_score, metric_func, n_bootstraps=1000, ci=95):
    """Calculate bootstrap confidence intervals for a metric."""
    np.random.seed(42)  # For reproducibility
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        score = metric_func(y_true_boot, y_score_boot)
        scores.append(score)
    
    # Calculate percentiles for CI
    lower = (100 - ci) / 2
    upper = 100 - lower
    ci_lower = np.percentile(scores, lower)
    ci_upper = np.percentile(scores, upper)
    
    return ci_lower, ci_upper


def calculate_statistics(results, task_type):
    if task_type == "regression":
        y_true = np.array([r["ground_truth_los_days"] for r in results])
        y_pred = np.array([r["predicted_los_days"] for r in results])
        
        if len(y_true) == 0:
            return {"mae": 0.0, "rmse": 0.0, "correlation": 0.0, "n_samples": 0}
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        
        correlation = 0.0
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            correlation, _ = pearsonr(y_true, y_pred)
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "correlation": float(correlation),
            "n_samples": len(y_true),
        }
    else:
        y_true = np.array([r["ground_truth"] for r in results])
        y_score = np.array([r["positive_probability"] for r in results])

        if len(np.unique(y_true)) < 2:
            return {"auroc": 0.0, "auprc": 0.0, "accuracy": 0.0}

        # Standard AUC calculation (commented for reference - using bootstrap instead)
        # auroc = roc_auc_score(y_true, y_score)
        
        # Bootstrap AUC with 95% CI (ETHOS paper style)
        auroc = roc_auc_score(y_true, y_score)
        auroc_ci_lower, auroc_ci_upper = bootstrap_ci(y_true, y_score, roc_auc_score, n_bootstraps=1000)
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recall, precision)
        
        y_pred = (y_score > 0.5).astype(int)
        accuracy = np.mean(y_true == y_pred)
        
        # Calculate positive rate
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


def plot_results(stats, task_name, output_dir):
    """Generate simple bar plot for results."""
    task_type = "regression" if "mae" in stats else "binary"
    
    if task_type == "binary":
        metrics = ["auroc", "auprc", "accuracy"]
        values = [stats.get(m, 0) for m in metrics]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{task_name.upper()} - Binary Classification', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    else:
        metrics = ["mae", "rmse", "correlation"]
        values = [stats.get(m, 0) for m in metrics]
        colors = ['#d62728', '#9467bd', '#8c564b']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{task_name.upper()} - Regression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plot_file = Path(output_dir) / f"{task_name}_plot.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    return plot_file


def main():
    parser = argparse.ArgumentParser(description="Direct Probability Inference")
    parser.add_argument("--config", default="scripts/eval_final_config.yaml", help="Config file")
    parser.add_argument("--test", required=True, choices=list(TASK_CONFIGS.keys()),
                       help="Task to evaluate")
    parser.add_argument("--model", help="Model path (overrides config)")
    parser.add_argument("--data-dir", help="Data directory (overrides config)")
    parser.add_argument("--output", default="infer_results", help="Output directory")
    parser.add_argument("--suffix", default="", help="Output file suffix (e.g., rep1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    args = parser.parse_args()

    load_env(None)
    
    # Set random seed for reproducibility
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

    # Get model and data paths
    model_path = args.model or cfg.get("model")
    data_dir = args.data_dir or cfg.get("data_dir")
    num_samples = cfg.get("num_samples", 100)
    batch_size = cfg.get("batch_size", 32)

    if model_path is None:
        raise ValueError("No model specified. Use --model or set in config.")
    if data_dir is None:
        raise ValueError("No data directory specified. Use --data-dir or set in config.")

    model_path = resolve_model_path(model_path)

    print(f"\n{'='*70}")
    print(f"DIRECT PROBABILITY INFERENCE")
    print(f"{'='*70}")
    print(f"Task: {args.test}")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Samples: {num_samples}")
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

    # Cap num_samples to dataset size
    num_samples = min(num_samples, len(dataset))
    print(f"✅ Evaluating {num_samples} samples")

    # Run evaluation
    results = evaluate_task(model, dataset, vocab, device, task_config, num_samples, batch_size)
    
    # Calculate statistics
    stats = calculate_statistics(results, task_config["task_type"])
    stats["task"] = args.test
    stats["method"] = "direct"
    stats["num_samples"] = num_samples
    stats["seed"] = args.seed
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    if task_config["task_type"] == "regression":
        print(f"MAE:         {stats['mae']:.4f} days")
        print(f"RMSE:        {stats['rmse']:.4f} days")
        print(f"Correlation: {stats['correlation']:.4f}")
        print(f"N samples:   {stats['n_samples']}")
    else:
        print(f"AUROC:       {stats['auroc']:.3f} [{stats['auroc_ci_lower']:.3f}, {stats['auroc_ci_upper']:.3f}]")
        print(f"AUPRC:       {stats['auprc']:.4f}")
        print(f"Accuracy:    {stats['accuracy']:.4f}")
        print(f"N samples:   {stats['n_samples']} ({stats['positive_rate']:.1f}% positives)")
    print(f"{'='*70}\n")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    suffix = f"_{args.suffix}" if args.suffix else ""
    output_file = Path(args.output) / f"{args.test}{suffix}.json"
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Results saved to {output_file}")
    
    # Generate plot if requested
    if args.plot:
        plot_file = plot_results(stats, args.test, args.output)
        print(f"✅ Plot saved to {plot_file}")
    
    print()


if __name__ == "__main__":
    main()

