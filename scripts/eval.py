#!/usr/bin/env python3
"""EHR-FM model evaluation script."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import yaml
import json
import argparse
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, roc_curve, average_precision_score
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.datasets.base import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary
from src.utils import load_env, resolve_model_path
from omegaconf import OmegaConf
from transformers import GPT2Config
from src.model import GPT2LMNoBiasModel

# Load environment variables
load_env()

def load_model(model_path):
    """Load model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'vocab_stoi' in checkpoint:
        # New format (MoE models)
        sorted_vocab_stoi = sorted(checkpoint['vocab_stoi'].items(), key=lambda item: item[1])
        vocab_list = [item[0] for item in sorted_vocab_stoi]
        vocab = Vocabulary(vocab=vocab_list)
        model_cfg = OmegaConf.create(checkpoint['model_configs'])
    elif 'vocab' in checkpoint:
        # Old format (Monolith models) - vocab is stored as stoi dict
        vocab_stoi = checkpoint['vocab']
        sorted_vocab_stoi = sorted(vocab_stoi.items(), key=lambda item: item[1])
        vocab_list = [item[0] for item in sorted_vocab_stoi]
        vocab = Vocabulary(vocab=vocab_list)
        # model_config is already a GPT2Config object, convert to dict first
        model_config_obj = checkpoint['model_config']
        model_cfg = OmegaConf.create({
            'n_positions': model_config_obj.n_positions,
            'n_embd': model_config_obj.n_embd,
            'n_layer': model_config_obj.n_layer,
            'n_head': model_config_obj.n_head,
            'activation_function': model_config_obj.activation_function,
            'resid_pdrop': model_config_obj.resid_pdrop,
            'embd_pdrop': model_config_obj.embd_pdrop,
            'attn_pdrop': model_config_obj.attn_pdrop,
            'bias': model_config_obj.bias,
            'n_inner': getattr(model_config_obj, 'n_inner', None),
        })
    else:
        raise KeyError("Vocabulary not found in checkpoint")
    
    # Calculate padded vocab size
    vocab_size = len(vocab)
    runtime_vocab_size = (vocab_size // 64 + 1) * 64 if vocab_size % 64 != 0 else vocab_size
    
    # Create model
    base_config = GPT2Config(
        vocab_size=runtime_vocab_size,
        n_positions=model_cfg.n_positions,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        activation_function=model_cfg.activation_function,
        resid_pdrop=model_cfg.resid_pdrop,
        embd_pdrop=model_cfg.embd_pdrop,
        attn_pdrop=model_cfg.attn_pdrop,
        bias=model_cfg.bias,
        n_inner=getattr(model_cfg, 'n_inner', None),
    )
    
    model = GPT2LMNoBiasModel(base_config, model_cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, vocab, device

def is_special_token(token_str):
    """Check if token is special."""
    special_prefixes = ['MEDS_', 'TIMELINE_', 'HOSPITAL_', 'ICU_', 'ED_', 'SOFA']
    return any(token_str.startswith(prefix) for prefix in special_prefixes)

def evaluate_model(model, dataset, vocab, device, num_samples=100):
    """Run next token prediction evaluation."""
    predictions = []
    
    print(f"Evaluating {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if i % 20 == 0:
                print(f"Progress: {i}/{num_samples}")
            
            # Get sample
            sample = dataset[i]
            if isinstance(sample, tuple):
                context, timeline = sample
                input_ids = torch.cat([context, timeline])
            else:
                input_ids = sample
            
            # Use only first 256 tokens to avoid issues
            input_ids = input_ids[:256].unsqueeze(0).to(device)
            
            # Skip if empty or too short
            if input_ids.size(1) < 2:
                continue
                
            # Get ground truth (next token from current sample)
            if input_ids.size(1) > 1:
                # Use the last token of current sample as ground truth
                true_token = input_ids[0, -1].item()
                # Use all but last token as context for prediction
                context_ids = input_ids[:, :-1]
            else:
                continue  # Skip if sample too short
            
            # Skip negative tokens (padding)
            if true_token < 0:
                continue
                
            # Get model prediction
            try:
                output = model(context_ids)
                logits = output.logits[0, -1, :]  # Last position logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get top-k predictions for k=1,3,5
                top_probs, top_indices = torch.topk(probs, k=5)
                
                # Decode tokens
                true_token_str = vocab.decode([true_token])[0]  # Get first element from list
                pred_tokens = [vocab.decode([idx.item()])[0] for idx in top_indices]
                
                # Store prediction
                predictions.append({
                    'true_token': true_token,
                    'true_token_str': true_token_str,
                    'true_prob': probs[true_token].item(),
                    'top_indices': top_indices.cpu().tolist(),
                    'top_probs': top_probs.cpu().tolist(),
                    'top_tokens': pred_tokens,
                    'is_special': is_special_token(true_token_str)
                })
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
    
    return predictions

def calculate_metrics(predictions, top_k_values=[1, 3, 5]):
    """Calculate evaluation metrics."""
    if not predictions:
        return {}
    
    metrics = {}
    
    # Separate normal and special tokens
    normal_preds = [p for p in predictions if not p['is_special']]
    special_preds = [p for p in predictions if p['is_special']]
    
    for pred_type, preds in [('overall', predictions), ('normal', normal_preds), ('special', special_preds)]:
        if not preds:
            continue
            
        type_metrics = {}
        
        # Top-k accuracy
        for k in top_k_values:
            correct = sum(1 for p in preds if p['true_token'] in p['top_indices'][:k])
            type_metrics[f'top_{k}_accuracy'] = correct / len(preds)
        
        # AUROC and F1 (binary classification: is prediction correct?)
        y_true = []
        y_scores = []
        
        for p in preds:
            # Binary: is top-1 prediction correct?
            y_true.append(1 if p['true_token'] == p['top_indices'][0] else 0)
            y_scores.append(p['top_probs'][0])  # Confidence of top prediction
        
        if len(set(y_true)) > 1:  # Need both classes for AUROC
            type_metrics['auroc'] = roc_auc_score(y_true, y_scores)
            type_metrics['auprc'] = average_precision_score(y_true, y_scores)
            type_metrics['f1'] = f1_score(y_true, [1 if s > 0.5 else 0 for s in y_scores])
        
        type_metrics['accuracy'] = accuracy_score(y_true, [1 if s > 0.5 else 0 for s in y_scores])
        
        # Store raw data for plotting
        type_metrics['y_true'] = y_true
        type_metrics['y_scores'] = y_scores
        
        metrics[pred_type] = type_metrics
    
    return metrics

def plot_comparison_results(all_results, output_dir):
    """Generate comparison plots for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    model_names = list(all_results.keys())
    
    # 1. AUROC Comparison (Overall tokens)
    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(model_names))
    auroc_values = [all_results[m]['metrics'].get('overall', {}).get('auroc', 0) for m in model_names]
    
    bars = ax.bar(x_pos, auroc_values, color=model_colors[:len(model_names)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, auroc_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - AUROC (Overall Tokens)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    
    # Smart y-axis limits - zoom to show differences
    if auroc_values and max(auroc_values) > 0:
        min_auroc = min([v for v in auroc_values if v > 0])
        max_auroc = max(auroc_values)
        
        # If all values are high (> 0.7), zoom in
        if min_auroc > 0.7:
            y_min = max(0.5, min_auroc - 0.05)  # Start from random or slightly below min
            y_max = min(1.0, max_auroc + 0.05)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, 1.0)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (0.5)')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_auroc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1 Score Comparison (Overall tokens)
    fig, ax = plt.subplots(figsize=(12, 7))
    f1_values = [all_results[m]['metrics'].get('overall', {}).get('f1', 0) for m in model_names]
    
    bars = ax.bar(x_pos, f1_values, color=model_colors[:len(model_names)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, f1_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - F1 Score (Overall Tokens)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    
    # Smart y-axis limits
    if f1_values and max(f1_values) > 0:
        min_f1 = min([v for v in f1_values if v > 0])
        max_f1 = max(f1_values)
        
        # If all values are high (> 0.6), zoom in
        if min_f1 > 0.6:
            y_min = max(0.0, min_f1 - 0.05)
            y_max = min(1.0, max_f1 + 0.05)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, 1.0)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_f1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy Comparison (Top-1, Top-3, Top-5)
    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.25
    
    all_acc_values = []
    for k_idx, k in enumerate([1, 3, 5]):
        accuracies = [all_results[m]['metrics'].get('overall', {}).get(f'top_{k}_accuracy', 0) for m in model_names]
        all_acc_values.extend(accuracies)
        bars = ax.bar(x_pos + k_idx * width, accuracies, width, label=f'Top-{k}', alpha=0.8)
        
        # Add value labels on bars for top-1 only (avoid clutter)
        if k == 1:
            for bar, val in zip(bars, accuracies):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Top-k Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    
    # Smart y-axis limits
    if all_acc_values and max(all_acc_values) > 0:
        min_acc = min([v for v in all_acc_values if v > 0])
        max_acc = max(all_acc_values)
        
        # If all values are high (> 0.6), zoom in
        if min_acc > 0.6:
            y_min = max(0.0, min_acc - 0.1)
            y_max = min(1.0, max_acc + 0.05)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, 1.0)
    
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curves Comparison (Overall tokens)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, model_name in enumerate(model_names):
        metrics = all_results[model_name]['metrics']
        if 'overall' in metrics and 'y_true' in metrics['overall'] and 'y_scores' in metrics['overall']:
            y_true = metrics['overall']['y_true']
            y_scores = metrics['overall']['y_scores']
            
            if len(set(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auroc = metrics['overall'].get('auroc', 0)
                ax.plot(fpr, tpr, color=model_colors[i % len(model_colors)], linewidth=3, 
                       label=f'{model_name} (AUROC={auroc:.3f})', alpha=0.8)
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Precision-Recall Curves Comparison (Overall tokens)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate baseline (random classifier based on prevalence)
    prevalence = None
    
    for i, model_name in enumerate(model_names):
        metrics = all_results[model_name]['metrics']
        if 'overall' in metrics and 'y_true' in metrics['overall'] and 'y_scores' in metrics['overall']:
            y_true = metrics['overall']['y_true']
            y_scores = metrics['overall']['y_scores']
            
            # Get prevalence from first model
            if prevalence is None and len(y_true) > 0:
                prevalence = np.mean(y_true)
            
            if len(set(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                auprc = metrics['overall'].get('auprc', 0)
                ax.plot(recall, precision, color=model_colors[i % len(model_colors)], linewidth=3, 
                       label=f'{model_name} (AUPRC={auprc:.3f})', alpha=0.8, marker='o', markersize=4, markevery=max(1, len(recall)//10))
    
    # Add baseline - random classifier has constant precision = prevalence
    if prevalence is not None:
        ax.axhline(y=prevalence, color='red', linestyle='--', linewidth=2, 
                   alpha=0.5, label=f'Random (prevalence={prevalence:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')  # Changed to lower left since curves are top-right
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_pr_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5b. Zoomed PR Curves (High Performance Region) - if models are doing well
    if prevalence is not None and prevalence < 0.9:  # Only create zoom if reasonable
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for i, model_name in enumerate(model_names):
            metrics = all_results[model_name]['metrics']
            if 'overall' in metrics and 'y_true' in metrics['overall'] and 'y_scores' in metrics['overall']:
                y_true = metrics['overall']['y_true']
                y_scores = metrics['overall']['y_scores']
                
                if len(set(y_true)) > 1:
                    precision, recall, _ = precision_recall_curve(y_true, y_scores)
                    auprc = metrics['overall'].get('auprc', 0)
                    ax.plot(recall, precision, color=model_colors[i % len(model_colors)], linewidth=3, 
                           label=f'{model_name} (AUPRC={auprc:.3f})', alpha=0.8, marker='o', markersize=6, markevery=max(1, len(recall)//10))
        
        # Zoom to high-performance region (top-right)
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison - Precision-Recall (Zoomed: High Performance)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.7, 1.0])  # Zoom to high recall
        ax.set_ylim([0.7, 1.0])  # Zoom to high precision
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_pr_curves_zoomed.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Decision Curves Comparison (Overall tokens)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    thresholds = np.linspace(0, 1, 101)
    
    for i, model_name in enumerate(model_names):
        metrics = all_results[model_name]['metrics']
        if 'overall' in metrics and 'y_true' in metrics['overall'] and 'y_scores' in metrics['overall']:
            y_true = np.array(metrics['overall']['y_true'])
            y_scores = np.array(metrics['overall']['y_scores'])
            
            if len(set(y_true)) > 1:
                net_benefits = []
                
                for threshold in thresholds:
                    if threshold == 0:
                        net_benefit = np.mean(y_true)
                    elif threshold == 1:
                        net_benefit = 0
                    else:
                        y_pred = (y_scores >= threshold).astype(int)
                        tp = np.sum((y_pred == 1) & (y_true == 1))
                        fp = np.sum((y_pred == 1) & (y_true == 0))
                        n = len(y_true)
                        net_benefit = (tp/n) - (fp/n) * (threshold/(1-threshold))
                    
                    net_benefits.append(net_benefit)
                
                ax.plot(thresholds, net_benefits, color=model_colors[i % len(model_colors)], 
                       linewidth=3, label=model_name, alpha=0.8)
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Treat None')
    
    ax.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Decision Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_decision_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Summary Comparison Table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create comprehensive table
    headers = ['Model'] + ['Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc', 'AUROC', 'AUPRC', 'F1', 'Accuracy']
    table_data = []
    
    for model_name in model_names:
        metrics = all_results[model_name]['metrics'].get('overall', {})
        row = [model_name]
        for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'auroc', 'auprc', 'f1', 'accuracy']:
            val = metrics.get(metric, 0)
            row.append(f'{val:.4f}' if val > 0 else 'N/A')
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values in each column
    for col_idx in range(1, len(headers)):
        values = [float(table_data[row][col_idx]) if table_data[row][col_idx] != 'N/A' else 0 
                 for row in range(len(table_data))]
        if values:
            max_val = max(values)
            for row_idx, val in enumerate(values):
                if val == max_val and val > 0:
                    table[(row_idx + 1, col_idx)].set_facecolor('#A8DADC')
                    table[(row_idx + 1, col_idx)].set_text_props(weight='bold')
    
    plt.title('Model Comparison - All Metrics (Best values highlighted)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f"{output_dir}/comparison_summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved to {output_dir}/")

def save_results_to_json(all_results, best_model, output_dir):
    """Save evaluation results to JSON file."""
    # Prepare results for JSON (remove non-serializable data)
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = {
            'metrics': {}
        }
        
        for token_type, metrics in results['metrics'].items():
            json_results[model_name]['metrics'][token_type] = {
                k: v for k, v in metrics.items() 
                if k not in ['y_true', 'y_scores']  # Remove raw arrays
            }
    
    # Add best model info
    json_results['best_model'] = {
        'name': best_model,
        'metrics': json_results[best_model]['metrics']['overall'] if best_model else {}
    }
    
    # Save to file
    output_file = f"{output_dir}/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

def load_config(config_path):
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="EHR-FM model evaluation")
    parser.add_argument('--config', type=str, default='eval_config.yaml',
                        help='Path to evaluation config file')
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file (optional)')
    args = parser.parse_args()
    
    # Load environment variables
    load_env(args.env)
    
    # Load configuration
    config = load_config(args.config)
    
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    num_samples = config.get('num_samples', 100)
    
    print("=== EHR-FM Evaluation ===")
    
    # Load dataset once
    print("Loading dataset...")
    dataset = TimelineDataset(Path(data_dir), n_positions=512)
    print(f"✅ Dataset loaded. Size: {len(dataset)}")
    
    all_results = {}
    
    # Evaluate each model
    for model_name, model_config in config['models'].items():
        print(f"\n--- Evaluating {model_name} ---")
        model_path_spec = model_config['path']
        
        # Resolve model path (handles S3 URIs and local paths)
        model_path = resolve_model_path(model_path_spec)
        
        # Load model
        print("Loading model...")
        model, vocab, device = load_model(str(model_path))
        print(f"✅ Model loaded. Vocab size: {len(vocab)}")
        
        # Run evaluation
        predictions = evaluate_model(model, dataset, vocab, device, num_samples)
        print(f"✅ Evaluation completed. Got {len(predictions)} valid predictions")
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(predictions)
        
        # Store results
        all_results[model_name] = {
            'metrics': metrics,
            'predictions': predictions
        }
        
        # Print results
        print(f"\n=== {model_name} Results ===")
        for token_type, type_metrics in metrics.items():
            print(f"\n{token_type.upper()} TOKENS:")
            for metric, value in type_metrics.items():
                if metric not in ['y_true', 'y_scores']:  # Skip raw data
                    print(f"  {metric}: {value:.4f}")
    
    # Generate comparison plots for all models in one folder
    print("\nGenerating comparison plots...")
    os.makedirs(output_dir, exist_ok=True)
    plot_comparison_results(all_results, output_dir)
    
    # Find and print best model
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)
    
    best_model = None
    best_auroc = 0
    
    for model_name, results in all_results.items():
        auroc = results['metrics'].get('overall', {}).get('auroc', 0)
        if auroc > best_auroc:
            best_auroc = auroc
            best_model = model_name
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   AUROC: {best_auroc:.4f}")
    
    if best_model:
        best_metrics = all_results[best_model]['metrics'].get('overall', {})
        print(f"\n   Full Metrics:")
        print(f"   - Top-1 Accuracy: {best_metrics.get('top_1_accuracy', 0):.4f}")
        print(f"   - Top-3 Accuracy: {best_metrics.get('top_3_accuracy', 0):.4f}")
        print(f"   - Top-5 Accuracy: {best_metrics.get('top_5_accuracy', 0):.4f}")
        print(f"   - AUROC: {best_metrics.get('auroc', 0):.4f}")
        print(f"   - F1 Score: {best_metrics.get('f1', 0):.4f}")
        print(f"   - AUPRC: {best_metrics.get('auprc', 0):.4f}")
    
    # Save results to JSON
    save_results_to_json(all_results, best_model, output_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
