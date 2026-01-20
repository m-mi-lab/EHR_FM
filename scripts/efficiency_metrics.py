#!/usr/bin/env python3
"""
Efficiency Metrics Evaluation Script

Evaluates model efficiency metrics including:
- Model parameters
- Inference time (clock time)
- Energy consumption estimation
- Pareto efficiency (accuracy vs FLOPs/GPU memory)
- Next token prediction accuracy

Compares mixture of experts models against baseline models.
"""

import sys
import time
import json
import torch
import psutil
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import GPT2LMNoBiasModel
from src.tokenizer.datasets.base import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary
from src.utils import artifact_loader

# Load environment variables
artifact_loader.load_env()

def count_model_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_M': total_params / 1e6,
        'trainable_parameters_M': trainable_params / 1e6
    }

def estimate_flops_per_token(model_cfg, vocab_size):
    """Estimate FLOPs per token for transformer model."""
    # Rough FLOP estimation for transformer
    # Based on: https://arxiv.org/abs/2001.08361
    
    d_model = model_cfg.n_embd
    n_layers = model_cfg.n_layer
    n_heads = model_cfg.n_head
    seq_len = model_cfg.n_positions
    
    # Attention FLOPs: 4 * n_layers * seq_len * d_model^2
    attention_flops = 4 * n_layers * seq_len * (d_model ** 2)
    
    # MLP FLOPs: 8 * n_layers * seq_len * d_model^2 (assuming 4x expansion)
    mlp_flops = 8 * n_layers * seq_len * (d_model ** 2)
    
    # Embedding FLOPs: 2 * seq_len * d_model * vocab_size
    embedding_flops = 2 * seq_len * d_model * vocab_size
    
    total_flops = attention_flops + mlp_flops + embedding_flops
    
    return {
        'flops_per_forward_pass': total_flops,
        'flops_per_token': total_flops / seq_len,
        'attention_flops': attention_flops,
        'mlp_flops': mlp_flops,
        'embedding_flops': embedding_flops
    }

def measure_gpu_memory(model, device):
    """Measure GPU memory usage."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Measure memory before and after model
        memory_before = torch.cuda.memory_allocated(device)
        
        # Move model to device if not already there
        model = model.to(device)
        torch.cuda.synchronize()
        
        memory_after = torch.cuda.memory_allocated(device)
        memory_peak = torch.cuda.max_memory_allocated(device)
        
        return {
            'gpu_memory_allocated_MB': (memory_after - memory_before) / 1024**2,
            'gpu_memory_peak_MB': memory_peak / 1024**2,
            'gpu_memory_reserved_MB': torch.cuda.memory_reserved(device) / 1024**2
        }
    else:
        return {
            'gpu_memory_allocated_MB': 0,
            'gpu_memory_peak_MB': 0,
            'gpu_memory_reserved_MB': 0
        }

def measure_inference_time(model, sample_inputs, device, num_runs=10):
    """Measure inference time with multiple runs for accuracy."""
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(3):
            _ = model(sample_inputs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(sample_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'avg_inference_time_ms': np.mean(times) * 1000,
        'std_inference_time_ms': np.std(times) * 1000,
        'min_inference_time_ms': np.min(times) * 1000,
        'max_inference_time_ms': np.max(times) * 1000,
        'throughput_tokens_per_sec': sample_inputs.size(1) / np.mean(times)
    }

def estimate_energy_consumption(inference_time_ms, gpu_power_watts=250):
    """Estimate energy consumption based on inference time."""
    # Rough estimation assuming GPU power consumption
    inference_time_hours = inference_time_ms / (1000 * 3600)
    energy_wh = gpu_power_watts * inference_time_hours
    
    return {
        'estimated_energy_per_inference_wh': energy_wh,
        'estimated_energy_per_token_wh': energy_wh / 1,  # per token
        'gpu_power_assumption_watts': gpu_power_watts
    }

def evaluate_next_token_accuracy(model, dataset, vocab, device, num_samples=100):
    """Evaluate next token prediction accuracy."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            if isinstance(sample, tuple):
                context, timeline = sample
                full_sequence = torch.cat([context, timeline])
            else:
                full_sequence = sample
            
            # Skip short sequences
            if full_sequence.size(0) < 10:
                continue
            
            # Use first part as context, last token as target
            context_len = min(full_sequence.size(0) - 1, 512)  # Limit context length
            context_ids = full_sequence[:context_len].unsqueeze(0).to(device)
            target_id = full_sequence[context_len].item()
            
            try:
                output = model(context_ids)
                logits = output.logits[0, -1, :]
                predicted_id = torch.argmax(logits).item()
                
                if predicted_id == target_id:
                    correct_predictions += 1
                total_predictions += 1
                
            except Exception as e:
                print(f"Error in sample {i}: {e}")
                continue
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return {
        'next_token_accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }

def load_model(model_path, device):
    """Load model and return model, vocab, and config."""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'vocab_stoi' in checkpoint:
        # New format (MoE models)
        sorted_vocab_stoi = sorted(checkpoint['vocab_stoi'].items(), key=lambda item: item[1])
        vocab_list = [item[0] for item in sorted_vocab_stoi]
        vocab = Vocabulary(vocab=vocab_list)
        model_cfg = OmegaConf.create(checkpoint['model_configs'])
    elif 'vocab' in checkpoint:
        # Old format (Monolith models)
        vocab_stoi = checkpoint['vocab']
        sorted_vocab_stoi = sorted(vocab_stoi.items(), key=lambda item: item[1])
        vocab_list = [item[0] for item in sorted_vocab_stoi]
        vocab = Vocabulary(vocab=vocab_list)
        
        # Convert GPT2Config to dict
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
    
    # Create model with GPT2Config like eval.py does
    from transformers import GPT2Config
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
    model = model.to(device)
    
    print(f"✅ Model loaded: {model_cfg.n_layer} layers, {model_cfg.n_head} heads")
    print(f"   Vocab size: {len(vocab)}, Runtime vocab size: {model.lm_head.out_features}")
    
    return model, vocab, model_cfg

def evaluate_model_efficiency(model_path, model_name, dataset, device, config):
    """Evaluate all efficiency metrics for a single model."""
    print(f"\n{'='*50}")
    print(f"Evaluating efficiency metrics for {model_name}")
    print(f"{'='*50}")
    
    # Load model
    model, vocab, model_cfg = load_model(model_path, device)
    
    # 1. Count parameters
    param_metrics = count_model_parameters(model)
    print(f"📊 Parameters: {param_metrics['total_parameters_M']:.2f}M total")
    
    # 2. Estimate FLOPs
    flop_metrics = estimate_flops_per_token(model_cfg, len(vocab))
    print(f"⚡ FLOPs per token: {flop_metrics['flops_per_token']:.2e}")
    
    # 3. Measure GPU memory
    memory_metrics = measure_gpu_memory(model, device)
    print(f"💾 GPU Memory: {memory_metrics['gpu_memory_allocated_MB']:.2f} MB")
    
    # 4. Create sample input for timing
    sample_input = torch.randint(0, len(vocab), (1, 512)).to(device)
    
    # 5. Measure inference time
    timing_metrics = measure_inference_time(model, sample_input, device, num_runs=config.timing_runs)
    print(f"⏱️  Inference time: {timing_metrics['avg_inference_time_ms']:.2f} ms")
    print(f"🚀 Throughput: {timing_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
    
    # 6. Estimate energy consumption
    energy_metrics = estimate_energy_consumption(timing_metrics['avg_inference_time_ms'])
    print(f"⚡ Energy per inference: {energy_metrics['estimated_energy_per_inference_wh']:.6f} Wh")
    
    # 7. Evaluate next token accuracy
    accuracy_metrics = evaluate_next_token_accuracy(
        model, dataset, vocab, device, num_samples=config.accuracy_samples
    )
    print(f"🎯 Next token accuracy: {accuracy_metrics['next_token_accuracy']:.4f}")
    
    # Combine all metrics
    all_metrics = {
        'model_name': model_name,
        'model_path': str(model_path),
        'model_config': OmegaConf.to_container(model_cfg),
        **param_metrics,
        **flop_metrics,
        **memory_metrics,
        **timing_metrics,
        **energy_metrics,
        **accuracy_metrics
    }
    
    return all_metrics

def calculate_pareto_efficiency(results):
    """Calculate Pareto efficiency metrics."""
    pareto_metrics = []
    
    for result in results:
        # Efficiency ratios
        accuracy_per_param = result['next_token_accuracy'] / result['total_parameters_M']
        accuracy_per_flop = result['next_token_accuracy'] / result['flops_per_token']
        accuracy_per_memory = result['next_token_accuracy'] / max(result['gpu_memory_allocated_MB'], 1)
        accuracy_per_time = result['next_token_accuracy'] / result['avg_inference_time_ms']
        
        pareto_metric = {
            'model_name': result['model_name'],
            'accuracy_per_param': accuracy_per_param,
            'accuracy_per_flop': accuracy_per_flop,
            'accuracy_per_memory': accuracy_per_memory,
            'accuracy_per_time': accuracy_per_time,
            'efficiency_score': (accuracy_per_param + accuracy_per_flop + accuracy_per_memory + accuracy_per_time) / 4
        }
        
        pareto_metrics.append(pareto_metric)
    
    return pareto_metrics

def create_parallel_coordinates_plot(results, output_dir):
    """Create parallel coordinates plot for all efficiency metrics."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Normalize metrics to 0-1 scale for fair comparison
    metrics = ['next_token_accuracy', 'throughput_tokens_per_sec', 'total_parameters_M', 
               'avg_inference_time_ms', 'estimated_energy_per_inference_wh']
    
    # Get values and normalize properly for relative comparison
    normalized_data = {}
    for metric in metrics:
        values = [r[metric] for r in results]
        
        # For time, energy, and parameters, lower is better, so invert them first
        if 'time' in metric or 'energy' in metric or 'parameters' in metric:
            # Convert to "efficiency" - invert so lower time/energy/params = higher score
            max_val = max(values)
            values = [max_val / v for v in values]  # Higher efficiency for lower values
        
        # Now normalize ALL metrics to 0-1 scale for relative comparison
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            normalized_data[metric] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized_data[metric] = [1.0] * len(values)  # All same, set to 1
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Clean labels for PCP
    labels = ['Accuracy', 'Throughput', 'Parameters', 'Speed', 'Energy']
    x_positions = np.arange(len(labels))
    
    # Colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot each model as a line connecting all metrics
    for i, result in enumerate(results):
        values = [normalized_data[metric][i] for metric in metrics]
        
        ax.plot(x_positions, values, 'o-', linewidth=3, label=result['model_name'], 
                color=colors[i], markersize=8, alpha=0.8)
    
    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical lines for each metric
    for x in x_positions:
        ax.axvline(x, color='gray', alpha=0.3, linestyle='--')
    
    plt.legend(fontsize=12, loc='upper right')
    plt.title('Model Efficiency', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Normalized Performance (Higher = Better)', fontsize=14, fontweight='bold')
    
    # Save plot
    output_path = Path(output_dir)
    pcp_file = output_path / "efficiency_parallel_coordinates.png"
    plt.tight_layout()
    plt.savefig(pcp_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Parallel coordinates plot saved: {pcp_file}")

def create_pareto_plots(results, output_dir):
    """Create Pareto frontier plots."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    
    # Plot 1: Accuracy vs Parameters
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_vals = [r['total_parameters_M'] for r in results]
    y_vals = [r['next_token_accuracy'] for r in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, result in enumerate(results):
        ax.scatter(x_vals[i], y_vals[i], s=200, c=colors[i], alpha=0.7, 
                  label=result['model_name'], edgecolors='black', linewidth=2)
        # Add model name annotations
        ax.annotate(result['model_name'], (x_vals[i], y_vals[i]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model Parameters (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Next Token Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Accuracy vs Model Size\n(Top-right is better)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Highlight Pareto frontier
    # Sort by parameters and draw line
    sorted_results = sorted(zip(x_vals, y_vals, [r['model_name'] for r in results]))
    pareto_x, pareto_y = [], []
    for x, y, name in sorted_results:
        if not pareto_y or y >= max(pareto_y):  # Pareto efficient point
            pareto_x.append(x)
            pareto_y.append(y)
    
    ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    plt.tight_layout()
    pareto_params_file = output_path / "pareto_accuracy_vs_parameters.png"
    plt.savefig(pareto_params_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Accuracy vs Energy
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_vals = [r['estimated_energy_per_inference_wh'] * 1000 for r in results]  # Convert to mWh
    y_vals = [r['next_token_accuracy'] for r in results]
    
    for i, result in enumerate(results):
        ax.scatter(x_vals[i], y_vals[i], s=200, c=colors[i], alpha=0.7, 
                  label=result['model_name'], edgecolors='black', linewidth=2)
        ax.annotate(result['model_name'], (x_vals[i], y_vals[i]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Energy per Inference (mWh)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Next Token Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Accuracy vs Energy Consumption\n(Top-left is better)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Highlight Pareto frontier (lower energy, higher accuracy is better)
    sorted_results = sorted(zip(x_vals, y_vals, [r['model_name'] for r in results]))
    pareto_x, pareto_y = [], []
    for x, y, name in sorted_results:
        # For energy, we want lower x and higher y
        if not pareto_x or (x <= min(pareto_x) and y >= max([pareto_y[j] for j, px in enumerate(pareto_x) if px >= x])):
            pareto_x.append(x)
            pareto_y.append(y)
    
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    plt.tight_layout()
    pareto_energy_file = output_path / "pareto_accuracy_vs_energy.png"
    plt.savefig(pareto_energy_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Pareto plots saved:")
    print(f"   - Accuracy vs Parameters: {pareto_params_file}")
    print(f"   - Accuracy vs Energy: {pareto_energy_file}")

def save_results(results, pareto_metrics, output_dir):
    """Save results to JSON files and create plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / "efficiency_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save Pareto efficiency metrics
    pareto_file = output_path / "pareto_efficiency.json"
    with open(pareto_file, 'w') as f:
        json.dump(pareto_metrics, f, indent=2)
    
    print(f"📊 Results saved to: {output_path}")
    print(f"   - Detailed metrics: {results_file}")
    print(f"   - Pareto efficiency: {pareto_file}")
    
    # Create plots
    print(f"\n📊 Creating efficiency plots...")
    create_parallel_coordinates_plot(results, output_dir)
    create_pareto_plots(results, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model efficiency metrics")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {config.data_dir}...")
    dataset = TimelineDataset(config.data_dir)
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Evaluate each model
    results = []
    for model_name, model_path in config.models.items():
        try:
            model_metrics = evaluate_model_efficiency(
                model_path, model_name, dataset, device, config
            )
            results.append(model_metrics)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    if not results:
        print("❌ No models were successfully evaluated!")
        return
    
    # Calculate Pareto efficiency
    print(f"\n{'='*50}")
    print("Calculating Pareto Efficiency Metrics")
    print(f"{'='*50}")
    
    pareto_metrics = calculate_pareto_efficiency(results)
    
    # Print summary
    print("\n📊 EFFICIENCY SUMMARY")
    print("-" * 50)
    for metric in pareto_metrics:
        print(f"{metric['model_name']:15} | Efficiency Score: {metric['efficiency_score']:.6f}")
    
    # Save results
    save_results(results, pareto_metrics, config.output_dir)
    
    print(f"\n✅ Efficiency evaluation complete!")

if __name__ == "__main__":
    main()
