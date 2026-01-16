#!/usr/bin/env python3
"""
Monte Carlo Evaluation for EHR-FM Models

Methodology:
1. Select N timelines from dataset (each containing special tokens from constants)
2. For EACH timeline, generate M random cut-off points
3. From each cut-off point, generate a trajectory by sampling tokens
4. Save trajectories with (token, probability) pairs
5. Find all trajectories where specific special tokens were predicted
6. Calculate probability and statistics of special token prediction

Total trajectories = N timelines × M cut-offs per timeline

This approach evaluates special token prediction across many patient contexts
to understand how well the model predicts events like DEATH, DISCHARGE, etc.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import yaml
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.datasets.base import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary
from src.utils import load_env, resolve_model_path
from omegaconf import OmegaConf
from transformers import GPT2Config
from src.model import GPT2LMNoBiasModel


def load_model(model_path):
    """Load model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'vocab_stoi' in checkpoint:
        sorted_vocab_stoi = sorted(checkpoint['vocab_stoi'].items(), key=lambda item: item[1])
        vocab_list = [item[0] for item in sorted_vocab_stoi]
        vocab = Vocabulary(vocab=vocab_list)
        model_cfg = OmegaConf.create(checkpoint['model_configs'])
    elif 'vocab' in checkpoint:
        vocab_stoi = checkpoint['vocab']
        sorted_vocab_stoi = sorted(vocab_stoi.items(), key=lambda item: item[1])
        vocab_list = [item[0] for item in sorted_vocab_stoi]
        vocab = Vocabulary(vocab=vocab_list)
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
    """Check if token is a special token (death, discharge, etc.)."""
    special_tokens = ['DEATH', 'DISCHARGE', 'HOSPITAL_ADMISSION', 'ICU_ADMISSION', 
                     'ED_ADMISSION', 'MEDS_', 'TIMELINE_', 'SOFA']
    return any(special in token_str.upper() for special in special_tokens)


def find_samples_with_special_tokens(dataset, vocab, num_samples=100):
    """Find samples that contain death or discharge tokens."""
    samples_with_special = []
    
    print(f"Scanning dataset for samples with special tokens...")
    
    for i in tqdm(range(min(num_samples * 10, len(dataset)))):
        sample = dataset[i]
        if isinstance(sample, tuple):
            context, timeline = sample
            full_sequence = torch.cat([context, timeline])
        else:
            full_sequence = sample
        
        # Decode tokens (filter out padding -100)
        token_ids = [t for t in full_sequence.tolist() if t >= 0]
        if not token_ids:
            continue
        tokens = vocab.decode(token_ids)
        
        # Check for special tokens
        special_indices = []
        for j, token in enumerate(tokens):
            if is_special_token(token):
                special_indices.append((j, token))
        
        if special_indices:
            samples_with_special.append({
                'sample_idx': i,
                'sequence': full_sequence,
                'special_tokens': special_indices,
                'length': len(full_sequence)
            })
        
        if len(samples_with_special) >= num_samples:
            break
    
    print(f"Found {len(samples_with_special)} samples with special tokens")
    return samples_with_special


def estimate_48h_tokens(vocab):
    """Estimate number of tokens that represent ~48 hours."""
    # Heuristic: assume average event is ~1-2 hours apart
    # 48 hours ≈ 24-48 events/tokens
    return 48  # Conservative estimate


def sample_next_token(logits, temperature=1.0, vocab_size=None):
    """Sample next token from logits with temperature."""
    # Apply temperature
    logits = logits / temperature
    
    # If vocab_size is provided, only use valid tokens
    if vocab_size is not None and vocab_size < len(logits):
        # Only sample from valid vocab tokens
        logits = logits[:vocab_size]
    
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Sample token
    sampled_token = torch.multinomial(probs, num_samples=1).item()
    sampled_prob = probs[sampled_token].item()
    
    # Safety: clamp token to valid range
    if vocab_size is not None:
        sampled_token = min(sampled_token, vocab_size - 1)
    
    return sampled_token, sampled_prob, probs.cpu().numpy()


def generate_trajectory(model, context, vocab, device, max_tokens=48, temperature=1.0):
    """
    Generate a trajectory by sampling tokens.
    
    Returns:
        trajectory: list of (token_id, token_str, probability) tuples
    """
    trajectory = []
    current_context = context.clone()
    vocab_size = len(vocab)
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Limit context length
            if current_context.size(0) > 512:
                current_context = current_context[-512:]
            
            # Get model prediction
            input_ids = current_context.unsqueeze(0).to(device)
            output = model(input_ids)
            logits = output.logits[0, -1, :]  # Last position
            
            # Sample next token (mask padding tokens)
            next_token, prob, all_probs = sample_next_token(logits, temperature, vocab_size=vocab_size)
            
            # Decode token
            token_str = vocab.decode([next_token])[0]
            
            # Add to trajectory
            trajectory.append({
                'token_id': next_token,
                'token_str': token_str,
                'probability': prob,
                'is_special': is_special_token(token_str)
            })
            
            # Check for special tokens - could end trajectory early
            if is_special_token(token_str):
                # Found special token, can optionally stop here
                # For now, continue to get full 48h trajectory
                pass
            
            # Add token to context for next prediction
            current_context = torch.cat([current_context, torch.tensor([next_token])])
    
    return trajectory


def generate_trajectories_batched(model, contexts, vocab, device, max_tokens=48, temperature=1.0, batch_size=32):
    """
    Generate multiple trajectories in parallel using batching.
    
    Args:
        contexts: list of context tensors
        batch_size: number of trajectories to generate in parallel
    
    Returns:
        list of trajectories
    """
    all_trajectories = []
    vocab_size = len(vocab)
    
    # Process in batches
    for batch_start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[batch_start:batch_start + batch_size]
        batch_size_actual = len(batch_contexts)
        
        # Pad contexts to same length for batching
        max_len = max(c.size(0) for c in batch_contexts)
        max_len = min(max_len, 512)  # Limit to model's max context
        
        # Create batch tensor
        batch_tensor = torch.zeros((batch_size_actual, max_len), dtype=torch.long, device=device)
        for i, ctx in enumerate(batch_contexts):
            ctx_len = min(ctx.size(0), max_len)
            batch_tensor[i, :ctx_len] = ctx[-ctx_len:].to(device)
        
        # Initialize trajectories for this batch
        batch_trajectories = [[] for _ in range(batch_size_actual)]
        current_contexts = batch_tensor.clone()
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Get model predictions for entire batch
                output = model(current_contexts)
                logits = output.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Sample next tokens for all items in batch
                logits = logits / temperature
                logits = logits[:, :vocab_size]  # Limit to valid vocab
                probs = torch.softmax(logits, dim=-1)
                
                # Sample tokens
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]
                sampled_probs = probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [batch_size]
                
                # Add to trajectories
                for i in range(batch_size_actual):
                    token_id = next_tokens[i].item()
                    prob = sampled_probs[i].item()
                    token_str = vocab.decode([token_id])[0]
                    
                    batch_trajectories[i].append({
                        'token_id': token_id,
                        'token_str': token_str,
                        'probability': prob,
                        'is_special': is_special_token(token_str)
                    })
                
                # Update contexts by appending new tokens
                current_contexts = torch.cat([current_contexts, next_tokens.unsqueeze(-1)], dim=1)
                
                # Trim to max length if needed
                if current_contexts.size(1) > 512:
                    current_contexts = current_contexts[:, -512:]
        
        all_trajectories.extend(batch_trajectories)
    
    return all_trajectories


def run_monte_carlo_simulation(model, timelines_with_special, vocab, device, 
                                num_cutoff_points_per_timeline=1000, max_tokens=48, 
                                temperature=1.0):
    """
    Run Monte Carlo simulation for trajectory generation from multiple timelines.
    
    For EACH timeline:
    1. Create num_cutoff_points_per_timeline random cut-off points
    2. Generate trajectory from each cut-off point by sampling
    3. Save trajectories with (token, probs)
    4. Find all trajectories where specific special_tokens were predicted
    5. Calculate probability of such trajectories
    
    This runs across ALL timelines to get statistics on special token prediction.
    """
    all_trajectories = []
    trajectory_stats = defaultdict(int)
    
    print(f"\nRunning Monte Carlo simulation on {len(timelines_with_special)} timelines...")
    print(f"Generating {num_cutoff_points_per_timeline} cut-off points per timeline...")
    print(f"Total trajectories to generate: {len(timelines_with_special) * num_cutoff_points_per_timeline}")
    
    # Process each timeline
    for timeline_idx, timeline_sample in enumerate(tqdm(timelines_with_special, desc="Processing timelines")):
        sequence = timeline_sample['sequence']
        special_tokens = timeline_sample['special_tokens']
        sample_idx = timeline_sample['sample_idx']
        
        # Generate random cut-off points in the timeline
        # We want to cut before the special tokens to see if model predicts them
        if len(special_tokens) > 0:
            # Get the position of the first special token
            first_special_idx = special_tokens[0][0]
            # Generate cut-off points before first special token
            if first_special_idx > 10:
                min_cutoff = 5
                max_cutoff = first_special_idx
            else:
                min_cutoff = 1
                max_cutoff = max(first_special_idx, len(sequence) // 2)
        else:
            # No special tokens, use full timeline
            min_cutoff = 5
            max_cutoff = max(10, len(sequence) - 10)
        
        # Generate random cut-off points for this timeline
        cutoff_points = []
        contexts = []
        valid_cutoff_info = []
        
        for i in range(num_cutoff_points_per_timeline):
            cutoff_point = np.random.randint(min_cutoff, max(min_cutoff + 1, max_cutoff))
            
            # Get context up to cut-off point
            context = sequence[:cutoff_point]
            
            # Filter out invalid tokens from context (padding, out of bounds)
            context = torch.tensor([t for t in context.tolist() if 0 <= t < len(vocab)])
            
            # Skip if context is too short after filtering
            if len(context) < 5:
                continue
            
            contexts.append(context)
            valid_cutoff_info.append({
                'cutoff_point': cutoff_point,
                'cutoff_idx': i,
                'context_length': len(context)
            })
        
        # Generate all trajectories for this timeline in batches
        trajectories = generate_trajectories_batched(
            model, contexts, vocab, device,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=16  # Process 16 trajectories in parallel (conservative for MoE models)
        )
        
        # Process generated trajectories
        for traj_idx, trajectory in enumerate(trajectories):
            cutoff_info = valid_cutoff_info[traj_idx]
            
            # Calculate trajectory probability (log prob for numerical stability)
            log_prob = sum(np.log(max(t['probability'], 1e-10)) for t in trajectory)
            trajectory_prob = np.exp(log_prob)
            
            # Check if trajectory contains special tokens
            special_in_trajectory = [t for t in trajectory if t['is_special']]
            
            # Store trajectory with (token, probs)
            traj_data = {
                'sample_idx': sample_idx,
                'timeline_idx': timeline_idx,
                'cutoff_point': cutoff_info['cutoff_point'],
                'cutoff_idx': cutoff_info['cutoff_idx'],
                'context_length': cutoff_info['context_length'],
                'trajectory': trajectory,  # Contains (token_id, token_str, probability, is_special)
                'trajectory_prob': trajectory_prob,
                'log_prob': log_prob,
                'special_tokens_predicted': special_in_trajectory,
                'has_special': len(special_in_trajectory) > 0,
                'ground_truth_special_tokens': special_tokens  # What special tokens actually exist in this timeline
            }
            
            all_trajectories.append(traj_data)
            
            # Update stats
            if len(special_in_trajectory) > 0:
                trajectory_stats['with_special'] += 1
                for st in special_in_trajectory:
                    trajectory_stats[f"special_{st['token_str']}"] += 1
            else:
                trajectory_stats['without_special'] += 1
    
    return all_trajectories, trajectory_stats


def analyze_trajectories(all_trajectories, trajectory_stats):
    """Analyze trajectory statistics."""
    total_trajectories = len(all_trajectories)
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nTotal trajectories: {total_trajectories}")
    print(f"Trajectories with special tokens: {trajectory_stats['with_special']}")
    print(f"Trajectories without special tokens: {trajectory_stats['without_special']}")
    
    # Probability of special token appearing
    prob_special = trajectory_stats['with_special'] / total_trajectories if total_trajectories > 0 else 0
    print(f"\nProbability of special token in trajectory: {prob_special:.4f}")
    
    # Special token breakdown
    print(f"\nSpecial token occurrences:")
    for key, count in sorted(trajectory_stats.items()):
        if key.startswith('special_'):
            token_name = key.replace('special_', '')
            prob = count / total_trajectories
            print(f"  {token_name}: {count} ({prob:.4f})")
    
    # Average trajectory probability
    avg_traj_prob = np.mean([t['trajectory_prob'] for t in all_trajectories])
    avg_log_prob = np.mean([t['log_prob'] for t in all_trajectories])
    
    print(f"\nAverage trajectory probability: {avg_traj_prob:.6e}")
    print(f"Average log probability: {avg_log_prob:.4f}")
    
    # Analyze trajectories with special tokens
    special_trajectories = [t for t in all_trajectories if t['has_special']]
    if special_trajectories:
        avg_special_prob = np.mean([t['trajectory_prob'] for t in special_trajectories])
        avg_special_log_prob = np.mean([t['log_prob'] for t in special_trajectories])
        
        print(f"\nTrajectories WITH special tokens:")
        print(f"  Count: {len(special_trajectories)}")
        print(f"  Average probability: {avg_special_prob:.6e}")
        print(f"  Average log probability: {avg_special_log_prob:.4f}")
    
    # Analyze trajectories without special tokens
    no_special_trajectories = [t for t in all_trajectories if not t['has_special']]
    if no_special_trajectories:
        avg_no_special_prob = np.mean([t['trajectory_prob'] for t in no_special_trajectories])
        avg_no_special_log_prob = np.mean([t['log_prob'] for t in no_special_trajectories])
        
        print(f"\nTrajectories WITHOUT special tokens:")
        print(f"  Count: {len(no_special_trajectories)}")
        print(f"  Average probability: {avg_no_special_prob:.6e}")
        print(f"  Average log probability: {avg_no_special_log_prob:.4f}")
    
    return {
        'total_trajectories': total_trajectories,
        'prob_special_token': prob_special,
        'avg_trajectory_prob': avg_traj_prob,
        'avg_log_prob': avg_log_prob,
        'special_token_counts': {k: v for k, v in trajectory_stats.items() if k.startswith('special_')},
        'trajectories_with_special': len(special_trajectories),
        'trajectories_without_special': len(no_special_trajectories)
    }


def save_results(all_trajectories, analysis, output_dir, model_name, original_timelines):
    """Save trajectory results to JSON and numpy files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save analysis summary (JSON - for statistics and distribution)
    summary_file = f"{output_dir}/mc_analysis_{model_name}.json"
    with open(summary_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n💾 Analysis summary saved to: {summary_file}")
    
    # Save original timelines as numpy array
    # Extract token sequences from original timelines
    timeline_sequences = []
    for timeline in original_timelines:
        seq = timeline['sequence'].cpu().numpy() if torch.is_tensor(timeline['sequence']) else np.array(timeline['sequence'])
        timeline_sequences.append(seq)
    
    timelines_file = f"{output_dir}/original_timelines_{model_name}.npy"
    np.save(timelines_file, timeline_sequences, allow_pickle=True)
    print(f"💾 Original timelines saved to: {timelines_file} ({len(timeline_sequences)} timelines)")
    
    # Save ALL trajectories with (token_id, probability) pairs as numpy array
    # Structure: list of trajectories, each with array of (token_id, prob) pairs
    trajectory_data = []
    for traj in all_trajectories:
        # Extract token_ids and probabilities from trajectory
        tokens = np.array([t['token_id'] for t in traj['trajectory']], dtype=np.int32)
        probs = np.array([t['probability'] for t in traj['trajectory']], dtype=np.float32)
        
        trajectory_data.append({
            'sample_idx': traj['sample_idx'],
            'timeline_idx': traj.get('timeline_idx', 0),
            'cutoff_point': traj['cutoff_point'],
            'cutoff_idx': traj['cutoff_idx'],
            'tokens': tokens,  # (token_id, ...)
            'probs': probs,    # (probability, ...)
            'trajectory_prob': traj['trajectory_prob'],
            'log_prob': traj['log_prob']
        })
    
    trajectories_file = f"{output_dir}/all_trajectories_{model_name}.npy"
    np.save(trajectories_file, trajectory_data, allow_pickle=True)
    print(f"💾 All trajectories saved to: {trajectories_file} ({len(trajectory_data)} trajectories)")
    print(f"   Each trajectory contains (token_ids, probabilities) arrays")
    
    # Also save a smaller JSON sample for quick inspection
    sample_size = min(100, len(all_trajectories))
    sampled_trajectories = np.random.choice(all_trajectories, sample_size, replace=False).tolist()
    
    serializable_trajectories = []
    for traj in sampled_trajectories:
        ground_truth = [
            {'position': pos, 'token': token}
            for pos, token in traj.get('ground_truth_special_tokens', [])
        ]
        
        serializable_trajectories.append({
            'sample_idx': traj['sample_idx'],
            'timeline_idx': traj.get('timeline_idx', 0),
            'cutoff_point': traj['cutoff_point'],
            'cutoff_idx': traj['cutoff_idx'],
            'context_length': traj['context_length'],
            'trajectory_prob': traj['trajectory_prob'],
            'log_prob': traj['log_prob'],
            'has_special': traj['has_special'],
            'ground_truth_special_tokens': ground_truth,
            'trajectory': [
                {
                    'token_id': t['token_id'],
                    'token_str': t['token_str'],
                    'probability': float(t['probability']),
                    'is_special': t['is_special']
                }
                for t in traj['trajectory'][:20]  # Only first 20 tokens for inspection
            ],
            'special_tokens_predicted': [
                {
                    'token_str': st['token_str'],
                    'probability': float(st['probability'])
                }
                for st in traj['special_tokens_predicted']
            ]
        })
    
    sample_json_file = f"{output_dir}/mc_trajectories_{model_name}_sample.json"
    with open(sample_json_file, 'w') as f:
        json.dump(serializable_trajectories, f, indent=2)
    print(f"💾 Sample trajectories (for inspection) saved to: {sample_json_file}")


def load_config(config_path):
    """Load Monte Carlo evaluation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main Monte Carlo evaluation function."""
    parser = argparse.ArgumentParser(description="EHR-FM Monte Carlo Trajectory Evaluation")
    parser.add_argument('--config', type=str, default='mc_eval_config.yaml',
                        help='Path to MC evaluation config file')
    parser.add_argument('--model', type=str, help='Path to specific model checkpoint')
    parser.add_argument('--num-trajectories', type=int, default=1000,
                        help='Number of random cut-off points PER timeline (generates one trajectory per cut-off)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--max-tokens', type=int, default=48,
                        help='Max tokens per trajectory (~48h)')
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file (optional)')
    args = parser.parse_args()
    
    # Load environment variables
    load_env(args.env)
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        data_dir = config['data_dir']
        output_dir = config.get('output_dir', 'mc_eval_results')
        num_samples = config.get('num_samples', 100)
        models_config = config.get('models', {})
        
        # Override args with config values if not explicitly set on command line
        if 'num_trajectories' in config and args.num_trajectories == 1000:  # 1000 is default
            args.num_trajectories = config['num_trajectories']
        if 'max_tokens' in config and args.max_tokens == 48:  # 48 is default
            args.max_tokens = config['max_tokens']
        if 'temperature' in config and args.temperature == 1.0:  # 1.0 is default
            args.temperature = config['temperature']
    else:
        print(f"Config file {args.config} not found. Using defaults...")
        data_dir = "/workspace/ehr_stuff/EHR_FM/data/tokenized_datasets/mimic_train"
        output_dir = "mc_eval_results"
        num_samples = 100
        models_config = {}
    
    print("=== Monte Carlo Trajectory Evaluation ===")
    print(f"Number of cut-off points per timeline: {args.num_trajectories}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens per trajectory: {args.max_tokens}")
    
    # Load dataset once
    print(f"\nLoading dataset from {data_dir}...")
    dataset = TimelineDataset(Path(data_dir), n_positions=512)
    print(f"✅ Dataset loaded. Size: {len(dataset)}")
    
    # Determine models to evaluate
    models_to_eval = {}
    if args.model:
        # Resolve the model path (handles S3 URIs)
        models_to_eval['custom_model'] = str(resolve_model_path(args.model))
    else:
        # Extract paths from config and resolve them
        for name, cfg in models_config.items():
            path_spec = cfg['path'] if isinstance(cfg, dict) else cfg
            models_to_eval[name] = str(resolve_model_path(path_spec))
    
    if not models_to_eval:
        print("ERROR: No models specified. Provide --model or use config file.")
        return
    
    # Load ONE model just to get the vocab for finding special tokens
    print("\nLoading first model to get vocabulary...")
    first_model_path = list(models_to_eval.values())[0]
    _, vocab, _ = load_model(first_model_path)
    print(f"✅ Vocab loaded. Size: {len(vocab)}")
    
    # Find samples with special tokens ONCE (before model loop)
    print(f"\nScanning dataset for {num_samples} timelines with special tokens...")
    samples_with_special = find_samples_with_special_tokens(
        dataset, vocab, num_samples=num_samples
    )
    
    if not samples_with_special:
        print("ERROR: No samples with special tokens found!")
        return
    
    print(f"\n✅ Found {len(samples_with_special)} timelines with special tokens")
    print(f"   Will generate {args.num_trajectories} cut-off points per timeline")
    print(f"   Total trajectories per model: {len(samples_with_special)} × {args.num_trajectories} = {len(samples_with_special) * args.num_trajectories}")
    
    # Show sample of special tokens found
    special_token_counts = defaultdict(int)
    for sample in samples_with_special[:100]:  # Sample first 100
        for _, token_str in sample['special_tokens']:
            special_token_counts[token_str] += 1
    
    print(f"\n   Special tokens found (in first 100 timelines):")
    for token, count in sorted(special_token_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      {token}: {count}")
    
    # Evaluate each model using the SAME timelines
    for model_name, model_path in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        print("\nLoading model...")
        model, vocab, device = load_model(model_path)
        print(f"✅ Model loaded. Vocab size: {len(vocab)}")
        
        # Run Monte Carlo simulation on ALL timelines
        all_trajectories, trajectory_stats = run_monte_carlo_simulation(
            model=model,
            timelines_with_special=samples_with_special,
            vocab=vocab,
            device=device,
            num_cutoff_points_per_timeline=args.num_trajectories,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Analyze trajectories
        analysis = analyze_trajectories(all_trajectories, trajectory_stats)
        
        # Save results (JSON summaries + numpy arrays)
        save_results(all_trajectories, analysis, output_dir, model_name, samples_with_special)
    
    print(f"\n✅ Monte Carlo evaluation complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()

