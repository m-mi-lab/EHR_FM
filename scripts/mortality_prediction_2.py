#!/usr/bin/env python3
"""
Mortality Prediction Upon Admission Simulation - Confidence-Weighted Trajectory Method

Methodology:
1. Find timelines with hospital admission followed by either death or discharge
2. For each patient, run 20 trajectories with different random seeds from admission
3. Each trajectory runs until it reaches death or discharge (forced outcome)
4. Weight each outcome by the model's confidence (softmax probability) at that token
5. Calculate probability as: sum(death_confidences) / (sum(death_confidences) + sum(discharge_confidences))
6. This combines trajectory sampling with model confidence for better discrimination

This is a hybrid approach between frequency-based (ETHOS) and direct probability methods.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import json
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.datasets.hospital_mortality import HospitalMortalityDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST
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
    
    return model, vocab, device, model_cfg


def generate_single_trajectory(model, context, vocab_size, device, death_token_id, discharge_token_id, 
                               max_tokens=500, temperature=1.0, seed=None, force_outcome=True):
    """
    Generate a single trajectory by sampling tokens until death or discharge.
    Returns the outcome AND the model's confidence at that outcome.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    trajectory_tokens = []
    trajectory_probs = []
    outcome_token = None
    outcome_prob = None
    outcome_position = None
    
    current_context = context.clone()
    
    # If force_outcome=True, set a much higher limit but still have a safety cutoff
    actual_max_tokens = 10000 if force_outcome else max_tokens
    
    with torch.no_grad():
        for step in range(actual_max_tokens):
            # Limit context length
            if current_context.size(0) > 512:
                current_context = current_context[-512:]
            
            input_ids = current_context.unsqueeze(0).to(device)
            output = model(input_ids)
            logits = output.logits[0, -1, :]  # Last position
            
            logits = logits / temperature
            logits = logits[:vocab_size]
            
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_prob = probs[next_token].item()
            
            trajectory_tokens.append(next_token)
            trajectory_probs.append(next_prob)
            
            if next_token == death_token_id:
                outcome_token = ST.DEATH
                outcome_prob = next_prob
                outcome_position = step
                break  # Stop trajectory at outcome
            elif next_token == discharge_token_id:
                outcome_token = ST.DISCHARGE
                outcome_prob = next_prob
                outcome_position = step
                break  # Stop trajectory at outcome
            
            current_context = torch.cat([current_context, torch.tensor([next_token])])
    
    return {
        'tokens': trajectory_tokens,
        'probs': trajectory_probs,
        'outcome_token': outcome_token,
        'outcome_prob': outcome_prob,
        'outcome_position': outcome_position,
        'trajectory_length': len(trajectory_tokens)
    }


def save_patient_cache(patients_data, cache_path):
    """Save patient data to cache file."""
    cache_data = {
        'patients': [
            {
                'context': p['context'].cpu().tolist() if isinstance(p['context'], torch.Tensor) else p['context'],
                'ground_truth_outcome': p['ground_truth_outcome'],
                'dataset_idx': p['dataset_idx']
            }
            for p in patients_data
        ]
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    print(f"💾 Saved patient cache to: {cache_path}")


def load_patient_cache(cache_path):
    """Load patient data from cache file."""
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    
    patients = [
        {
            'context': torch.tensor(p['context']),
            'ground_truth_outcome': p['ground_truth_outcome'],
            'dataset_idx': p['dataset_idx']
        }
        for p in cache_data['patients']
    ]
    print(f"✅ Loaded {len(patients)} patients from cache: {cache_path}")
    return patients


def run_mortality_prediction_for_patient(model, context, ground_truth_outcome, vocab, device, 
                                         num_trajectories=10, max_tokens=500, temperature=1.0,
                                         base_seed=42, discard_unfinished=False):
    """
    Run mortality prediction for a single patient using confidence-weighted trajectories.
    
    NEW: Instead of just counting outcomes, we weight each outcome by the model's confidence.
    """
    vocab_size = len(vocab)

    if ST.DEATH not in vocab.stoi:
        raise ValueError(f"Death token {ST.DEATH} not found in vocabulary")
    if ST.DISCHARGE not in vocab.stoi:
        raise ValueError(f"Discharge token {ST.DISCHARGE} not found in vocabulary")
    death_token_id = vocab.stoi.get(ST.DEATH)
    discharge_token_id = vocab.stoi.get(ST.DISCHARGE)
    
    trajectories = []
    death_count = 0
    discharge_count = 0
    no_outcome_count = 0
    
    # NEW: Track confidence-weighted sums
    death_confidence_sum = 0.0
    discharge_confidence_sum = 0.0
    
    # Run multiple trajectories with different seeds
    for traj_idx in range(num_trajectories):
        seed = base_seed + traj_idx
        
        trajectory = generate_single_trajectory(
            model=model,
            context=context,
            vocab_size=vocab_size,
            device=device,
            death_token_id=death_token_id,
            discharge_token_id=discharge_token_id,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            force_outcome=True  # Always generate until outcome is reached
        )
        
        trajectories.append(trajectory)
        
        # Count outcomes (for compatibility)
        if trajectory['outcome_token'] == ST.DEATH:
            death_count += 1
            death_confidence_sum += trajectory['outcome_prob']
        elif trajectory['outcome_token'] == ST.DISCHARGE:
            discharge_count += 1
            discharge_confidence_sum += trajectory['outcome_prob']
        else:
            no_outcome_count += 1
    
    # OLD: Frequency-based probability
    if discard_unfinished:
        total_for_prob = death_count + discharge_count
    else:
        total_for_prob = num_trajectories
    
    if total_for_prob > 0:
        mortality_probability_freq = death_count / total_for_prob
        survival_probability_freq = discharge_count / total_for_prob
    else:
        mortality_probability_freq = 0.0
        survival_probability_freq = 0.0
    
    # NEW: Confidence-weighted probability
    total_confidence = death_confidence_sum + discharge_confidence_sum
    if total_confidence > 0:
        mortality_probability = death_confidence_sum / total_confidence
        survival_probability = discharge_confidence_sum / total_confidence
    else:
        mortality_probability = 0.5
        survival_probability = 0.5
    
    # Determine predicted outcome based on confidence-weighted probability
    if mortality_probability > 0.5:
        predicted_outcome = ST.DEATH
        prediction_confidence = mortality_probability
    else:
        predicted_outcome = ST.DISCHARGE
        prediction_confidence = survival_probability
    
    is_correct = (predicted_outcome == ground_truth_outcome)
    
    return {
        'trajectories': trajectories,
        'num_trajectories': num_trajectories,
        'death_count': death_count,
        'discharge_count': discharge_count,
        'no_outcome_count': no_outcome_count,
        'death_confidence_sum': death_confidence_sum,
        'discharge_confidence_sum': discharge_confidence_sum,
        'mortality_probability': mortality_probability,  # Confidence-weighted
        'mortality_probability_freq': mortality_probability_freq,  # Frequency-based
        'survival_probability': survival_probability,
        'survival_probability_freq': survival_probability_freq,
        'predicted_outcome': predicted_outcome,
        'ground_truth_outcome': ground_truth_outcome,
        'prediction_confidence': prediction_confidence,
        'is_correct': is_correct
    }


def run_mortality_prediction_simulation(model, dataset, vocab, device, model_cfg,
                                        num_patients=100, num_trajectories_per_patient=10,
                                        max_tokens=500, temperature=1.0, base_seed=42,
                                        target_death_patients=None, cached_patients=None,
                                        discard_unfinished=False):
    """
    Run mortality prediction simulation on multiple patients.
    """
    print(f"\n{'='*70}")
    print(f"MORTALITY PREDICTION SIMULATION - CONFIDENCE-WEIGHTED METHOD")
    print(f"{'='*70}")
    print(f"Num patients: {num_patients}")
    if target_death_patients is not None:
        print(f"Target death patients: {target_death_patients}")
        print(f"Target discharge patients: {num_patients - target_death_patients}")
    print(f"Trajectories per patient: {num_trajectories_per_patient}")
    print(f"Method: Confidence-weighted (trajectory outcomes weighted by model confidence)")
    print(f"Max tokens per trajectory: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Total trajectories: {num_patients * num_trajectories_per_patient}")
    
    patient_results = []
    
    death_token_id = vocab.stoi.get(ST.DEATH, -1)
    discharge_token_id = vocab.stoi.get(ST.DISCHARGE, -1)
    
    print(f"\nDeath token: {ST.DEATH} (ID: {death_token_id})")
    print(f"Discharge token: {ST.DISCHARGE} (ID: {discharge_token_id})")
    
    print(f"\nProcessing {num_patients} patients...")
    
    if cached_patients is not None:
        print(f"📦 Using cached patient data ({len(cached_patients)} patients)")
        patients_to_process = cached_patients
        new_patient_cache = None
    else:
        print(f"📋 Loading patients from dataset...")
        patients_to_process = []
        patients_processed = 0
        death_patients_found = 0
        discharge_patients_found = 0
        dataset_idx = 0
        
        while patients_processed < num_patients and dataset_idx < len(dataset):
            try:
                context_tensor, metadata = dataset[dataset_idx]
                dataset_idx += 1
                
                ground_truth = metadata.get('expected', None)
                if ground_truth is None:
                    continue
                
                # within max_tokens limit?
                true_token_dist = metadata.get('true_token_dist', None)
                if true_token_dist is not None and true_token_dist > max_tokens:
                    continue
                
                context_tensor = context_tensor[context_tensor >= 0]
                
                if len(context_tensor) < 5:
                    continue
                
                if ground_truth == ST.DEATH:
                    ground_truth_outcome = ST.DEATH
                elif ground_truth == ST.DISCHARGE:
                    ground_truth_outcome = ST.DISCHARGE
                else:
                    continue
                
                if target_death_patients is not None:
                    if ground_truth_outcome == ST.DEATH:
                        if death_patients_found >= target_death_patients:
                            continue
                        death_patients_found += 1
                    else:  # DISCHARGE
                        target_discharge = num_patients - target_death_patients
                        if discharge_patients_found >= target_discharge:
                            continue
                        discharge_patients_found += 1
                
                patients_to_process.append({
                    'context': context_tensor,
                    'ground_truth_outcome': ground_truth_outcome,
                    'dataset_idx': dataset_idx - 1,
                    'patient_id': metadata.get('patient_id', dataset_idx - 1),
                    'hadm_id': metadata.get('hadm_id', None)
                })
                patients_processed += 1
                
            except Exception as e:
                print(f"⚠️  Error loading patient {dataset_idx}: {e}")
                continue
        
        new_patient_cache = patients_to_process
        print(f"✅ Loaded {len(patients_to_process)} patients from dataset")
    
    patients_processed = 0
    death_patients_found = 0
    discharge_patients_found = 0
    
    with tqdm(total=len(patients_to_process), desc="Patients") as pbar:
        for patient_data in patients_to_process:
            try:
                context_tensor = patient_data['context']
                ground_truth_outcome = patient_data['ground_truth_outcome']
                
                patient_result = run_mortality_prediction_for_patient(
                    model=model,
                    context=context_tensor,
                    ground_truth_outcome=ground_truth_outcome,
                    vocab=vocab,
                    device=device,
                    num_trajectories=num_trajectories_per_patient,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    base_seed=base_seed + patients_processed * 1000,  # Unique seed per patient
                    discard_unfinished=discard_unfinished
                )
                
                # Add metadata
                patient_result['patient_id'] = patient_data.get('patient_id', patients_processed)
                patient_result['hadm_id'] = patient_data.get('hadm_id', None)
                patient_result['context_length'] = len(context_tensor)
                
                patient_results.append(patient_result)
                patients_processed += 1
                
                # Update class counts
                if ground_truth_outcome == ST.DEATH:
                    death_patients_found += 1
                else:
                    discharge_patients_found += 1
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing patient: {e}")
                continue
    
    print(f"\n✅ Processed {patients_processed} patients")
    print(f"   Death patients: {death_patients_found}")
    print(f"   Discharge patients: {discharge_patients_found}")
    
    # Calculate aggregate statistics
    from mortality_prediction import calculate_aggregate_statistics
    stats = calculate_aggregate_statistics(patient_results)
    
    return {
        'patient_results': patient_results,
        'statistics': stats,
        'config': {
            'num_patients': patients_processed,
            'num_trajectories_per_patient': num_trajectories_per_patient,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'base_seed': base_seed,
            'method': 'confidence_weighted'
        },
        'new_patient_cache': new_patient_cache  # For caching newly loaded patients
    }


def print_statistics(stats):
    """Print statistics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"MORTALITY PREDICTION RESULTS - CONFIDENCE-WEIGHTED METHOD")
    print(f"{'='*70}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total patients: {stats['total_patients']}")
    print(f"   Total trajectories: {stats['total_trajectories']}")
    print(f"   Method: Confidence-weighted (outcomes weighted by model confidence)")
    
    print(f"\n📈 AUROC Metrics:")
    print(f"   AUROC (sklearn): {stats['auroc']:.4f}")
    if stats.get('auroc_ci_lower') and stats.get('auroc_ci_upper'):
        print(f"   AUROC (bootstrap mean): {stats['auroc_mean']:.4f} (95% CI: {stats['auroc_ci_lower']:.4f}-{stats['auroc_ci_upper']:.4f})")
        print(f"   AUROC std: {stats['auroc_std']:.4f}")
    if stats.get('auroc_gaussian_fit') and stats['auroc_gaussian_fit'] > 0:
        print(f"   AUROC Gaussian Fit (ETHOS): {stats['auroc_gaussian_fit']:.4f}")
        if stats.get('gaussian_params'):
            params = stats['gaussian_params']
            print(f"      d' = {params['d_prime']:.4f}, σ_ratio = {params['sigma_ratio']:.4f}, R² = {params['r_squared']:.4f}")
    print(f"   AUPRC (Area Under PR Curve): {stats['auprc']:.4f}")
    
    print(f"\n🎯 TRAJECTORY-LEVEL ACCURACY (Averaged per Patient):")
    print(f"   Death prediction accuracy: {stats['avg_death_traj_accuracy']:.2%}")
    print(f"   Non-death prediction accuracy: {stats['avg_discharge_traj_accuracy']:.2%}")
    
    print(f"\n💀 Death Patients (n={stats['death_patients']}):")
    print(f"   Total trajectories: {stats['death_patient_trajectories_total']}")
    print(f"   Predicted DEATH: {stats['death_patient_death_traj']} ({stats['death_patient_death_traj']/stats['death_patient_trajectories_total']:.2%})")
    print(f"   Predicted DISCHARGE: {stats['death_patient_discharge_traj']} ({stats['death_patient_discharge_traj']/stats['death_patient_trajectories_total']:.2%})")
    print(f"   No outcome: {stats['death_patient_no_outcome_traj']} ({stats['death_patient_no_outcome_traj']/stats['death_patient_trajectories_total']:.2%})")
    print(f"   Average accuracy per patient: {stats['avg_death_traj_accuracy']:.2%}")
    
    print(f"\n✅ Discharge Patients (n={stats['discharge_patients']}):")
    print(f"   Total trajectories: {stats['discharge_patient_trajectories_total']}")
    print(f"   Predicted DEATH: {stats['discharge_patient_death_traj']} ({stats['discharge_patient_death_traj']/stats['discharge_patient_trajectories_total']:.2%})")
    print(f"   Did NOT predict DEATH: {stats['discharge_patient_non_death_traj']} ({stats['discharge_patient_non_death_traj']/stats['discharge_patient_trajectories_total']:.2%})")
    print(f"      └─ Predicted DISCHARGE: {stats['discharge_patient_discharge_traj']}")
    print(f"      └─ No outcome: {stats['discharge_patient_no_outcome_traj']}")
    print(f"   Average accuracy per patient: {stats['avg_discharge_traj_accuracy']:.2%}")


def save_results(results, output_dir, model_name):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create serializable version (remove trajectory details for brevity)
    serializable_results = {
        'statistics': results['statistics'],
        'config': results['config'],
        'patient_summary': []
    }
    
    for patient in results['patient_results']:
        patient_summary = {
            'patient_id': int(patient['patient_id']) if patient.get('patient_id') is not None else None,
            'hadm_id': int(patient['hadm_id']) if patient.get('hadm_id') is not None else None,
            'context_length': patient['context_length'],
            'ground_truth_outcome': patient['ground_truth_outcome'],
            'predicted_outcome': patient['predicted_outcome'],
            'is_correct': patient['is_correct'],
            'mortality_probability': patient['mortality_probability'],
            'mortality_probability_freq': patient.get('mortality_probability_freq', None),
            'survival_probability': patient['survival_probability'],
            'prediction_confidence': patient['prediction_confidence'],
            'death_count': patient['death_count'],
            'discharge_count': patient['discharge_count'],
            'no_outcome_count': patient['no_outcome_count'],
            'death_confidence_sum': patient.get('death_confidence_sum', None),
            'discharge_confidence_sum': patient.get('discharge_confidence_sum', None)
        }
        serializable_results['patient_summary'].append(patient_summary)
    
    # Save to JSON
    output_file = f"{output_dir}/mortality_prediction_{model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Also save full trajectories to numpy for detailed analysis
    trajectory_file = f"{output_dir}/mortality_trajectories_{model_name}.npy"
    trajectory_data = []
    
    for patient in results['patient_results']:
        patient_traj_data = {
            'patient_id': int(patient['patient_id']) if patient.get('patient_id') is not None else None,
            'ground_truth': patient['ground_truth_outcome'],
            'predicted': patient['predicted_outcome'],
            'trajectories': [
                {
                    'tokens': np.array(t['tokens'], dtype=np.int32),
                    'probs': np.array(t['probs'], dtype=np.float32),
                    'outcome_token': t['outcome_token'],
                    'outcome_prob': t['outcome_prob'],
                    'outcome_position': t['outcome_position']
                }
                for t in patient['trajectories']
            ]
        }
        trajectory_data.append(patient_traj_data)
    
    np.save(trajectory_file, trajectory_data, allow_pickle=True)
    print(f"💾 Trajectories saved to: {trajectory_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Mortality Prediction - Confidence-Weighted Method")
    parser.add_argument('--config', type=str, default='mortality_pred_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to single model checkpoint (overrides config)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to tokenized dataset (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    parser.add_argument('--num-patients', type=int, default=None,
                        help='Number of patients to evaluate (overrides config)')
    parser.add_argument('--num-trajectories', type=int, default=None,
                        help='Number of trajectories per patient (overrides config)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Max tokens per trajectory (overrides config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (overrides config)')
    parser.add_argument('--patient-cache', type=str, default=None,
                        help='Path to save/load cached patient data')
    parser.add_argument('--discard-unfinished-trajectories', action='store_true',
                        help='Exclude trajectories without death/discharge outcome')
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file (optional)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env(args.env)
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {args.config}")
    else:
        print(f"⚠️  Config file {args.config} not found, using defaults")
    
    # Override config with command line args
    data_dir = args.data_dir or config.get('data_dir', '/workspace/ehr_stuff/EHR_FM/data/tokenized_datasets/mimic_train')
    output_dir = args.output_dir or config.get('output_dir', 'mortality_prediction_results')
    num_patients = args.num_patients or config.get('num_patients', 100)
    num_trajectories = args.num_trajectories or config.get('num_trajectories_per_patient', 10)
    max_tokens = args.max_tokens or config.get('max_tokens', 500)
    temperature = args.temperature or config.get('temperature', 1.0)
    base_seed = args.seed or config.get('base_seed', 42)
    target_death_patients = config.get('target_death_patients', None)
    
    print("="*70)
    print("MORTALITY PREDICTION - CONFIDENCE-WEIGHTED METHOD")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of patients: {num_patients}")
    if target_death_patients is not None:
        print(f"Target death patients: {target_death_patients}")
        print(f"Target discharge patients: {num_patients - target_death_patients}")
    print(f"Trajectories per patient: {num_trajectories}")
    print(f"Max tokens per trajectory: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Base seed: {base_seed}")
    
    # Determine models to evaluate
    models_to_eval = {}
    if args.model:
        models_to_eval['custom_model'] = str(resolve_model_path(args.model))
    elif 'models' in config and config['models']:
        for name, model_config in config['models'].items():
            path_spec = model_config['path'] if isinstance(model_config, dict) else model_config
            models_to_eval[name] = str(resolve_model_path(path_spec))
    else:
        print("❌ ERROR: No models specified. Use --model or add models to config file.")
        return
    
    print(f"\n📋 Models to evaluate: {list(models_to_eval.keys())}")
    
    # Load dataset once
    print(f"\nLoading dataset from {data_dir}...")
    first_model_path = list(models_to_eval.values())[0]
    _, _, _, temp_model_cfg = load_model(first_model_path)
    
    dataset = HospitalMortalityDataset(
        input_dir=Path(data_dir),
        n_positions=temp_model_cfg.n_positions
    )
    print(f"✅ Dataset loaded. Size: {len(dataset)}")
    
    # Load or prepare patient cache
    cached_patients = None
    if args.patient_cache and Path(args.patient_cache).exists():
        cached_patients = load_patient_cache(args.patient_cache)
    
    # Evaluate each model
    all_model_results = {}
    patient_cache_to_save = None
    
    for model_name, model_path in models_to_eval.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*70}")
        print(f"Model path: {model_path}")
        
        # Load model
        print("\nLoading model...")
        model, vocab, device, model_cfg = load_model(model_path)
        print(f"✅ Model loaded. Vocab size: {len(vocab)}")
        
        # Run simulation
        results = run_mortality_prediction_simulation(
            model=model,
            dataset=dataset,
            vocab=vocab,
            device=device,
            model_cfg=model_cfg,
            num_patients=num_patients,
            num_trajectories_per_patient=num_trajectories,
            max_tokens=max_tokens,
            temperature=temperature,
            base_seed=base_seed,
            target_death_patients=target_death_patients,
            cached_patients=cached_patients,
            discard_unfinished=args.discard_unfinished_trajectories
        )
        
        # Save patient cache from first model if not already cached
        if patient_cache_to_save is None and results.get('new_patient_cache') is not None:
            patient_cache_to_save = results['new_patient_cache']
        
        # Print statistics
        print_statistics(results['statistics'])
        
        # Save results
        save_results(results, output_dir, model_name)
        
        # Store for comparison
        all_model_results[model_name] = results['statistics']
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save patient cache if we created one
    if args.patient_cache and patient_cache_to_save is not None:
        save_patient_cache(patient_cache_to_save, args.patient_cache)
    
    # Print comparison if multiple models
    if len(all_model_results) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"\n{'Model':<20} {'AUROC':<12} {'Death Traj Acc':<18} {'Non-Death Traj Acc':<18}")
        print("-" * 70)
        for model_name, stats in all_model_results.items():
            print(f"{model_name:<20} {stats['auroc']:>10.4f}  {stats['avg_death_traj_accuracy']:>16.2%}  {stats['avg_discharge_traj_accuracy']:>16.2%}")
    
    print(f"\n✅ Mortality prediction simulation complete!")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()


