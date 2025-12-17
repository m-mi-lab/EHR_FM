#!/usr/bin/env python3
"""
Mortality Prediction Upon Admission Simulation

Methodology:
1. Find timelines with hospital admission followed by either death or discharge
2. For each patient, run 10 trajectories with different random seeds from admission
3. Each trajectory runs until it reaches death or discharge (up to 500 tokens max)
4. Calculate probability of correctly predicting mortality vs survival
5. Evaluate on 100 patients total
6. Calculate AUROC and other classification metrics

This simulates real-world mortality prediction where we predict outcomes at admission time.
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
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.datasets.hospital_mortality import HospitalMortalityDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST
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
                               max_tokens=500, temperature=1.0, seed=None):
    """
    Generate a single trajectory by sampling tokens until death or discharge.
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
    
    with torch.no_grad():
        for step in range(max_tokens):
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
    Run mortality prediction for a single patient using multiple trajectories.
    """
    vocab_size = len(vocab)
    death_token_id = vocab.stoi.get(ST.DEATH, -1)
    discharge_token_id = vocab.stoi.get(ST.DISCHARGE, -1)
    
    trajectories = []
    death_count = 0
    discharge_count = 0
    no_outcome_count = 0
    
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
            seed=seed
        )
        
        trajectories.append(trajectory)
        
        # Count outcomes
        if trajectory['outcome_token'] == ST.DEATH:
            death_count += 1
        elif trajectory['outcome_token'] == ST.DISCHARGE:
            discharge_count += 1
        else:
            no_outcome_count += 1
    
    if discard_unfinished:
        total_for_prob = death_count + discharge_count
    else:
        total_for_prob = num_trajectories
    
    if total_for_prob > 0:
        mortality_probability = death_count / total_for_prob
        survival_probability = discharge_count / total_for_prob
    else:
        mortality_probability = 0.0
        survival_probability = 0.0
    
    # only for threshold based metrics --- confusion matrix, accuracy
    if death_count > discharge_count:
        predicted_outcome = ST.DEATH
        prediction_confidence = mortality_probability
    elif discharge_count > death_count:
        predicted_outcome = ST.DISCHARGE
        prediction_confidence = survival_probability
    else:
        predicted_outcome = None  # Tie
        prediction_confidence = 0.5
    
    is_correct = (predicted_outcome == ground_truth_outcome)
    
    return {
        'trajectories': trajectories,
        'num_trajectories': num_trajectories,
        'death_count': death_count,
        'discharge_count': discharge_count,
        'no_outcome_count': no_outcome_count,
        'mortality_probability': mortality_probability,
        'survival_probability': survival_probability,
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
    print(f"MORTALITY PREDICTION SIMULATION")
    print(f"{'='*70}")
    print(f"Num patients: {num_patients}")
    if target_death_patients is not None:
        print(f"Target death patients: {target_death_patients}")
        print(f"Target discharge patients: {num_patients - target_death_patients}")
    print(f"Trajectories per patient: {num_trajectories_per_patient}")
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
    stats = calculate_aggregate_statistics(patient_results)
    
    return {
        'patient_results': patient_results,
        'statistics': stats,
        'config': {
            'num_patients': patients_processed,
            'num_trajectories_per_patient': num_trajectories_per_patient,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'base_seed': base_seed
        },
        'new_patient_cache': new_patient_cache  # For caching newly loaded patients
    }


def calculate_aggregate_statistics(patient_results):
    """Calculate aggregate statistics across all patients, including AUROC."""
    
    if not patient_results:
        return {}
    
    correct_predictions = sum(1 for p in patient_results if p['is_correct'])
    total_patients = len(patient_results)
    accuracy = correct_predictions / total_patients if total_patients > 0 else 0
    
    # Separate by ground truth outcome
    death_patients = [p for p in patient_results if p['ground_truth_outcome'] == ST.DEATH]
    discharge_patients = [p for p in patient_results if p['ground_truth_outcome'] == ST.DISCHARGE]
    
    # Prepare data for AUROC calculation
    # Ground truth: 1 for death, 0 for discharge
    y_true = np.array([1 if p['ground_truth_outcome'] == ST.DEATH else 0 
                       for p in patient_results])
    y_score = np.array([p['mortality_probability'] for p in patient_results])
    
    if death_patients:
        death_correct = sum(1 for p in death_patients if p['is_correct'])
        death_accuracy = death_correct / len(death_patients)
        death_avg_mortality_prob = np.mean([p['mortality_probability'] for p in death_patients])
    else:
        death_accuracy = 0
        death_avg_mortality_prob = 0
    
    if discharge_patients:
        discharge_correct = sum(1 for p in discharge_patients if p['is_correct'])
        discharge_accuracy = discharge_correct / len(discharge_patients)
        discharge_avg_survival_prob = np.mean([p['survival_probability'] for p in discharge_patients])
    else:
        discharge_accuracy = 0
        discharge_avg_survival_prob = 0
    
    # Average probabilities across all patients
    avg_mortality_prob = np.mean([p['mortality_probability'] for p in patient_results])
    avg_survival_prob = np.mean([p['survival_probability'] for p in patient_results])
    avg_confidence = np.mean([p['prediction_confidence'] for p in patient_results])
    
    # Trajectory outcome statistics (TOTAL across all patients)
    total_trajectories = sum(p['num_trajectories'] for p in patient_results)
    total_death_outcomes = sum(p['death_count'] for p in patient_results)
    total_discharge_outcomes = sum(p['discharge_count'] for p in patient_results)
    total_no_outcomes = sum(p['no_outcome_count'] for p in patient_results)
    
    # TRAJECTORY-LEVEL ACCURACY: What we actually care about!
    # For each death patient: what % of their 25 trajectories predicted DEATH token?
    # Then average across all death patients
    death_patient_traj_accuracies = []
    for p in death_patients:
        # % of this patient's trajectories that predicted death
        accuracy = p['death_count'] / p['num_trajectories'] if p['num_trajectories'] > 0 else 0
        death_patient_traj_accuracies.append(accuracy)
    
    avg_death_traj_accuracy = np.mean(death_patient_traj_accuracies) if death_patient_traj_accuracies else 0
    
    # For each discharge patient: what % of their 25 trajectories did NOT predict DEATH token?
    # (= discharge + no outcome)
    # Then average across all discharge patients
    discharge_patient_traj_accuracies = []
    for p in discharge_patients:
        # % of this patient's trajectories that did NOT predict death
        non_death_count = p['discharge_count'] + p['no_outcome_count']
        accuracy = non_death_count / p['num_trajectories'] if p['num_trajectories'] > 0 else 0
        discharge_patient_traj_accuracies.append(accuracy)
    
    avg_discharge_traj_accuracy = np.mean(discharge_patient_traj_accuracies) if discharge_patient_traj_accuracies else 0
    
    # Aggregate counts for reporting
    death_patient_trajectories_total = sum(p['num_trajectories'] for p in death_patients)
    death_patient_death_traj = sum(p['death_count'] for p in death_patients)
    death_patient_discharge_traj = sum(p['discharge_count'] for p in death_patients)
    death_patient_no_outcome_traj = sum(p['no_outcome_count'] for p in death_patients)
    
    discharge_patient_trajectories_total = sum(p['num_trajectories'] for p in discharge_patients)
    discharge_patient_death_traj = sum(p['death_count'] for p in discharge_patients)
    discharge_patient_discharge_traj = sum(p['discharge_count'] for p in discharge_patients)
    discharge_patient_no_outcome_traj = sum(p['no_outcome_count'] for p in discharge_patients)
    discharge_patient_non_death_traj = discharge_patient_discharge_traj + discharge_patient_no_outcome_traj
    
    # Calculate AUROC and related metrics
    try:
        # Only calculate if we have both classes
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_score)
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
            auprc = auc(recall, precision)
        else:
            auroc = 0.0
            auprc = 0.0
            fpr, tpr, thresholds = None, None, None
            precision, recall, pr_thresholds = None, None, None
    except Exception as e:
        print(f"Warning: Could not calculate AUROC: {e}")
        auroc = 0.0
        auprc = 0.0
        fpr, tpr, thresholds = None, None, None
        precision, recall, pr_thresholds = None, None, None
    
    stats = {
        # Patient-level stats
        'total_patients': total_patients,
        'correct_predictions': correct_predictions,
        'overall_accuracy': accuracy,
        
        # AUROC metrics
        'auroc': auroc,
        'auprc': auprc,
        
        'death_patients': len(death_patients),
        'death_accuracy': death_accuracy,
        'death_avg_mortality_prob': death_avg_mortality_prob,
        
        'discharge_patients': len(discharge_patients),
        'discharge_accuracy': discharge_accuracy,
        'discharge_avg_survival_prob': discharge_avg_survival_prob,
        
        'avg_mortality_prob': avg_mortality_prob,
        'avg_survival_prob': avg_survival_prob,
        'avg_confidence': avg_confidence,
        
        # Trajectory-level stats (THE IMPORTANT ONES!)
        'total_trajectories': total_trajectories,
        'total_death_outcomes': total_death_outcomes,
        'total_discharge_outcomes': total_discharge_outcomes,
        'total_no_outcomes': total_no_outcomes,
        
        # Death patient trajectory stats
        'death_patient_trajectories_total': death_patient_trajectories_total,
        'death_patient_death_traj': death_patient_death_traj,
        'death_patient_discharge_traj': death_patient_discharge_traj,
        'death_patient_no_outcome_traj': death_patient_no_outcome_traj,
        'avg_death_traj_accuracy': avg_death_traj_accuracy,  # Average % across patients
        
        # Discharge patient trajectory stats
        'discharge_patient_trajectories_total': discharge_patient_trajectories_total,
        'discharge_patient_death_traj': discharge_patient_death_traj,
        'discharge_patient_discharge_traj': discharge_patient_discharge_traj,
        'discharge_patient_no_outcome_traj': discharge_patient_no_outcome_traj,
        'discharge_patient_non_death_traj': discharge_patient_non_death_traj,
        'avg_discharge_traj_accuracy': avg_discharge_traj_accuracy,  # Average % across patients
    }
    
    return stats


def print_statistics(stats):
    """Print statistics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"MORTALITY PREDICTION RESULTS")
    print(f"{'='*70}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total patients: {stats['total_patients']}")
    print(f"   Total trajectories: {stats['total_trajectories']}")
    
    print(f"\n📈 AUROC Metrics:")
    print(f"   AUROC (Area Under ROC): {stats['auroc']:.4f}")
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
            'survival_probability': patient['survival_probability'],
            'prediction_confidence': patient['prediction_confidence'],
            'death_count': patient['death_count'],
            'discharge_count': patient['discharge_count'],
            'no_outcome_count': patient['no_outcome_count']
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


def load_all_results(results_dir):
    """Load all mortality prediction results for plotting."""
    results_path = Path(results_dir)
    all_results = {}
    
    # Find all JSON files
    for json_file in results_path.glob('mortality_prediction_*.json'):
        model_name = json_file.stem.replace('mortality_prediction_', '')
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract patient-level data for plotting
        patients = data['patient_summary']
        
        # Create binary labels and scores
        # Check for both possible death token names
        y_true = np.array([1 if ('DEATH' in p['ground_truth_outcome'].upper()) else 0 
                          for p in patients])
        y_score = np.array([p['mortality_probability'] for p in patients])
        
        all_results[model_name] = {
            'statistics': data['statistics'],
            'config': data['config'],
            'patients': patients,
            'y_true': y_true,
            'y_score': y_score
        }
    
    return all_results


def bootstrap_auc(y_true, y_score, n_bootstraps=1000, confidence_level=0.95):
    """Calculate AUC confidence intervals using bootstrapping."""
    np.random.seed(42)
    n_samples = len(y_true)
    aucs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Check if bootstrap has both classes
        if len(np.unique(y_true_boot)) > 1:
            try:
                auc_boot = roc_auc_score(y_true_boot, y_score_boot)
                aucs.append(auc_boot)
            except:
                pass
    
    if len(aucs) == 0:
        return None, None
    
    aucs = np.array(aucs)
    alpha = 1 - confidence_level
    lower = np.percentile(aucs, alpha/2 * 100)
    upper = np.percentile(aucs, (1 - alpha/2) * 100)
    
    return lower, upper


def find_clinically_relevant_thresholds(y_true, y_score, fpr, tpr, thresholds):
    """Find multiple clinically relevant thresholds (ETHOS-style).
    
    Returns:
        list of (name, threshold, fpr, tpr) tuples
    """
    clinical_thresholds = []
    
    # 1. High Sensitivity (95% sensitivity) - catch most deaths
    target_sens = 0.95
    idx_high_sens = np.argmin(np.abs(tpr - target_sens))
    if idx_high_sens < len(thresholds):
        clinical_thresholds.append((
            'High Sens (95%)',
            thresholds[idx_high_sens],
            fpr[idx_high_sens],
            tpr[idx_high_sens]
        ))
    
    # 2. Balanced (Youden's J) - optimal trade-off
    j_scores = tpr - fpr
    idx_balanced = np.argmax(j_scores)
    if idx_balanced < len(thresholds):
        clinical_thresholds.append((
            'Balanced',
            thresholds[idx_balanced],
            fpr[idx_balanced],
            tpr[idx_balanced]
        ))
    
    # 3. High Specificity (95% specificity) - minimize false alarms
    specificity = 1 - fpr
    target_spec = 0.95
    idx_high_spec = np.argmin(np.abs(specificity - target_spec))
    if idx_high_spec < len(thresholds):
        clinical_thresholds.append((
            'High Spec (95%)',
            thresholds[idx_high_spec],
            fpr[idx_high_spec],
            tpr[idx_high_spec]
        ))
    
    return clinical_thresholds


def plot_roc_curves(all_results, output_dir):
    """Plot ROC curves for all models with 95% CI and optimal thresholds (ETHOS-style)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        y_true = results['y_true']
        y_score = results['y_score']
        
        print(f"\nPlotting {model_name}:")
        print(f"  y_true: {len(y_true)} samples, {y_true.sum()} deaths, {(y_true==0).sum()} survived")
        print(f"  y_score range: [{y_score.min():.3f}, {y_score.max():.3f}]")
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            print(f"  ⚠️ WARNING: Only one class present, skipping {model_name}")
            continue
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)
        
        print(f"  AUROC: {auroc:.4f}")
        print(f"  FPR points: {len(fpr)}, TPR points: {len(tpr)}")
        
        # Calculate 95% CI for AUC
        lower_ci, upper_ci = bootstrap_auc(y_true, y_score)
        
        # Get prevalence and case counts
        n_total = len(y_true)
        n_positive = np.sum(y_true == 1)
        prevalence = n_positive / n_total
        
        # Plot ROC curve
        color = colors[idx % len(colors)]
        if lower_ci is not None and upper_ci is not None:
            label = f'{model_name}: AUROC={auroc:.3f} (95% CI: {lower_ci:.3f}-{upper_ci:.3f})\nN={n_total}, Prev={prevalence:.3f}'
        else:
            label = f'{model_name}: AUROC={auroc:.3f}, N={n_total}, Prev={prevalence:.3f}'
        
        ax.plot(fpr, tpr, label=label, linewidth=3, color=color)
        
        # Mark multiple clinically relevant thresholds (ETHOS-style)
        clinical_thresholds = find_clinically_relevant_thresholds(
            y_true, y_score, fpr, tpr, thresholds)
        
        for name, thresh, fpr_val, tpr_val in clinical_thresholds:
            ax.plot(fpr_val, tpr_val, marker='X', markersize=15, 
                   color=color, markeredgecolor='black', markeredgewidth=2)
            # Add small label for threshold type
            # ax.annotate(name, (fpr_val, tpr_val), fontsize=7, color=color)
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)', alpha=0.7)
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title('ROC - Mortality', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_roc_curves.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\n✅ Saved ROC curves to {output_path}")
    plt.close()


def bootstrap_auprc(y_true, y_score, n_bootstraps=1000, confidence_level=0.95):
    """Calculate AUPRC confidence intervals using bootstrapping."""
    np.random.seed(42)
    n_samples = len(y_true)
    auprcs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Check if bootstrap has both classes
        if len(np.unique(y_true_boot)) > 1:
            try:
                precision, recall, _ = precision_recall_curve(y_true_boot, y_score_boot)
                auprc_boot = auc(recall, precision)
                auprcs.append(auprc_boot)
            except:
                pass
    
    if len(auprcs) == 0:
        return None, None
    
    auprcs = np.array(auprcs)
    alpha = 1 - confidence_level
    lower = np.percentile(auprcs, alpha/2 * 100)
    upper = np.percentile(auprcs, (1 - alpha/2) * 100)
    
    return lower, upper


def plot_pr_curves(all_results, output_dir):
    """Plot Precision-Recall curves for all models with 95% CI (ETHOS-style)."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        y_true = results['y_true']
        y_score = results['y_score']
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        auprc = auc(recall, precision)
        
        # Calculate 95% CI for AUPRC
        lower_ci, upper_ci = bootstrap_auprc(y_true, y_score)
        
        # Get prevalence and case counts
        n_total = len(y_true)
        n_positive = np.sum(y_true == 1)
        prevalence = n_positive / n_total
        
        # Plot
        color = colors[idx % len(colors)]
        if lower_ci is not None and upper_ci is not None:
            label = f'{model_name}\nAUPRC={auprc:.3f} (95% CI: {lower_ci:.3f}-{upper_ci:.3f})\nN={n_total}, Prevalence={prevalence:.3f}'
        else:
            label = f'{model_name}\nAUPRC={auprc:.3f}\nN={n_total}, Prevalence={prevalence:.3f}'
        
        plt.plot(recall, precision, label=label, linewidth=2.5, color=color)
    
    # Baseline (prevalence)
    prevalence = results['y_true'].mean()
    plt.axhline(y=prevalence, color='k', linestyle='--', linewidth=1.5, 
                label=f'No Skill Baseline (Prevalence={prevalence:.3f})', alpha=0.5)
    
    plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    plt.ylabel('Precision (PPV)', fontsize=13, fontweight='bold')
    plt.title('Precision-Recall Curves - In-Hospital Mortality Prediction (Patient-Wise)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_pr_curves.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved PR curves to {output_path}")
    plt.close()


def plot_probability_distributions(all_results, output_dir):
    """Plot distribution of predicted probabilities."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        y_true = results['y_true']
        y_score = results['y_score']
        
        # Separate by ground truth
        death_probs = y_score[y_true == 1]
        discharge_probs = y_score[y_true == 0]
        
        # Plot histograms
        ax.hist(death_probs, bins=20, alpha=0.6, color='red', label='Died', 
                edgecolor='black', density=True)
        ax.hist(discharge_probs, bins=20, alpha=0.6, color='green', label='Survived', 
                edgecolor='black', density=True)
        
        ax.set_xlabel('Predicted Mortality Probability', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Distribution of Predicted Mortality Probabilities (Patient-Wise)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_probability_distributions.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved probability distributions to {output_path}")
    plt.close()


def plot_calibration_curves(all_results, output_dir, n_bins=10):
    """Plot calibration curves to assess probability calibration."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        y_true = results['y_true']
        y_score = results['y_score']
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy='uniform')
        
        # Plot
        color = colors[idx % len(colors)]
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', 
                linewidth=2.5, markersize=8, color=color, label=model_name)
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', alpha=0.5)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Positives (Actual)', fontsize=12, fontweight='bold')
    plt.title('Calibration Curves - Mortality Prediction (Patient-Wise)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_calibration_curves.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved calibration curves to {output_path}")
    plt.close()


def plot_confusion_matrices(all_results, output_dir, threshold=0.5):
    """Plot confusion matrices for all models."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        y_true = results['y_true']
        y_score = results['y_score']
        
        # Apply threshold
        y_pred = (y_score >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar=True, square=True, annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\n(Threshold={threshold})', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Survived', 'Died'])
        ax.set_yticklabels(['Survived', 'Died'], rotation=0)
        
        # Add metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics_text = f'Sens: {sensitivity:.2f}\nSpec: {specificity:.2f}\nPPV: {ppv:.2f}\nNPV: {npv:.2f}\nAcc: {accuracy:.2f}'
        ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Confusion Matrices - Mortality Prediction (Patient-Wise)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_confusion_matrices.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved confusion matrices to {output_path}")
    plt.close()


def plot_threshold_analysis(all_results, output_dir):
    """Plot sensitivity, specificity, and other metrics across different thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    thresholds_range = np.linspace(0, 1, 101)
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        y_true = results['y_true']
        y_score = results['y_score']
        color = colors[idx % len(colors)]
        
        sensitivities = []
        specificities = []
        ppvs = []
        npvs = []
        accuracies = []
        f1_scores = []
        
        for thresh in thresholds_range:
            y_pred = (y_score >= thresh).astype(int)
            
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            ppvs.append(ppv)
            npvs.append(npv)
            accuracies.append(accuracy)
            f1_scores.append(f1)
        
        # Plot sensitivity and specificity
        axes[0, 0].plot(thresholds_range, sensitivities, label=f'{model_name} - Sensitivity', 
                       linewidth=2, color=color, linestyle='-')
        axes[0, 0].plot(thresholds_range, specificities, label=f'{model_name} - Specificity', 
                       linewidth=2, color=color, linestyle='--', alpha=0.7)
        
        # Plot PPV and NPV
        axes[0, 1].plot(thresholds_range, ppvs, label=f'{model_name} - PPV', 
                       linewidth=2, color=color, linestyle='-')
        axes[0, 1].plot(thresholds_range, npvs, label=f'{model_name} - NPV', 
                       linewidth=2, color=color, linestyle='--', alpha=0.7)
        
        # Plot accuracy
        axes[1, 0].plot(thresholds_range, accuracies, label=model_name, 
                       linewidth=2, color=color)
        
        # Plot F1 score
        axes[1, 1].plot(thresholds_range, f1_scores, label=model_name, 
                       linewidth=2, color=color)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Sensitivity vs Specificity', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('PPV vs NPV', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Accuracy vs Threshold', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('F1 Score vs Threshold', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    
    fig.suptitle('Threshold Analysis - Mortality Prediction (Patient-Wise)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_threshold_analysis.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved threshold analysis to {output_path}")
    plt.close()


def plot_model_comparison_bar(all_results, output_dir):
    """Plot bar chart comparing model metrics."""
    metrics_to_plot = ['auroc', 'auprc', 'avg_death_traj_accuracy', 'avg_discharge_traj_accuracy']
    metric_names = ['AUROC', 'AUPRC', 'Death Traj Acc', 'Discharge Traj Acc']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    models = list(all_results.keys())
    
    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        values = [all_results[model]['statistics'][metric] for model in models]
        
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(values) * 1.15])
        
        # Rotate x labels if needed
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle('Model Performance Comparison - Mortality Prediction (Patient-Wise)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_model_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved model comparison to {output_path}")
    plt.close()


def plot_patient_scatter(all_results, output_dir):
    """Plot scatter of predicted vs actual outcomes for each patient."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        y_true = results['y_true']
        y_score = results['y_score']
        
        # Add jitter to y_true for visualization
        jitter = np.random.normal(0, 0.02, size=len(y_true))
        y_true_jittered = y_true + jitter
        
        # Color by correctness at threshold 0.5
        y_pred_binary = (y_score >= 0.5).astype(int)
        correct = (y_true == y_pred_binary)
        
        # Plot
        ax.scatter(y_score[correct], y_true_jittered[correct], 
                  alpha=0.6, s=50, c='green', label='Correct', edgecolors='black', linewidth=0.5)
        ax.scatter(y_score[~correct], y_true_jittered[~correct], 
                  alpha=0.6, s=50, c='red', label='Incorrect', edgecolors='black', linewidth=0.5)
        
        # Add decision threshold line
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Threshold=0.5')
        
        ax.set_xlabel('Predicted Mortality Probability', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Actual Outcome\n(0=Survived, 1=Died)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.3, 1.3])
        ax.set_yticks([0, 1])
    
    fig.suptitle('Patient-Level Predictions (Patient-Wise AUROC)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'mortality_patient_scatter.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved patient scatter plot to {output_path}")
    plt.close()


def create_summary_report(all_results, output_dir):
    """Create a text summary report with confidence intervals."""
    output_path = Path(output_dir) / 'mortality_prediction_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MORTALITY PREDICTION SUMMARY REPORT (PATIENT-WISE AUROC)\n")
        f.write("="*80 + "\n\n")
        
        f.write("AUROC CALCULATION METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. For each patient: Run N trajectories with different random seeds\n")
        f.write("2. Count outcomes: death_count / (death_count + discharge_count)\n")
        f.write("3. This gives ONE mortality probability per patient\n")
        f.write("4. Create vectors:\n")
        f.write("   - y_true: [1 if died, 0 if discharged] (length = num_patients)\n")
        f.write("   - y_score: [mortality_probability] (length = num_patients)\n")
        f.write("5. Calculate: AUROC = roc_auc_score(y_true, y_score)\n")
        f.write("6. Bootstrap 95% CI: Resample patients 1000 times, calculate AUC each time\n")
        f.write("\n")
        f.write("This is PATIENT-WISE: one prediction per patient, not trajectory-wise.\n")
        f.write("Similar to ETHOS and other clinical prediction models.\n")
        f.write("\n\n")
        
        for model_name, results in all_results.items():
            stats = results['statistics']
            y_true = results['y_true']
            y_score = results['y_score']
            
            # Calculate confidence intervals
            auroc_lower, auroc_upper = bootstrap_auc(y_true, y_score)
            auprc_lower, auprc_upper = bootstrap_auprc(y_true, y_score)
            
            f.write(f"{'='*80}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Dataset:\n")
            f.write(f"  Total patients (N): {stats['total_patients']}\n")
            f.write(f"  Death patients: {stats['death_patients']}\n")
            f.write(f"  Discharge patients: {stats['discharge_patients']}\n")
            prevalence = stats['death_patients'] / stats['total_patients']
            f.write(f"  Prevalence: {prevalence:.4f}\n")
            f.write(f"  Total trajectories: {stats['total_trajectories']}\n\n")
            
            f.write(f"Patient-Wise Metrics (AUROC Input):\n")
            if auroc_lower is not None and auroc_upper is not None:
                f.write(f"  AUROC: {stats['auroc']:.4f} (95% CI: {auroc_lower:.4f}-{auroc_upper:.4f})\n")
            else:
                f.write(f"  AUROC: {stats['auroc']:.4f}\n")
            
            if auprc_lower is not None and auprc_upper is not None:
                f.write(f"  AUPRC: {stats['auprc']:.4f} (95% CI: {auprc_lower:.4f}-{auprc_upper:.4f})\n")
            else:
                f.write(f"  AUPRC: {stats['auprc']:.4f}\n")
            
            f.write(f"  Overall Accuracy: {stats['overall_accuracy']:.4f}\n")
            f.write(f"  Death Patient Accuracy: {stats['death_accuracy']:.4f}\n")
            f.write(f"  Discharge Patient Accuracy: {stats['discharge_accuracy']:.4f}\n\n")
            
            f.write(f"Trajectory-Level Performance:\n")
            f.write(f"  Avg Death Traj Accuracy: {stats['avg_death_traj_accuracy']:.4f}\n")
            f.write(f"  Avg Discharge Traj Accuracy: {stats['avg_discharge_traj_accuracy']:.4f}\n\n")
            
            f.write(f"Probability Statistics:\n")
            f.write(f"  Avg Mortality Prob (All): {stats['avg_mortality_prob']:.4f}\n")
            f.write(f"  Avg Mortality Prob (Death Patients): {stats['death_avg_mortality_prob']:.4f}\n")
            f.write(f"  Avg Survival Prob (Discharge Patients): {stats['discharge_avg_survival_prob']:.4f}\n\n")
            
            f.write(f"Trajectory Outcome Counts:\n")
            f.write(f"  Death outcomes: {stats['total_death_outcomes']}\n")
            f.write(f"  Discharge outcomes: {stats['total_discharge_outcomes']}\n")
            f.write(f"  No outcomes: {stats['total_no_outcomes']}\n\n")
            
            f.write("\n")
    
    print(f"✅ Saved summary report to {output_path}")


def generate_all_plots(output_dir):
    """Generate all plots from saved results."""
    print("\n" + "="*70)
    print("GENERATING ALL PLOTS")
    print("="*70)
    
    # Set matplotlib style
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # Load results
    print(f"\n📊 Loading results from {output_dir}...")
    all_results = load_all_results(output_dir)
    print(f"✅ Loaded results for {len(all_results)} models: {list(all_results.keys())}")
    
    if len(all_results) == 0:
        print("⚠️  No results found to plot!")
        return
    
    # Generate all plots
    print("\n🎨 Generating ROC curves...")
    plot_roc_curves(all_results, output_dir)
    
    print("\n🎨 Generating PR curves...")
    plot_pr_curves(all_results, output_dir)
    
    print("\n🎨 Generating probability distributions...")
    plot_probability_distributions(all_results, output_dir)
    
    # Create summary report
    print("\n📝 Creating summary report...")
    create_summary_report(all_results, output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL PLOTS GENERATED!")
    print("="*70)
    print(f"\n📁 Files saved to: {output_dir}/")
    print("  - mortality_roc_curves.png")
    print("  - mortality_pr_curves.png")
    print("  - mortality_probability_distributions.png")
    print("  - mortality_prediction_summary.txt")
    print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Mortality Prediction Upon Admission Simulation")
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
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after running simulation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing results (skip simulation)')
    parser.add_argument('--patient-cache', type=str, default=None,
                        help='Path to save/load cached patient data (avoids reprocessing dataset)')
    parser.add_argument('--discard-unfinished-trajectories', action='store_true',
                        help='Exclude trajectories without death/discharge outcome from mortality probability calculation')
    
    args = parser.parse_args()
    
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
    
    # If plot-only mode, just generate plots and exit
    if args.plot_only:
        print("="*70)
        print("PLOT-ONLY MODE: Generating plots from existing results")
        print("="*70)
        print(f"Results directory: {output_dir}")
        generate_all_plots(output_dir)
        return
    num_patients = args.num_patients or config.get('num_patients', 100)
    num_trajectories = args.num_trajectories or config.get('num_trajectories_per_patient', 10)
    max_tokens = args.max_tokens or config.get('max_tokens', 500)
    temperature = args.temperature or config.get('temperature', 1.0)
    base_seed = args.seed or config.get('base_seed', 42)
    target_death_patients = config.get('target_death_patients', None)
    
    print("="*70)
    print("MORTALITY PREDICTION UPON ADMISSION - SIMULATION")
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
        # Single model from command line
        models_to_eval['custom_model'] = args.model
    elif 'models' in config and config['models']:
        # Multiple models from config
        for name, model_config in config['models'].items():
            if isinstance(model_config, dict):
                models_to_eval[name] = model_config['path']
            else:
                models_to_eval[name] = model_config
    else:
        print("❌ ERROR: No models specified. Use --model or add models to config file.")
        return
    
    print(f"\n📋 Models to evaluate: {list(models_to_eval.keys())}")
    
    # Load dataset once (use first model to get n_positions)
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
    
    # Generate plots if requested
    if args.plot:
        generate_all_plots(output_dir)


if __name__ == "__main__":
    main()

