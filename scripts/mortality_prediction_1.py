#!/usr/bin/env python3
"""
Mortality Prediction Upon Admission Simulation - Direct Probability Version

Methodology:
1. Find timelines with hospital admission followed by either death or discharge
2. For each patient, extract model's DIRECT probability at admission time
3. Use softmax(logits)[death_token_id] / (death + discharge) for continuous probability
4. Calculate AUROC and other classification metrics

This version uses direct probabilities instead of frequency-based sampling for:
- Continuous probabilities (0.0 to 1.0) instead of discrete (0.0, 0.1, ..., 1.0)
- 10x faster (1 forward pass vs 10)
- Deterministic results (no sampling variance)
- Better AUROC discrimination
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
from scipy.stats import mannwhitneyu, norm
from scipy.optimize import curve_fit
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


def get_direct_mortality_probability(model, context, vocab, device):
    """
    Get model's direct probability of death vs discharge at admission time.
    Returns continuous probability instead of frequency-based estimate.
    """
    death_token_id = vocab.stoi.get(ST.DEATH)
    discharge_token_id = vocab.stoi.get(ST.DISCHARGE)
    
    with torch.no_grad():
        # Limit context length
        if context.size(0) > 512:
            context = context[-512:]
        
        input_ids = context.unsqueeze(0).to(device)
        output = model(input_ids)
        logits = output.logits[0, -1, :]  # Last position logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        death_prob = probs[death_token_id].item()
        discharge_prob = probs[discharge_token_id].item()
        
        # Normalize to only death vs discharge
        total_prob = death_prob + discharge_prob
        if total_prob > 0:
            mortality_probability = death_prob / total_prob
        else:
            mortality_probability = 0.5  # Fallback if both are zero
        
        return {
            'mortality_probability': mortality_probability,
            'death_prob_raw': death_prob,
            'discharge_prob_raw': discharge_prob
        }


def fit_gaussian_roc(fpr, tpr):
    """
    Fit Gaussian model to ROC curve with unequal variances (ETHOS method).
    
    Returns:
        auroc_gaussian_fit: AUROC from Gaussian fit
        params: (mu_pos, sigma_pos, mu_neg, sigma_neg) - fitted parameters
    """
    try:
        # Convert FPR/TPR to decision variable space using inverse normal CDF
        # Remove boundary points (0 and 1) to avoid infinities
        mask = (fpr > 0) & (fpr < 1) & (tpr > 0) & (tpr < 1)
        if mask.sum() < 4:  # Need at least 4 points for fitting
            return None, None
        
        fpr_filtered = fpr[mask]
        tpr_filtered = tpr[mask]
        
        # Convert to z-scores (decision variable space)
        z_fpr = norm.ppf(fpr_filtered)  # Inverse CDF for false positive rate
        z_tpr = norm.ppf(tpr_filtered)  # Inverse CDF for true positive rate
        
        # Fit linear relationship: z_tpr = a * z_fpr + b
        # This assumes two Gaussian distributions with different means and variances
        coeffs = np.polyfit(z_fpr, z_tpr, 1)
        a, b = coeffs[0], coeffs[1]
        
        # Extract distribution parameters
        # For unequal variance Gaussians:
        # a = sigma_neg / sigma_pos
        # b = (mu_pos - mu_neg) / sigma_pos
        sigma_ratio = a
        d_prime = b  # Sensitivity index (separation in standard deviations)
        
        # Calculate AUROC from d' (d-prime)
        # For unequal variances: AUROC = Φ(d' / sqrt(1 + sigma_ratio^2))
        auroc_gaussian_fit = norm.cdf(d_prime / np.sqrt(1 + sigma_ratio**2))
        
        params = {
            'sigma_ratio': sigma_ratio,
            'd_prime': d_prime,
            'r_squared': np.corrcoef(z_fpr, z_tpr)[0, 1]**2
        }
        
        return auroc_gaussian_fit, params
    except Exception as e:
        print(f"Warning: Gaussian ROC fitting failed: {e}")
        return None, None


def bootstrap_auroc_ci(y_true, y_score, n_bootstraps=1000, confidence_level=0.95, random_seed=42):
    """
    Calculate AUROC confidence intervals using bootstrapping (ETHOS method).
    
    Returns:
        auroc_mean: Mean AUROC across bootstrap samples
        auroc_ci_lower: Lower bound of confidence interval
        auroc_ci_upper: Upper bound of confidence interval
        auroc_std: Standard deviation of AUROC
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    aurocs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Check if bootstrap has both classes
        if len(np.unique(y_true_boot)) > 1:
            try:
                auroc_boot = roc_auc_score(y_true_boot, y_score_boot)
                aurocs.append(auroc_boot)
            except:
                pass
    
    if len(aurocs) == 0:
        return None, None, None, None
    
    aurocs = np.array(aurocs)
    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    auroc_ci_lower = np.percentile(aurocs, alpha/2 * 100)
    auroc_ci_upper = np.percentile(aurocs, (1 - alpha/2) * 100)
    
    return auroc_mean, auroc_ci_lower, auroc_ci_upper, auroc_std


def run_mortality_prediction_for_patient(model, context, ground_truth_outcome, vocab, device, 
                                         use_direct_prob=True, **kwargs):
    """
    Run mortality prediction for a single patient.
    
    If use_direct_prob=True: Uses model's direct probability (fast, continuous)
    If use_direct_prob=False: Falls back to frequency-based sampling (not implemented here)
    """
    
    if not use_direct_prob:
        raise NotImplementedError("Frequency-based sampling not implemented in this version. Use mortality_prediction.py instead.")
    
    # NEW: Direct probability approach
    prob_result = get_direct_mortality_probability(model, context, vocab, device)
    mortality_probability = prob_result['mortality_probability']
    
    # For compatibility, still determine predicted outcome
    if mortality_probability > 0.5:
        predicted_outcome = ST.DEATH
        prediction_confidence = mortality_probability
    else:
        predicted_outcome = ST.DISCHARGE
        prediction_confidence = 1.0 - mortality_probability
    
    is_correct = (predicted_outcome == ground_truth_outcome)
    
    return {
        'method': 'direct_probability',
        'mortality_probability': mortality_probability,
        'death_prob_raw': prob_result['death_prob_raw'],
        'discharge_prob_raw': prob_result['discharge_prob_raw'],
        'predicted_outcome': predicted_outcome,
        'ground_truth_outcome': ground_truth_outcome,
        'prediction_confidence': prediction_confidence,
        'is_correct': is_correct,
        # Add dummy fields for compatibility with plotting code
        'num_trajectories': 1,
        'death_count': 1 if predicted_outcome == ST.DEATH else 0,
        'discharge_count': 1 if predicted_outcome == ST.DISCHARGE else 0,
        'no_outcome_count': 0,
        'survival_probability': 1.0 - mortality_probability
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


def run_mortality_prediction_simulation(model, dataset, vocab, device, model_cfg,
                                        num_patients=100, num_trajectories_per_patient=10,
                                        max_tokens=500, temperature=1.0, base_seed=42,
                                        target_death_patients=None, cached_patients=None,
                                        discard_unfinished=False, use_direct_prob=True):
    """
    Run mortality prediction simulation on multiple patients.
    """
    print(f"\n{'='*70}")
    print(f"MORTALITY PREDICTION SIMULATION - DIRECT PROBABILITY METHOD")
    print(f"{'='*70}")
    print(f"Num patients: {num_patients}")
    if target_death_patients is not None:
        print(f"Target death patients: {target_death_patients}")
        print(f"Target discharge patients: {num_patients - target_death_patients}")
    print(f"Method: Direct probability (1 forward pass per patient)")
    print(f"Note: num_trajectories, max_tokens, temperature are ignored in direct prob mode")
    
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
                    use_direct_prob=use_direct_prob
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
            'method': 'direct_probability',
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
    
    # Trajectory outcome statistics (for compatibility)
    total_trajectories = sum(p['num_trajectories'] for p in patient_results)
    total_death_outcomes = sum(p['death_count'] for p in patient_results)
    total_discharge_outcomes = sum(p['discharge_count'] for p in patient_results)
    total_no_outcomes = sum(p['no_outcome_count'] for p in patient_results)
    
    # Trajectory-level accuracy (for compatibility)
    death_patient_traj_accuracies = []
    for p in death_patients:
        accuracy = p['death_count'] / p['num_trajectories'] if p['num_trajectories'] > 0 else 0
        death_patient_traj_accuracies.append(accuracy)
    
    avg_death_traj_accuracy = np.mean(death_patient_traj_accuracies) if death_patient_traj_accuracies else 0
    
    discharge_patient_traj_accuracies = []
    for p in discharge_patients:
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
            # Standard AUROC
            auroc = roc_auc_score(y_true, y_score)
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
            auprc = auc(recall, precision)
            
            # Bootstrap confidence intervals for AUROC
            auroc_mean, auroc_ci_lower, auroc_ci_upper, auroc_std = bootstrap_auroc_ci(
                y_true, y_score, n_bootstraps=1000, confidence_level=0.95
            )
            
            # Gaussian AUROC fit (ETHOS method)
            auroc_gaussian_fit, gaussian_params = fit_gaussian_roc(fpr, tpr)
            
            # Mann-Whitney U statistic (alternative Gaussian AUROC)
            pos_scores = y_score[y_true == 1]
            neg_scores = y_score[y_true == 0]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                u_stat, _ = mannwhitneyu(pos_scores, neg_scores, alternative='greater')
                auroc_mw = u_stat / (len(pos_scores) * len(neg_scores))
            else:
                auroc_mw = 0.0
        else:
            auroc = 0.0
            auroc_mean = 0.0
            auroc_ci_lower = 0.0
            auroc_ci_upper = 0.0
            auroc_std = 0.0
            auroc_gaussian_fit = 0.0
            auroc_mw = 0.0
            gaussian_params = None
            auprc = 0.0
            fpr, tpr, thresholds = None, None, None
            precision, recall, pr_thresholds = None, None, None
    except Exception as e:
        print(f"Warning: Could not calculate AUROC: {e}")
        auroc = 0.0
        auroc_mean = 0.0
        auroc_ci_lower = 0.0
        auroc_ci_upper = 0.0
        auroc_std = 0.0
        auroc_gaussian_fit = 0.0
        auroc_mw = 0.0
        gaussian_params = None
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
        'auroc_mean': auroc_mean,
        'auroc_ci_lower': auroc_ci_lower,
        'auroc_ci_upper': auroc_ci_upper,
        'auroc_std': auroc_std,
        'auroc_gaussian_fit': auroc_gaussian_fit if auroc_gaussian_fit is not None else 0.0,
        'auroc_mw': auroc_mw,
        'gaussian_params': gaussian_params,
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
        
        # Trajectory-level stats (for compatibility)
        'total_trajectories': total_trajectories,
        'total_death_outcomes': total_death_outcomes,
        'total_discharge_outcomes': total_discharge_outcomes,
        'total_no_outcomes': total_no_outcomes,
        
        # Death patient trajectory stats
        'death_patient_trajectories_total': death_patient_trajectories_total,
        'death_patient_death_traj': death_patient_death_traj,
        'death_patient_discharge_traj': death_patient_discharge_traj,
        'death_patient_no_outcome_traj': death_patient_no_outcome_traj,
        'avg_death_traj_accuracy': avg_death_traj_accuracy,
        
        # Discharge patient trajectory stats
        'discharge_patient_trajectories_total': discharge_patient_trajectories_total,
        'discharge_patient_death_traj': discharge_patient_death_traj,
        'discharge_patient_discharge_traj': discharge_patient_discharge_traj,
        'discharge_patient_no_outcome_traj': discharge_patient_no_outcome_traj,
        'discharge_patient_non_death_traj': discharge_patient_non_death_traj,
        'avg_discharge_traj_accuracy': avg_discharge_traj_accuracy,
    }
    
    return stats


def print_statistics(stats):
    """Print statistics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"MORTALITY PREDICTION RESULTS - DIRECT PROBABILITY METHOD")
    print(f"{'='*70}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total patients: {stats['total_patients']}")
    print(f"   Method: Direct probability (continuous 0.0-1.0)")
    
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
    print(f"   AUROC Mann-Whitney U: {stats['auroc_mw']:.4f}")
    print(f"   AUPRC (Area Under PR Curve): {stats['auprc']:.4f}")
    
    print(f"\n🎯 Patient-Level Accuracy:")
    print(f"   Overall accuracy: {stats['overall_accuracy']:.2%}")
    print(f"   Death prediction accuracy: {stats['death_accuracy']:.2%}")
    print(f"   Discharge prediction accuracy: {stats['discharge_accuracy']:.2%}")
    
    print(f"\n💀 Death Patients (n={stats['death_patients']}):")
    print(f"   Correct predictions: {int(stats['death_accuracy'] * stats['death_patients'])}/{stats['death_patients']}")
    print(f"   Average mortality probability: {stats['death_avg_mortality_prob']:.4f}")
    
    print(f"\n✅ Discharge Patients (n={stats['discharge_patients']}):")
    print(f"   Correct predictions: {int(stats['discharge_accuracy'] * stats['discharge_patients'])}/{stats['discharge_patients']}")
    print(f"   Average survival probability: {stats['discharge_avg_survival_prob']:.4f}")


def save_results(results, output_dir, model_name):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create serializable version
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
            'survival_probability': patient.get('survival_probability', 1.0 - patient['mortality_probability']),
            'prediction_confidence': patient['prediction_confidence'],
            'death_prob_raw': patient.get('death_prob_raw', None),
            'discharge_prob_raw': patient.get('discharge_prob_raw', None),
            'method': patient.get('method', 'direct_probability')
        }
        serializable_results['patient_summary'].append(patient_summary)
    
    # Save to JSON
    output_file = f"{output_dir}/mortality_prediction_{model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Mortality Prediction - Direct Probability Method")
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (overrides config)')
    parser.add_argument('--patient-cache', type=str, default=None,
                        help='Path to save/load cached patient data')
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
    data_dir = args.data_dir or config.get('data_dir', os.getenv('TOKENIZED_TRAIN_DIR', '/workspace/ehr_stuff/EHR_FM/data/tokenized_datasets/mimic_train'))
    output_dir = args.output_dir or config.get('output_dir', 'mortality_prediction_results')
    num_patients = args.num_patients or config.get('num_patients', 100)
    base_seed = args.seed or config.get('base_seed', 42)
    target_death_patients = config.get('target_death_patients', None)
    
    print("="*70)
    print("MORTALITY PREDICTION - DIRECT PROBABILITY METHOD")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of patients: {num_patients}")
    if target_death_patients is not None:
        print(f"Target death patients: {target_death_patients}")
        print(f"Target discharge patients: {num_patients - target_death_patients}")
    print(f"Base seed: {base_seed}")
    print(f"Method: Direct probability (continuous)")
    
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
            num_trajectories_per_patient=1,  # Ignored in direct prob mode
            max_tokens=500,  # Ignored in direct prob mode
            temperature=1.0,  # Ignored in direct prob mode
            base_seed=base_seed,
            target_death_patients=target_death_patients,
            cached_patients=cached_patients,
            discard_unfinished=False,
            use_direct_prob=True
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
        print(f"\n{'Model':<20} {'AUROC':<12} {'Overall Acc':<15}")
        print("-" * 50)
        for model_name, stats in all_model_results.items():
            print(f"{model_name:<20} {stats['auroc']:>10.4f}  {stats['overall_accuracy']:>13.2%}")
    
    print(f"\n✅ Mortality prediction simulation complete!")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

