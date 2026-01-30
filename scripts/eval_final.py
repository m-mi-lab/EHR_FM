#!/usr/bin/env python3
"""
Final Comprehensive Evaluation Script for EHR Foundation Model

Evaluates model on multiple clinical prediction tasks:
1. Hospital Mortality (hosp_mortality)
2. ICU Mortality (icu_mortality)
3. Hospital Readmission (hosp_readmission)
4. ICU Readmission (icu_readmission)
5. ICU Length of Stay (icu_los) - regression task

Uses frequency-based trajectory method (ETHOS-style) where probability is
calculated as the fraction of trajectories predicting the positive outcome.
This is the standard approach used in the ETHOS paper.

For ICU LOS: Aggregates time-interval tokens from trajectories until discharge,
excluding cases where patient died (ETHOS methodology).
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc, 
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.datasets.hospital_mortality import HospitalMortalityDataset
from src.tokenizer.datasets.mimic_icu import (
    ICUMortalityDataset, ICUReadmissionDataset, 
    DrgPredictionDataset, SofaPredictionDataset
)
from src.tokenizer.datasets.readmission import ReadmissionDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST
from src.utils import load_env, resolve_model_path
from omegaconf import OmegaConf
from transformers import GPT2Config
from src.model import GPT2LMNoBiasModel


# Task configurations
TASK_CONFIGS = {
    'hosp_mortality': {
        'dataset_class': HospitalMortalityDataset,
        'positive_token': ST.DEATH,
        'negative_tokens': [ST.DISCHARGE],
        'description': 'Hospital Mortality Prediction',
        'task_type': 'binary'
    },
    'icu_mortality': {
        'dataset_class': ICUMortalityDataset,
        'positive_token': ST.DEATH,
        'negative_tokens': [ST.ICU_DISCHARGE],
        'description': 'ICU Mortality Prediction',
        'task_type': 'binary',
        'calculate_los': False  # Can be enabled to also calculate LOS
    },
    'icu_los': {
        'dataset_class': ICUMortalityDataset,  # Same dataset as ICU mortality
        'positive_token': ST.DEATH,
        'negative_tokens': [ST.ICU_DISCHARGE],
        'description': 'ICU Length of Stay Prediction (Regression)',
        'task_type': 'regression',
        'calculate_los': True  # Calculate LOS from trajectories
    },
    'hosp_readmission': {
        'dataset_class': ReadmissionDataset,
        'positive_token': ST.ADMISSION,
        'negative_tokens': [ST.DEATH, ST.TIMELINE_END],
        'description': 'Hospital 30-day Readmission Prediction',
        'task_type': 'binary'
    },
    'icu_readmission': {
        'dataset_class': ICUReadmissionDataset,
        'positive_token': ST.ICU_ADMISSION,
        'negative_tokens': [ST.DISCHARGE, ST.DEATH],
        'description': 'ICU Readmission Prediction',
        'task_type': 'binary'
    },
    'drg_prediction': {
        'dataset_class': DrgPredictionDataset,
        'description': 'DRG Code Prediction',
        'task_type': 'multiclass'
    },
    'sofa_prediction': {
        'dataset_class': SofaPredictionDataset,
        'description': 'SOFA Score Prediction',
        'task_type': 'multiclass'
    }
}


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


def generate_single_trajectory(model, context, vocab_size, device, outcome_tokens, 
                               max_tokens=10000, temperature=1.0, seed=None,
                               time_interval_tokens=None, max_time_days=None, vocab=None):
    """
    Generate a single trajectory by sampling tokens until reaching an outcome token.
    ETHOS-style: No max token limit - runs until outcome or time threshold.
    
    From ETHOS paper:
    "This generative sequence proceeds until it encounters predefined stopping conditions,
    which may include the appearance of a token showing the patient's death or
    the sum of time-interval tokens surpassing a certain threshold."
    
    For 30-day readmission: "ceased upon the appearance of either a new admission 
    or death token or when the cumulative time tokens generated exceeded 30 days."
    
    Args:
        outcome_tokens: List of token IDs that are considered outcomes
        max_tokens: Safety cutoff (default 10000, ETHOS uses no hard limit)
        time_interval_tokens: Dict mapping token IDs to time in days (for time-based stopping)
        max_time_days: Maximum cumulative time in days (e.g., 30 for readmission)
        vocab: Vocabulary object (needed for decoding time intervals)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    trajectory_tokens = []
    outcome_token = None
    outcome_position = None
    cumulative_time_days = 0.0
    
    current_context = context.clone()
    
    # ETHOS-style: No hard limit, but safety cutoff at max_tokens
    actual_max_tokens = max_tokens
    
    with torch.no_grad():
        for step in range(actual_max_tokens):
            # ETHOS-style: Maintain sliding window of 2048 tokens
            # "appending the new token to the sequence while removing the oldest one
            # to maintain the context size at 2048 tokens"
            max_context_length = 2048  # ETHOS standard
            if current_context.size(0) > max_context_length:
                current_context = current_context[-max_context_length:]
            
            input_ids = current_context.unsqueeze(0).to(device)
            output = model(input_ids)
            logits = output.logits[0, -1, :]  # Last position
            
            logits = logits / temperature
            logits = logits[:vocab_size]
            
            probs = torch.softmax(logits, dim=-1)
            
            # ETHOS: "(2) stochastically selecting a new token based on these probabilities"
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            trajectory_tokens.append(next_token)
            
            # Track cumulative time if time-based stopping is enabled
            if time_interval_tokens is not None and next_token in time_interval_tokens:
                cumulative_time_days += time_interval_tokens[next_token]
            
            # Check if we hit any outcome token
            # ETHOS: "stopping conditions, which may include the appearance of a token 
            # showing the patient's death"
            if next_token in outcome_tokens:
                outcome_token = next_token
                outcome_position = step
                break
            
            # ETHOS: Time-based stopping for readmission tasks
            # "when the cumulative time tokens generated exceeded 30 days"
            if max_time_days is not None and cumulative_time_days > max_time_days:
                outcome_token = None  # Timeout, no outcome reached
                outcome_position = step
                break
            
            # ETHOS: "(3) appending the new token to the sequence while removing the oldest one"
            current_context = torch.cat([current_context, torch.tensor([next_token])])
    
    return {
        'tokens': trajectory_tokens,
        'outcome_token': outcome_token,
        'outcome_position': outcome_position,
        'trajectory_length': len(trajectory_tokens),
        'cumulative_time_days': cumulative_time_days
    }


def run_prediction_for_patient(model, context, ground_truth_outcome, vocab, device,
                               positive_token_id, negative_token_ids,
                               num_trajectories=20, max_tokens=10000, temperature=1.0,
                               base_seed=42, max_time_days=None):
    """
    Run prediction for a single patient using frequency-based trajectories (ETHOS-style).
    
    From ETHOS paper:
    "The generative process began with the admission token and ended upon generating 
    a discharge or death token, repeating this cycle 20 times. The 'N', representing 
    the number of times a death token was generated first, was divided by 20 to 
    estimate the chance of inpatient mortality."
    
    Probability = N / 20 where N = count of positive outcomes across 20 trajectories
    
    Args:
        positive_token_id: Token ID for positive outcome (e.g., DEATH, ADMISSION)
        negative_token_ids: List of token IDs for negative outcomes
        num_trajectories: Number of trajectories to generate (ETHOS uses 20)
        max_tokens: Safety cutoff (ETHOS has no hard limit, we use 10k)
    """
    vocab_size = len(vocab)
    
    # All outcome tokens
    outcome_token_ids = [positive_token_id] + negative_token_ids
    
    # Prepare time interval tracking (for readmission tasks only)
    # NOTE: This is NEW functionality for ETHOS readmission tasks (30-day time limit)
    # The original mortality_prediction scripts didn't use this, so it's optional
    time_interval_tokens = None
    
    # Only try to use time intervals for readmission tasks
    # Check if this is a readmission task by looking at positive token
    is_readmission_task = (positive_token_id == vocab.stoi.get(ST.ADMISSION) or 
                          positive_token_id == vocab.stoi.get(ST.ICU_ADMISSION))
    
    # For readmission tasks, set default to 30 days if not specified
    if is_readmission_task and max_time_days is None:
        max_time_days = 30.0  # ETHOS standard
    
    # Try to enable time-based stopping if this is a readmission task and time limit is set
    if is_readmission_task and max_time_days is not None:
        try:
            # Try to access interval estimates (may not be available)
            # This is wrapped in try-except because vocab.interval_estimates raises ValueError if not loaded
            interval_estimates = vocab.interval_estimates
            time_interval_stokens = vocab.time_interval_stokens
            
            # Get time intervals and convert to days
            time_interval_tokens = {}
            for token_str in time_interval_stokens:
                token_id = vocab.stoi.get(token_str)
                if token_id is not None:
                    try:
                        mean_time_seconds = interval_estimates['mean'].get(token_str, 0)
                        time_interval_tokens[token_id] = mean_time_seconds / (24 * 3600)  # Convert to days
                    except (KeyError, AttributeError, TypeError):
                        pass  # Skip if interval not found
            
        except (ValueError, AttributeError, KeyError):
            # Interval estimates not available - that's OK, we'll proceed without time-based stopping
            # The task will still work, just won't have the time limit
            print(f"⚠️  Warning: max_time_days={max_time_days} requested but interval estimates not available. Time-based stopping disabled.")
            max_time_days = None  # Disable time-based stopping
            time_interval_tokens = None
    
    trajectories = []
    positive_count = 0
    negative_count = 0
    no_outcome_count = 0
    
    # Run multiple trajectories with different seeds (ETHOS-style)
    for traj_idx in range(num_trajectories):
        seed = base_seed + traj_idx
        
        trajectory = generate_single_trajectory(
            model=model,
            context=context,
            vocab_size=vocab_size,
            device=device,
            outcome_tokens=outcome_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            time_interval_tokens=time_interval_tokens,
            max_time_days=max_time_days,
            vocab=vocab
        )
        
        trajectories.append(trajectory)
        
        # Count outcomes (frequency-based, ETHOS-style)
        if trajectory['outcome_token'] == positive_token_id:
            positive_count += 1
        elif trajectory['outcome_token'] in negative_token_ids:
            negative_count += 1
        else:
            no_outcome_count += 1
    
    # Frequency-based probability (ETHOS-style)
    total_with_outcome = positive_count + negative_count
    if total_with_outcome > 0:
        positive_probability = positive_count / total_with_outcome
        negative_probability = negative_count / total_with_outcome
    else:
        # If no trajectories reached an outcome, default to 0.5
        positive_probability = 0.5
        negative_probability = 0.5
    
    # Determine predicted outcome
    predicted_outcome = positive_token_id if positive_probability > 0.5 else negative_token_ids[0]
    prediction_confidence = max(positive_probability, negative_probability)
    
    # Check if correct (ground truth should be token ID)
    is_correct = (predicted_outcome == ground_truth_outcome)
    
    # Calculate ICU LOS if applicable (ETHOS methodology)
    # "LOS in the ICU was estimated by aggregating the time-interval tokens generated 
    # in the simulated timeline until the discharge token appeared. Instances where 
    # the patient died in the ICU during the simulation were excluded from the LOS calculation."
    los_days_list = []
    death_token_id = vocab.stoi.get(ST.DEATH)
    for traj in trajectories:
        # Exclude trajectories where patient died
        if traj['outcome_token'] != death_token_id:
            los_days_list.append(traj['cumulative_time_days'])
    
    # Calculate mean LOS from non-death trajectories
    mean_los_days = np.mean(los_days_list) if los_days_list else 0.0
    median_los_days = np.median(los_days_list) if los_days_list else 0.0
    
    return {
        'trajectories': trajectories,
        'num_trajectories': num_trajectories,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'no_outcome_count': no_outcome_count,
        'positive_probability': positive_probability,
        'negative_probability': negative_probability,
        'predicted_outcome': predicted_outcome,
        'ground_truth_outcome': ground_truth_outcome,
        'prediction_confidence': prediction_confidence,
        'is_correct': is_correct,
        'mean_los_days': mean_los_days,
        'median_los_days': median_los_days,
        'los_trajectory_count': len(los_days_list)
    }


def evaluate_task(model, dataset, vocab, device, task_name, task_config,
                 num_patients=100, num_trajectories_per_patient=20,
                 max_tokens=10000, temperature=1.0, base_seed=42,
                 target_positive_patients=None, cached_patients=None, max_time_days=None):
    """
    Evaluate model on a single task.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {task_config['description']}")
    print(f"{'='*70}")
    print(f"Task: {task_name}")
    print(f"Num patients: {num_patients}")
    if target_positive_patients is not None:
        print(f"Target positive patients: {target_positive_patients}")
        print(f"Target negative patients: {num_patients - target_positive_patients}")
    print(f"Trajectories per patient: {num_trajectories_per_patient}")
    print(f"Method: Frequency-based (ETHOS-style)")
    print(f"Max tokens per trajectory: {max_tokens}")
    print(f"Temperature: {temperature}")
    
    # Get token IDs
    positive_token = task_config['positive_token']
    negative_tokens = task_config['negative_tokens']
    
    if positive_token not in vocab.stoi:
        raise ValueError(f"Positive token {positive_token} not found in vocabulary")
    
    positive_token_id = vocab.stoi[positive_token]
    negative_token_ids = [vocab.stoi[t] for t in negative_tokens if t in vocab.stoi]
    
    print(f"\nPositive token: {positive_token} (ID: {positive_token_id})")
    print(f"Negative tokens: {negative_tokens} (IDs: {negative_token_ids})")
    
    # Load patients (use cache if available)
    if cached_patients:
        print(f"\n📦 Using cached patients ({len(cached_patients)} patients)...")
        patients_to_process = cached_patients[:num_patients]  # Take first N
        positive_patients_found = sum(1 for p in patients_to_process if p['ground_truth_outcome'] == positive_token_id)
        negative_patients_found = len(patients_to_process) - positive_patients_found
    else:
        print(f"\n📋 Loading patients from dataset...")
        patients_to_process = []
        patients_processed = 0
        positive_patients_found = 0
        negative_patients_found = 0
        dataset_idx = 0
        
        while patients_processed < num_patients and dataset_idx < len(dataset):
            try:
                context_tensor, metadata = dataset[dataset_idx]
                dataset_idx += 1
                
                ground_truth = metadata.get('expected', None)
                if ground_truth is None:
                    continue
                
                # ETHOS-style: No max token filtering - let trajectories run until outcome
                # true_token_dist = metadata.get('true_token_dist', None)
                
                context_tensor = context_tensor[context_tensor >= 0]
                
                if len(context_tensor) < 5:
                    continue
                
                # Convert ground truth to token ID
                if ground_truth == positive_token:
                    ground_truth_outcome = positive_token_id
                elif ground_truth in negative_tokens:
                    ground_truth_outcome = vocab.stoi[ground_truth]
                else:
                    continue
                
                # Balance classes if requested
                if target_positive_patients is not None:
                    if ground_truth_outcome == positive_token_id:
                        if positive_patients_found >= target_positive_patients:
                            continue
                        positive_patients_found += 1
                    else:
                        target_negative = num_patients - target_positive_patients
                        if negative_patients_found >= target_negative:
                            continue
                        negative_patients_found += 1
                else:
                    if ground_truth_outcome == positive_token_id:
                        positive_patients_found += 1
                    else:
                        negative_patients_found += 1
                
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
        
        print(f"✅ Loaded {len(patients_to_process)} patients")
        print(f"   Positive patients: {positive_patients_found}")
        print(f"   Negative patients: {negative_patients_found}")
    
    # Evaluate patients
    patient_results = []
    
    with tqdm(total=len(patients_to_process), desc="Patients") as pbar:
        for patient_data in patients_to_process:
            try:
                context_tensor = patient_data['context']
                ground_truth_outcome = patient_data['ground_truth_outcome']
                
                patient_result = run_prediction_for_patient(
                    model=model,
                    context=context_tensor,
                    ground_truth_outcome=ground_truth_outcome,
                    vocab=vocab,
                    device=device,
                    positive_token_id=positive_token_id,
                    negative_token_ids=negative_token_ids,
                    num_trajectories=num_trajectories_per_patient,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    base_seed=base_seed + len(patient_results) * 1000,
                    max_time_days=max_time_days
                )
                
                # Add metadata
                patient_result['patient_id'] = patient_data.get('patient_id', len(patient_results))
                patient_result['hadm_id'] = patient_data.get('hadm_id', None)
                patient_result['context_length'] = len(context_tensor)
                
                patient_results.append(patient_result)
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing patient: {e}")
                continue
    
    print(f"\n✅ Processed {len(patient_results)} patients")
    
    # Calculate statistics based on task type
    task_type = task_config.get('task_type', 'binary')
    if task_type == 'regression' and task_config.get('calculate_los', False):
        # For LOS regression task
        stats = calculate_los_statistics(patient_results, dataset)
    else:
        # For binary classification tasks
        stats = calculate_task_statistics(patient_results, positive_token_id)
    
    return {
        'task_name': task_name,
        'patient_results': patient_results,
        'statistics': stats,
        'task_type': task_type,
        'patients_data': patients_to_process,  # Include for caching
        'config': {
            'num_patients': len(patient_results),
            'num_trajectories_per_patient': num_trajectories_per_patient,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'base_seed': base_seed
        }
    }


def fit_gaussian_roc(fpr, tpr):
    """
    Fit Gaussian model to ROC curve with unequal variances (ETHOS method).
    
    Returns:
        auroc_gaussian_fit: AUROC from Gaussian fit
        params: Dictionary with fitted parameters
    """
    try:
        # Remove points at extremes (0,0) and (1,1)
        mask = (fpr > 0) & (fpr < 1) & (tpr > 0) & (tpr < 1)
        fpr_fit = fpr[mask]
        tpr_fit = tpr[mask]
        
        if len(fpr_fit) < 3:
            return None, None
        
        # Convert to z-scores (inverse normal CDF)
        z_fpr = norm.ppf(fpr_fit)
        z_tpr = norm.ppf(tpr_fit)
        
        # Fit linear relationship: z_tpr = a * z_fpr + b
        coeffs = np.polyfit(z_fpr, z_tpr, 1)
        a, b = coeffs[0], coeffs[1]
        
        # Extract parameters (ETHOS method)
        sigma_ratio = a  # sigma_neg / sigma_pos
        d_prime = b * np.sqrt(1 + sigma_ratio**2)  # Discriminability
        
        # Calculate AUROC from d-prime
        auroc_gaussian_fit = norm.cdf(d_prime / np.sqrt(1 + sigma_ratio**2))
        
        # R-squared for fit quality
        z_tpr_pred = a * z_fpr + b
        ss_res = np.sum((z_tpr - z_tpr_pred)**2)
        ss_tot = np.sum((z_tpr - np.mean(z_tpr))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        params = {
            'd_prime': d_prime,
            'sigma_ratio': sigma_ratio,
            'r_squared': r_squared
        }
        
        return auroc_gaussian_fit, params
    except Exception as e:
        print(f"Warning: Gaussian ROC fitting failed: {e}")
        return None, None


def calculate_los_statistics(patient_results, dataset):
    """
    Calculate statistics for ICU Length of Stay regression task (ETHOS methodology).
    
    From ETHOS paper:
    "LOS in the ICU was estimated by aggregating the time-interval tokens generated 
    in the simulated timeline until the discharge token appeared. Instances where 
    the patient died in the ICU during the simulation were excluded from the LOS calculation."
    """
    if not patient_results:
        return {}
    
    # Extract predicted and ground truth LOS
    predicted_los = np.array([p['mean_los_days'] for p in patient_results])
    
    # Get ground truth LOS from dataset metadata
    # For now, we'll use the true_token_time from metadata if available
    ground_truth_los = []
    for p in patient_results:
        # This would need to be extracted from dataset metadata
        # For now, use a placeholder - you'll need to add this to the dataset loading
        gt_los = p.get('ground_truth_los_days', 0)  # Placeholder
        ground_truth_los.append(gt_los)
    
    ground_truth_los = np.array(ground_truth_los)
    
    # Calculate regression metrics
    mae = np.mean(np.abs(predicted_los - ground_truth_los))
    mse = np.mean((predicted_los - ground_truth_los) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate correlation
    correlation = np.corrcoef(predicted_los, ground_truth_los)[0, 1] if len(predicted_los) > 1 else 0
    
    # Calculate percentiles of errors
    errors = predicted_los - ground_truth_los
    error_percentiles = {
        'p25': np.percentile(np.abs(errors), 25),
        'p50': np.percentile(np.abs(errors), 50),
        'p75': np.percentile(np.abs(errors), 75),
        'p90': np.percentile(np.abs(errors), 90)
    }
    
    return {
        'total_patients': len(patient_results),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'error_percentiles': error_percentiles,
        'mean_predicted_los': np.mean(predicted_los),
        'mean_ground_truth_los': np.mean(ground_truth_los),
        'median_predicted_los': np.median(predicted_los),
        'median_ground_truth_los': np.median(ground_truth_los),
        'avg_trajectories_used': np.mean([p['los_trajectory_count'] for p in patient_results])
    }


def calculate_task_statistics(patient_results, positive_token_id):
    """Calculate aggregate statistics for a task."""
    if not patient_results:
        # Return empty stats structure to avoid KeyError
        return {
            'total_patients': 0,
            'accuracy': 0.0,
            'auroc': 0.0,
            'auprc': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            'positive_patients': 0,
            'negative_patients': 0,
            'positive_accuracy': 0.0,
            'negative_accuracy': 0.0,
            'positive_avg_prob': 0.0,
            'negative_avg_prob': 0.0,
            'avg_positive_traj_accuracy': 0.0,
            'avg_negative_traj_accuracy': 0.0
        }
    
    # Basic accuracy
    correct_predictions = sum(1 for p in patient_results if p['is_correct'])
    total_patients = len(patient_results)
    accuracy = correct_predictions / total_patients if total_patients > 0 else 0
    
    # Separate by ground truth
    positive_patients = [p for p in patient_results if p['ground_truth_outcome'] == positive_token_id]
    negative_patients = [p for p in patient_results if p['ground_truth_outcome'] != positive_token_id]
    
    # Prepare data for AUROC
    y_true = np.array([1 if p['ground_truth_outcome'] == positive_token_id else 0 
                       for p in patient_results])
    y_score = np.array([p['positive_probability'] for p in patient_results])
    y_pred = np.array([1 if p['predicted_outcome'] == positive_token_id else 0 
                       for p in patient_results])
    
    # Calculate metrics
    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    
    # Gaussian ROC fitting (ETHOS method)
    auroc_gaussian_fit = None
    gaussian_params = None
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc_gaussian_fit, gaussian_params = fit_gaussian_roc(fpr, tpr)
    
    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    
    # Classification metrics at 0.5 threshold
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision_score_val = precision_score(y_true, y_pred, zero_division=0)
    recall_score_val = recall_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
    
    # Per-class statistics
    if positive_patients:
        positive_correct = sum(1 for p in positive_patients if p['is_correct'])
        positive_accuracy = positive_correct / len(positive_patients)
        positive_avg_prob = np.mean([p['positive_probability'] for p in positive_patients])
    else:
        positive_accuracy = 0
        positive_avg_prob = 0
    
    if negative_patients:
        negative_correct = sum(1 for p in negative_patients if p['is_correct'])
        negative_accuracy = negative_correct / len(negative_patients)
        negative_avg_prob = np.mean([p['positive_probability'] for p in negative_patients])
    else:
        negative_accuracy = 0
        negative_avg_prob = 0
    
    # Trajectory-level accuracy
    positive_traj_accuracies = []
    for p in positive_patients:
        acc = p['positive_count'] / p['num_trajectories'] if p['num_trajectories'] > 0 else 0
        positive_traj_accuracies.append(acc)
    
    negative_traj_accuracies = []
    for p in negative_patients:
        acc = p['negative_count'] / p['num_trajectories'] if p['num_trajectories'] > 0 else 0
        negative_traj_accuracies.append(acc)
    
    avg_positive_traj_accuracy = np.mean(positive_traj_accuracies) if positive_traj_accuracies else 0
    avg_negative_traj_accuracy = np.mean(negative_traj_accuracies) if negative_traj_accuracies else 0
    
    return {
        'total_patients': total_patients,
        'accuracy': accuracy,
        'auroc': auroc,
        'auroc_gaussian_fit': auroc_gaussian_fit,
        'gaussian_params': gaussian_params,
        'auprc': auprc,
        'f1_score': f1,
        'precision': precision_score_val,
        'recall': recall_score_val,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'positive_patients': len(positive_patients),
        'negative_patients': len(negative_patients),
        'positive_accuracy': positive_accuracy,
        'negative_accuracy': negative_accuracy,
        'positive_avg_prob': positive_avg_prob,
        'negative_avg_prob': negative_avg_prob,
        'avg_positive_traj_accuracy': avg_positive_traj_accuracy,
        'avg_negative_traj_accuracy': avg_negative_traj_accuracy,
    }


def print_task_statistics(task_name, stats, task_config):
    """Print statistics for a task."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {task_config['description']}")
    print(f"{'='*70}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total patients: {stats['total_patients']}")
    print(f"   Accuracy: {stats['accuracy']:.4f}")
    print(f"   AUROC: {stats['auroc']:.4f}")
    if stats.get('auroc_gaussian_fit'):
        print(f"   AUROC (Gaussian Fit - ETHOS): {stats['auroc_gaussian_fit']:.4f}")
        if stats.get('gaussian_params'):
            params = stats['gaussian_params']
            print(f"      d' = {params['d_prime']:.4f}, σ_ratio = {params['sigma_ratio']:.4f}, R² = {params['r_squared']:.4f}")
    print(f"   AUPRC: {stats['auprc']:.4f}")
    print(f"   F1 Score: {stats['f1_score']:.4f}")
    print(f"   Precision: {stats['precision']:.4f}")
    print(f"   Recall: {stats['recall']:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    cm = stats['confusion_matrix']
    print(f"   TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"   FN: {cm['fn']}, TP: {cm['tp']}")
    
    print(f"\n🎯 Per-Class Performance:")
    print(f"   Positive patients (n={stats['positive_patients']}):")
    print(f"      Accuracy: {stats['positive_accuracy']:.4f}")
    print(f"      Avg probability: {stats['positive_avg_prob']:.4f}")
    print(f"      Avg trajectory accuracy: {stats['avg_positive_traj_accuracy']:.4f}")
    
    print(f"   Negative patients (n={stats['negative_patients']}):")
    print(f"      Accuracy: {stats['negative_accuracy']:.4f}")
    print(f"      Avg probability: {stats['negative_avg_prob']:.4f}")
    print(f"      Avg trajectory accuracy: {stats['avg_negative_traj_accuracy']:.4f}")


def plot_task_roc_pr_curves(results, output_dir, model_name):
    """Plot ROC and PR curves for a single task."""
    task_name = results['task_name']
    task_output_dir = Path(output_dir) / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_results = results['patient_results']
    stats = results['statistics']
    task_type = results.get('task_type', 'binary')
    
    # Only plot for binary classification tasks
    if task_type != 'binary':
        return
    
    # Get predictions
    y_true = np.array([1 if p['ground_truth_outcome'] == p['predicted_outcome'] or 
                      p['positive_probability'] > 0.5 else 0 
                      for p in patient_results])
    y_score = np.array([p['positive_probability'] for p in patient_results])
    
    if len(np.unique(y_true)) <= 1:
        print(f"⚠️  Skipping plots for {task_name}: need both classes")
        return
    
    # Calculate curves
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auroc = stats['auroc']
    auprc = stats['auprc']
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auroc:.3f})')
    if stats.get('auroc_gaussian_fit'):
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.text(0.6, 0.2, f"Gaussian Fit AUC: {stats['auroc_gaussian_fit']:.3f}", 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {task_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    roc_file = task_output_dir / f'roc_curve_{task_name}_{model_name}.png'
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 ROC curve saved to: {roc_file}")
    
    # Plot PR curve
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, linewidth=2, label=f'PR (AUC = {auprc:.3f})')
    baseline = np.sum(y_true) / len(y_true)  # Prevalence
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {task_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    pr_file = task_output_dir / f'pr_curve_{task_name}_{model_name}.png'
    plt.savefig(pr_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 PR curve saved to: {pr_file}")


def save_task_results(results, output_dir, model_name):
    """Save results for a task."""
    task_name = results['task_name']
    task_output_dir = Path(output_dir) / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create serializable version
    serializable_results = {
        'task_name': task_name,
        'statistics': results['statistics'],
        'config': results['config'],
        'patient_summary': []
    }
    
    for patient in results['patient_results']:
        patient_summary = {
            'patient_id': int(patient['patient_id']) if patient.get('patient_id') is not None else None,
            'hadm_id': int(patient['hadm_id']) if patient.get('hadm_id') is not None else None,
            'context_length': patient['context_length'],
            'ground_truth_outcome': int(patient['ground_truth_outcome']),
            'predicted_outcome': int(patient['predicted_outcome']),
            'is_correct': patient['is_correct'],
            'positive_probability': patient['positive_probability'],
            'negative_probability': patient['negative_probability'],
            'prediction_confidence': patient['prediction_confidence'],
            'positive_count': patient['positive_count'],
            'negative_count': patient['negative_count'],
            'no_outcome_count': patient['no_outcome_count'],
            'num_trajectories': patient['num_trajectories']
        }
        serializable_results['patient_summary'].append(patient_summary)
    
    # Save to JSON
    output_file = task_output_dir / f"{task_name}_{model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    
    # Save trajectories
    trajectory_file = task_output_dir / f"{task_name}_trajectories_{model_name}.npy"
    trajectory_data = []
    
    for patient in results['patient_results']:
        patient_traj_data = {
            'patient_id': int(patient['patient_id']) if patient.get('patient_id') is not None else None,
            'ground_truth': int(patient['ground_truth_outcome']),
            'predicted': int(patient['predicted_outcome']),
            'trajectories': [
                {
                    'tokens': np.array(t['tokens'], dtype=np.int32),
                    'outcome_token': t['outcome_token'],
                    'outcome_position': t['outcome_position'],
                    'trajectory_length': t['trajectory_length']
                }
                for t in patient['trajectories']
            ]
        }
        trajectory_data.append(patient_traj_data)
    
    np.save(trajectory_file, trajectory_data, allow_pickle=True)
    print(f"💾 Trajectories saved to: {trajectory_file}")
    
    # Create and save plots for this task
    plot_task_roc_pr_curves(results, output_dir, model_name)


def create_comparison_plots(all_results, output_dir, model_name):
    """Create comparison plots across all tasks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract statistics
    task_names = []
    aurocs = []
    auprcs = []
    accuracies = []
    f1_scores = []
    
    for result in all_results:
        task_names.append(result['task_name'])
        stats = result['statistics']
        aurocs.append(stats['auroc'])
        auprcs.append(stats['auprc'])
        accuracies.append(stats['accuracy'])
        f1_scores.append(stats['f1_score'])
    
    # Create bar plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AUROC
    axes[0, 0].bar(task_names, aurocs, color='steelblue')
    axes[0, 0].set_ylabel('AUROC')
    axes[0, 0].set_title('AUROC by Task')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # AUPRC
    axes[0, 1].bar(task_names, auprcs, color='coral')
    axes[0, 1].set_ylabel('AUPRC')
    axes[0, 1].set_title('AUPRC by Task')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Accuracy
    axes[1, 0].bar(task_names, accuracies, color='mediumseagreen')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy by Task')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # F1 Score
    axes[1, 1].bar(task_names, f1_scores, color='mediumpurple')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score by Task')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'task_comparison_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Comparison plot saved to: {output_dir / f'task_comparison_{model_name}.png'}")


def save_patient_cache(patients_data, cache_path):
    """Save patient data to cache file."""
    cache_data = {
        'patients': [
            {
                'context': p['context'].cpu().tolist() if isinstance(p['context'], torch.Tensor) else p['context'],
                'ground_truth_outcome': p['ground_truth_outcome'],
                'dataset_idx': p['dataset_idx'],
                'patient_id': p.get('patient_id', None),
                'hadm_id': p.get('hadm_id', None)
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
            'dataset_idx': p['dataset_idx'],
            'patient_id': p.get('patient_id', None),
            'hadm_id': p.get('hadm_id', None)
        }
        for p in cache_data['patients']
    ]
    print(f"✅ Loaded {len(patients)} patients from cache: {cache_path}")
    return patients


def plot_roc_curves(all_results, output_dir, model_name):
    """Plot ROC curves for all tasks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, result in enumerate(all_results):
        task_name = result['task_name']
        patient_results = result['patient_results']
        
        # Get predictions
        y_true = np.array([1 if p['ground_truth_outcome'] == p['predicted_outcome'] or 
                          p['positive_probability'] > 0.5 else 0 
                          for p in patient_results])
        y_score = np.array([p['positive_probability'] for p in patient_results])
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = result['statistics']['auroc']
        
        # Plot
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f"{task_name} (AUROC={auroc:.3f})", 
                linewidth=2, color=color)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Tasks', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curves_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 ROC curves saved to: {output_dir / f'roc_curves_{model_name}.png'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Final Comprehensive Evaluation")
    parser.add_argument('--config', type=str, default='eval_final_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to tokenized dataset (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Tasks to evaluate (overrides config)')
    parser.add_argument('--num-patients', type=int, default=None,
                        help='Number of patients per task (overrides config)')
    parser.add_argument('--num-trajectories', type=int, default=None,
                        help='Number of trajectories per patient (overrides config)')
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
        config = OmegaConf.load(args.config)
        config = OmegaConf.to_container(config, resolve=True)
        print(f"✅ Loaded config from {args.config}")
    else:
        print(f"⚠️  Config file {args.config} not found, using defaults")
    
    # Override config with command line args
    data_dir = args.data_dir or config.get('data_dir', os.getenv('TOKENIZED_TRAIN_DIR'))
    output_dir = args.output_dir or config.get('output_dir', 'eval_final_results')
    tasks_to_eval = args.tasks or config.get('tasks', list(TASK_CONFIGS.keys()))
    num_patients = args.num_patients or config.get('num_patients', 100)
    num_trajectories = args.num_trajectories or config.get('num_trajectories_per_patient', 20)
    max_tokens = config.get('max_tokens', 500)
    temperature = config.get('temperature', 1.0)
    base_seed = config.get('base_seed', 42)
    
    print("="*70)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tasks: {tasks_to_eval}")
    print(f"Number of patients per task: {num_patients}")
    print(f"Trajectories per patient: {num_trajectories}")
    
    # Get model path
    if args.model:
        model_path = str(resolve_model_path(args.model))
        model_name = 'custom_model'
    elif 'model' in config:
        model_path = str(resolve_model_path(config['model']))
        model_name = config.get('model_name', 'model')
    else:
        print("❌ ERROR: No model specified. Use --model or add model to config file.")
        return
    
    print(f"\n📋 Model: {model_name}")
    print(f"   Path: {model_path}")
    
    # Load model
    print("\nLoading model...")
    model, vocab, device, model_cfg = load_model(model_path)
    print(f"✅ Model loaded. Vocab size: {len(vocab)}")
    
    # Load or prepare patient cache
    cached_patients = {}
    if args.patient_cache and Path(args.patient_cache).exists():
        print(f"\n📦 Loading patient cache from: {args.patient_cache}")
        try:
            cached_patients_list = load_patient_cache(args.patient_cache)
            # Group by task if cache has task info, otherwise use for all tasks
            for task_name in tasks_to_eval:
                cached_patients[task_name] = cached_patients_list
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}, will load from dataset")
    
    # Evaluate each task
    all_results = []
    patient_cache_to_save = {}
    
    for task_name in tasks_to_eval:
        if task_name not in TASK_CONFIGS:
            print(f"⚠️  Unknown task: {task_name}, skipping...")
            continue
        
        task_config = TASK_CONFIGS[task_name]
        
        # Load dataset
        print(f"\nLoading dataset for {task_name}...")
        dataset = task_config['dataset_class'](
            input_dir=Path(data_dir),
            n_positions=model_cfg.n_positions
        )
        print(f"✅ Dataset loaded. Size: {len(dataset)}")
        
        # Get task-specific config
        task_specific_config = config.get('task_configs', {}).get(task_name, {})
        task_num_patients = task_specific_config.get('num_patients', num_patients)
        task_target_positive = task_specific_config.get('target_positive_patients', None)
        task_max_time_days = task_specific_config.get('max_time_days', None)  # Configurable time limit
        
        # Evaluate task (use cached patients if available)
        cached_patients_for_task = cached_patients.get(task_name) if cached_patients else None
        results = evaluate_task(
            model=model,
            dataset=dataset,
            vocab=vocab,
            device=device,
            task_name=task_name,
            task_config=task_config,
            num_patients=task_num_patients,
            num_trajectories_per_patient=num_trajectories,
            max_tokens=max_tokens,
            temperature=temperature,
            base_seed=base_seed,
            target_positive_patients=task_target_positive,
            cached_patients=cached_patients_for_task,
            max_time_days=task_max_time_days  # Pass configurable time limit
        )
        
        # Print statistics
        print_task_statistics(task_name, results['statistics'], task_config)
        
        # Save results
        save_task_results(results, output_dir, model_name)
        
        all_results.append(results)
        
        # Store patients for caching (if cache path provided)
        if args.patient_cache and 'patients_data' in results:
            patient_cache_to_save[task_name] = results['patients_data']
    
    # Save patient cache if we created one
    if args.patient_cache and patient_cache_to_save:
        # Save first task's patients as cache (they can be reused)
        first_task = list(patient_cache_to_save.keys())[0]
        save_patient_cache(patient_cache_to_save[first_task], args.patient_cache)
        print(f"💾 Patient cache saved to: {args.patient_cache}")
    
    # Create comparison plots
    if len(all_results) > 1:
        print(f"\n📊 Creating comparison plots...")
        create_comparison_plots(all_results, output_dir, model_name)
        plot_roc_curves(all_results, output_dir, model_name)
    
    # Save summary
    summary = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'tasks': {
            result['task_name']: result['statistics']
            for result in all_results
        }
    }
    
    summary_file = Path(output_dir) / f'evaluation_summary_{model_name}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Task':<25} {'AUROC':<10} {'AUPRC':<10} {'Accuracy':<10} {'F1':<10}")
    print("-" * 70)
    for result in all_results:
        stats = result['statistics']
        print(f"{result['task_name']:<25} {stats['auroc']:<10.4f} {stats['auprc']:<10.4f} {stats['accuracy']:<10.4f} {stats['f1_score']:<10.4f}")
    
    print(f"\n✅ Final evaluation complete!")
    print(f"📁 Results saved to: {output_dir}/")
    
    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

