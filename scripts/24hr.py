#!/usr/bin/env python3
"""
24-Hour Prediction Script for EHR-FM Models
Predicts discharge and mortality within 24 hours of admission.
"""

import torch
import numpy as np
from pathlib import Path
import json
import yaml
import argparse
import os
from collections import defaultdict
from datetime import timedelta
from omegaconf import OmegaConf
from transformers import GPT2Config
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model import GPT2LMNoBiasModel, ModelOutput
from src.tokenizer.datasets.base import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST

def load_config(config_path):
    """Load evaluation configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_config(model_path):
    """Load model and configuration from checkpoint."""
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_configs' not in checkpoint:
        raise KeyError("Model configuration not found in checkpoint")
    if 'vocab_stoi' not in checkpoint:
        raise KeyError("Vocabulary not found in checkpoint")
    
    # Reconstruct vocabulary
    sorted_vocab_stoi = sorted(checkpoint['vocab_stoi'].items(), key=lambda item: item[1])
    vocab_list = [item[0] for item in sorted_vocab_stoi]
    vocab = Vocabulary(vocab=vocab_list)
    
    # Get model config
    model_cfg = OmegaConf.create(checkpoint['model_configs'])
    
    # Calculate padded vocab size (as done in training)
    vocab_size = len(vocab)
    runtime_vocab_size = (vocab_size // 64 + 1) * 64 if vocab_size % 64 != 0 else vocab_size
    
    # Create GPT2Config
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
    
    # Create model
    model = GPT2LMNoBiasModel(
        base_gpt_config=base_config,
        moe_hydra_config=model_cfg
    )
    
    # Load model state
    model_state = checkpoint['model']
    current_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in model_state.items() if k in current_keys}
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {model_cfg.n_layer} layers, {model_cfg.n_head} heads")
    print(f"   Vocab size: {vocab_size}, Runtime vocab size: {runtime_vocab_size}")
    
    return model, vocab, device, model_cfg

def load_dataset(data_dir, vocab, n_positions=512):
    """Load the tokenized dataset."""
    dataset = TimelineDataset(data_dir, n_positions=n_positions)
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    return dataset

def is_24hr_plus_time_token(token_str):
    """
    Check if a token represents a time interval >= 24 hours.
    
    Args:
        token_str: Token string to check
    
    Returns:
        bool: True if token represents >= 24 hours
    """
    # Time interval tokens that indicate 24+ hours
    day_plus_intervals = [
        '1d-2d', '2d-4d', '4d-7d', '7d-12d', '12d-20d', '20d-30d', 
        '30d-2mt', '2mt-6mt', '≥6mt', '>=6mt'
    ]
    
    # Check if token contains any of these intervals
    for interval in day_plus_intervals:
        if interval in token_str:
            return True
    
    return False


def check_discharge_or_death_within_24hr(token_strings, start_pos):
    """
    Check if DISCHARGE or DEATH occurs within the next 24 hours from start_pos.
    
    Args:
        token_strings: List of token strings
        start_pos: Position to start checking from (e.g., after admission)
    
    Returns:
        tuple: (has_event, event_type, event_pos)
            - has_event: True if discharge/death found before 24hr+ time token
            - event_type: 'DISCHARGE' or 'DEATH' or None
            - event_pos: Position of the event or None
    """
    for i in range(start_pos, len(token_strings)):
        token = token_strings[i]
        
        # Check if we hit a 24hr+ time token (end of window)
        if is_24hr_plus_time_token(token):
            return False, None, None
        
        # Check for discharge
        if token == 'HOSPITAL_DISCHARGE':
            return True, 'DISCHARGE', i
        
        # Check for death
        if token == 'MEDS_DEATH':
            return True, 'DEATH', i
    
    # Reached end of timeline without finding event or 24hr marker
    return False, None, None


def has_24hr_passed(timeline_tokens, vocab, current_pos):
    """
    Check if 24 hours have passed from admission to current position.
    
    Args:
        timeline_tokens: List of token strings in the timeline
        vocab: Vocabulary object with interval estimates
        current_pos: Current position in timeline
    
    Returns:
        bool: True if 24+ hours have passed
    """
    # Check for any day+ interval tokens
    for token in timeline_tokens[:current_pos]:
        if is_24hr_plus_time_token(token):
            return True
    
    # Alternative: accumulate time intervals using vocabulary estimates
    if hasattr(vocab, 'get_timeline_total_time'):
        try:
            total_time = vocab.get_timeline_total_time(
                timeline_tokens[:current_pos], 
                stat='mean', 
                input_str=True
            )
            return total_time >= timedelta(hours=24)
        except:
            pass
    
    # Fallback: look for specific 18h+ and 12h+ combinations
    time_accumulated = 0
    for token in timeline_tokens[:current_pos]:
        if token == '18h-1d':
            time_accumulated += 18
        elif token == '12h-18h':
            time_accumulated += 12
        elif token == '8h-12h':
            time_accumulated += 8
        elif token == '5h-8h':
            time_accumulated += 5
        elif token == '3h-5h':
            time_accumulated += 3
        elif token == '2h-3h':
            time_accumulated += 2
        
        if time_accumulated >= 24:
            return True
    
    return False

def find_samples_with_events(dataset, vocab, target_tokens, max_search=50000):
    """First pass: find samples that contain the target events."""
    print(f"🔍 Scanning {max_search} samples to find events: {target_tokens}")
    
    samples_with_events = {token: [] for token in target_tokens}
    target_token_ids = {token: vocab.stoi.get(token, -1) for token in target_tokens}
    
    for i in range(min(max_search, len(dataset))):
        if i % 5000 == 0:
            counts = {token: len(samples) for token, samples in samples_with_events.items()}
            print(f"  Scanned {i}/{max_search}, found: {counts}")
        
        sample = dataset[i]
        if isinstance(sample, tuple):
            context, timeline = sample
            full_sequence = torch.cat([context, timeline])
        else:
            full_sequence = sample
        
        token_ids = full_sequence.tolist()
        valid_token_ids = [tid for tid in token_ids if tid >= 0 and tid in vocab.itos]
        token_strings = [vocab.decode([tid])[0] for tid in valid_token_ids]
        
        # Find admission events
        admission_positions = [pos for pos, token in enumerate(token_strings) if token == 'HOSPITAL_ADMISSION']
        
        # For each admission, check if discharge/death occurs within 24hr
        for adm_pos in admission_positions:
            if adm_pos < 10:  # Need some context
                continue
            
            # Check what happens within 24hr after this admission
            has_event, event_type, event_pos = check_discharge_or_death_within_24hr(token_strings, adm_pos + 1)
            
            if has_event:
                if event_type == 'DISCHARGE' and 'HOSPITAL_DISCHARGE' in target_tokens:
                    samples_with_events['HOSPITAL_DISCHARGE'].append((i, adm_pos, valid_token_ids, event_pos, True))
                elif event_type == 'DEATH' and 'MEDS_DEATH' in target_tokens:
                    samples_with_events['MEDS_DEATH'].append((i, adm_pos, valid_token_ids, event_pos, True))
            else:
                # No discharge/death within 24hr - negative example
                # Store as "HOSPITAL_ADMISSION" to indicate no event
                if 'HOSPITAL_ADMISSION' in target_tokens:
                    samples_with_events['HOSPITAL_ADMISSION'].append((i, adm_pos, valid_token_ids, None, False))
    
    total_found = sum(len(samples) for samples in samples_with_events.values())
    print(f"✅ Found {total_found} events across all types")
    for token, samples in samples_with_events.items():
        print(f"   {token}: {len(samples)} events")
    
    return samples_with_events

def test_predictions_on_events(model, samples_with_events, vocab, device, model_cfg, target_tokens, top_k=5, max_per_event=100):
    """
    Second pass: test model predictions at admission points.
    
    For each admission, predict if discharge/death will occur within 24hr.
    Labels: 
      - HOSPITAL_DISCHARGE: discharge within 24hr (positive)
      - MEDS_DEATH: death within 24hr (positive)
      - HOSPITAL_ADMISSION: no event within 24hr (negative)
    """
    print(f"🧠 Testing 24hr discharge/death predictions (max {max_per_event} per event type)...")
    
    all_predictions = []
    discharge_token_id = vocab.stoi.get('HOSPITAL_DISCHARGE', -1)
    death_token_id = vocab.stoi.get('MEDS_DEATH', -1)
    
    for token_name in target_tokens:
        events = samples_with_events[token_name]
        if not events:
            continue
            
        # Limit to max_per_event samples
        events_to_test = events[:max_per_event]
        print(f"  Testing {len(events_to_test)} {token_name} cases (out of {len(events)} found)...")
        
        for event_data in events_to_test:
            sample_id, adm_pos, valid_token_ids, event_pos, has_event = event_data
            
            # Use context up to admission (predict what happens AFTER admission)
            context_tokens = valid_token_ids[:adm_pos + 1]  # Include admission token
            
            # Truncate if too long
            max_len = model_cfg.n_positions - 1
            if len(context_tokens) > max_len:
                context_tokens = context_tokens[-max_len:]
            
            context_ids = torch.tensor(context_tokens).unsqueeze(0).to(device)
            
            if context_ids.size(1) == 0:
                continue
                
            try:
                with torch.no_grad():
                    output = model(context_ids)
                    logits = output.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Get probabilities for discharge and death
                    discharge_prob = probs[discharge_token_id].item() if discharge_token_id >= 0 else 0.0
                    death_prob = probs[death_token_id].item() if death_token_id >= 0 else 0.0
                    
                    top_probs, top_indices = torch.topk(probs, k=top_k)
                    
                    # Ground truth label
                    true_label = None
                    if token_name == 'HOSPITAL_DISCHARGE' and has_event:
                        true_label = 'DISCHARGE_WITHIN_24HR'
                    elif token_name == 'MEDS_DEATH' and has_event:
                        true_label = 'DEATH_WITHIN_24HR'
                    elif token_name == 'HOSPITAL_ADMISSION' and not has_event:
                        true_label = 'NO_EVENT_WITHIN_24HR'
                    
                    prediction = {
                        'sample_id': sample_id,
                        'admission_pos': adm_pos,
                        'event_pos': event_pos,
                        'true_label': true_label,
                        'has_event_within_24hr': has_event,
                        'discharge_prob': discharge_prob,
                        'death_prob': death_prob,
                        'top_predictions': top_indices.cpu().tolist(),
                        'top_probabilities': top_probs.cpu().tolist(),
                        'is_discharge': token_name == 'HOSPITAL_DISCHARGE',
                        'is_death': token_name == 'MEDS_DEATH',
                        'is_no_event': token_name == 'HOSPITAL_ADMISSION',
                        'context_length': context_ids.size(1)
                    }
                    all_predictions.append(prediction)
                    
            except Exception as e:
                print(f"Error predicting {token_name} at sample {sample_id}: {e}")
                continue
    
    return all_predictions

def find_special_tokens_and_predict(model, dataset, vocab, device, model_cfg, target_tokens=['HOSPITAL_DISCHARGE', 'MEDS_DEATH', 'HOSPITAL_ADMISSION'], max_samples=50000, top_k=5):
    """
    Two-pass approach: 
    1. Find samples with target events
    2. Test model predictions on those specific events
    """
    # First pass: find samples with events
    samples_with_events = find_samples_with_events(dataset, vocab, target_tokens, max_samples)
    
    # Second pass: test predictions on found events (limit to 100 per event type)
    all_predictions = test_predictions_on_events(model, samples_with_events, vocab, device, model_cfg, target_tokens, top_k, max_per_event=100)
    
    print(f"✅ Tested {len(all_predictions)} predictions")
    
    # Separate by event type
    discharge_preds = [p for p in all_predictions if p['is_discharge']]
    mortality_preds = [p for p in all_predictions if p['is_death']]
    admission_preds = [p for p in all_predictions if p['is_admission']]
    
    print(f"   Discharge predictions: {len(discharge_preds)}")
    print(f"   Mortality predictions: {len(mortality_preds)}")
    print(f"   Admission predictions: {len(admission_preds)}")
    
    # Calculate accuracy for each target token
    results = {}
    target_token_ids = {token: vocab.stoi.get(token, -1) for token in target_tokens}
    
    for token_name in target_tokens:
        token_id = target_token_ids[token_name]
        if token_id == -1:
            continue
            
        token_preds = [p for p in all_predictions if p['true_token_str'] == token_name]
        if not token_preds:
            continue
            
        # Calculate top-k accuracy
        accuracies = {}
        for k in [1, 3, 5]:
            if k <= top_k:
                correct = sum(1 for p in token_preds if token_id in p['top_predictions'][:k])
                accuracies[f'top_{k}_accuracy'] = correct / len(token_preds) if token_preds else 0
        
        results[token_name] = {
            'predictions': token_preds,
            'count': len(token_preds),
            'accuracies': accuracies
        }
    
    return results, all_predictions

def find_admission_indices(dataset, vocab, max_samples=1000):
    """Find indices where hospital admissions occur."""
    admission_indices = []
    
    print(f"Scanning first {max_samples} samples for admission tokens...")
    for i in range(min(max_samples, len(dataset))):
        if i % 100 == 0:
            print(f"  Scanned {i}/{max_samples} samples, found {len(admission_indices)} admissions...")
        
        sample = dataset[i]
        if isinstance(sample, tuple):
            context, timeline = sample
            full_sequence = torch.cat([context, timeline])
        else:
            full_sequence = sample
        
        # Convert to token strings, filtering out invalid token IDs (like -100)
        token_ids = full_sequence.tolist()
        valid_token_ids = [tid for tid in token_ids if tid >= 0 and tid in vocab.itos]
        token_strings = [vocab.decode([tid])[0] for tid in valid_token_ids]
        
        # Find admission tokens
        for j, token in enumerate(token_strings):
            if token == 'HOSPITAL_ADMISSION':
                admission_indices.append((i, j, token_strings))
                break  # Only need first admission per sample
    
    print(f"Found {len(admission_indices)} hospital admissions")
    return admission_indices

def predict_24hr_discharge(model, dataset, vocab, device, model_cfg, num_samples=1000, top_k_values=[1, 3, 5]):
    """
    Predict discharge within 24 hours of admission.
    
    Returns accuracy for predicting HOSPITAL_DISCHARGE token when 24+ hours have passed.
    """
    model.eval()
    predictions = []
    
    print(f"🏥 Evaluating 24hr discharge prediction on {num_samples} samples...")
    
    # Find admission indices
    admission_indices = find_admission_indices(dataset, vocab, num_samples)
    
    with torch.no_grad():
        for idx, (sample_idx, admission_pos, token_strings) in enumerate(admission_indices[:num_samples]):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{min(num_samples, len(admission_indices))}")
            
            # Get the sample
            sample = dataset[sample_idx]
            if isinstance(sample, tuple):
                context, timeline = sample
                input_ids = torch.cat([context, timeline]).unsqueeze(0).to(device)
            else:
                input_ids = sample.unsqueeze(0).to(device)
            
            # Look for positions where 24+ hours have passed since admission
            for pos in range(admission_pos + 1, min(len(token_strings), model_cfg.n_positions - 1)):
                if has_24hr_passed(token_strings, vocab, pos - admission_pos):
                    # Use context up to this position
                    context_ids = input_ids[:, :pos]
                    
                    # Get model prediction for next token
                    output = model(context_ids)
                    logits = output.logits[0, -1, :]
                    
                    # Get top-k predictions
                    top_logits, top_indices = torch.topk(logits, k=max(top_k_values))
                    top_probs = torch.softmax(top_logits, dim=-1)
                    
                    top_predictions = top_indices.cpu().tolist()
                    top_probabilities = top_probs.cpu().tolist()
                    
                    # Get true next token
                    true_token_id = input_ids[0, pos].item() if pos < input_ids.size(1) else vocab.stoi.get('<pad>', 0)
                    true_token_str = vocab.decode([true_token_id])[0]
                    
                    # Check if it's a discharge token
                    is_discharge = (true_token_str == 'HOSPITAL_DISCHARGE')
                    
                    # Store prediction
                    predictions.append({
                        'sample_id': sample_idx,
                        'admission_pos': admission_pos,
                        'prediction_pos': pos,
                        'true_token': true_token_id,
                        'true_token_str': true_token_str,
                        'top_predictions': top_predictions,
                        'top_probabilities': top_probabilities,
                        'is_discharge': is_discharge,
                        'hours_since_admission': pos - admission_pos  # Approximate
                    })
                    
                    # Only predict once per admission (at first 24hr+ point)
                    break
    
    print("Calculating discharge prediction metrics...")
    
    # Calculate metrics
    discharge_predictions = [p for p in predictions if p['is_discharge']]
    all_predictions = predictions
    
    metrics = {}
    
    # Overall accuracy for predicting discharge token
    for k in top_k_values:
        discharge_correct = sum(1 for p in discharge_predictions 
                              if p['true_token'] in p['top_predictions'][:k])
        total_discharge = len(discharge_predictions)
        
        if total_discharge > 0:
            metrics[f'discharge_top_{k}_accuracy'] = discharge_correct / total_discharge
        else:
            metrics[f'discharge_top_{k}_accuracy'] = 0.0
    
    # Binary classification: discharge vs non-discharge
    for k in top_k_values:
        y_true = [1 if p['is_discharge'] else 0 for p in all_predictions]
        y_pred_probs = []
        
        for p in all_predictions:
            # Check if HOSPITAL_DISCHARGE is in top-k predictions
            discharge_token_id = vocab.stoi.get('HOSPITAL_DISCHARGE', -1)
            if discharge_token_id in p['top_predictions'][:k]:
                # Get probability of discharge token
                discharge_idx = p['top_predictions'][:k].index(discharge_token_id)
                y_pred_probs.append(p['top_probabilities'][discharge_idx])
            else:
                y_pred_probs.append(0.0)
        
        if len(set(y_true)) > 1:
            metrics[f'discharge_binary_top_{k}_auroc'] = roc_auc_score(y_true, y_pred_probs)
            
            y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
            metrics[f'discharge_binary_top_{k}_f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
            metrics[f'discharge_binary_top_{k}_precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics[f'discharge_binary_top_{k}_recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        else:
            metrics[f'discharge_binary_top_{k}_auroc'] = 0.5
            metrics[f'discharge_binary_top_{k}_f1'] = 0.0
            metrics[f'discharge_binary_top_{k}_precision'] = 0.0
            metrics[f'discharge_binary_top_{k}_recall'] = 0.0
    
    metrics['total_predictions'] = len(all_predictions)
    metrics['discharge_events'] = len(discharge_predictions)
    
    return predictions, metrics

def predict_24hr_mortality(model, dataset, vocab, device, model_cfg, num_samples=1000, top_k_values=[1, 3, 5]):
    """
    Predict mortality within 24 hours of admission.
    
    Returns accuracy for predicting MEDS_DEATH token when 24+ hours have passed.
    """
    model.eval()
    predictions = []
    
    print(f"💀 Evaluating 24hr mortality prediction on {num_samples} samples...")
    
    # Find admission indices
    admission_indices = find_admission_indices(dataset, vocab, num_samples)
    
    with torch.no_grad():
        for idx, (sample_idx, admission_pos, token_strings) in enumerate(admission_indices[:num_samples]):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{min(num_samples, len(admission_indices))}")
            
            # Get the sample
            sample = dataset[sample_idx]
            if isinstance(sample, tuple):
                context, timeline = sample
                input_ids = torch.cat([context, timeline]).unsqueeze(0).to(device)
            else:
                input_ids = sample.unsqueeze(0).to(device)
            
            # Look for positions where 24+ hours have passed since admission
            for pos in range(admission_pos + 1, min(len(token_strings), model_cfg.n_positions - 1)):
                if has_24hr_passed(token_strings, vocab, pos - admission_pos):
                    # Use context up to this position
                    context_ids = input_ids[:, :pos]
                    
                    # Get model prediction for next token
                    output = model(context_ids)
                    logits = output.logits[0, -1, :]
                    
                    # Get top-k predictions
                    top_logits, top_indices = torch.topk(logits, k=max(top_k_values))
                    top_probs = torch.softmax(top_logits, dim=-1)
                    
                    top_predictions = top_indices.cpu().tolist()
                    top_probabilities = top_probs.cpu().tolist()
                    
                    # Get true next token
                    true_token_id = input_ids[0, pos].item() if pos < input_ids.size(1) else vocab.stoi.get('<pad>', 0)
                    true_token_str = vocab.decode([true_token_id])[0]
                    
                    # Check if it's a death token
                    is_death = (true_token_str == 'MEDS_DEATH')
                    
                    # Store prediction
                    predictions.append({
                        'sample_id': sample_idx,
                        'admission_pos': admission_pos,
                        'prediction_pos': pos,
                        'true_token': true_token_id,
                        'true_token_str': true_token_str,
                        'top_predictions': top_predictions,
                        'top_probabilities': top_probabilities,
                        'is_death': is_death,
                        'hours_since_admission': pos - admission_pos  # Approximate
                    })
                    
                    # Only predict once per admission (at first 24hr+ point)
                    break
    
    print("Calculating mortality prediction metrics...")
    
    # Calculate metrics
    death_predictions = [p for p in predictions if p['is_death']]
    all_predictions = predictions
    
    metrics = {}
    
    # Overall accuracy for predicting death token
    for k in top_k_values:
        death_correct = sum(1 for p in death_predictions 
                           if p['true_token'] in p['top_predictions'][:k])
        total_death = len(death_predictions)
        
        if total_death > 0:
            metrics[f'mortality_top_{k}_accuracy'] = death_correct / total_death
        else:
            metrics[f'mortality_top_{k}_accuracy'] = 0.0
    
    # Binary classification: death vs non-death
    for k in top_k_values:
        y_true = [1 if p['is_death'] else 0 for p in all_predictions]
        y_pred_probs = []
        
        for p in all_predictions:
            # Check if MEDS_DEATH is in top-k predictions
            death_token_id = vocab.stoi.get('MEDS_DEATH', -1)
            if death_token_id in p['top_predictions'][:k]:
                # Get probability of death token
                death_idx = p['top_predictions'][:k].index(death_token_id)
                y_pred_probs.append(p['top_probabilities'][death_idx])
            else:
                y_pred_probs.append(0.0)
        
        if len(set(y_true)) > 1:
            metrics[f'mortality_binary_top_{k}_auroc'] = roc_auc_score(y_true, y_pred_probs)
            
            y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
            metrics[f'mortality_binary_top_{k}_f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
            metrics[f'mortality_binary_top_{k}_precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics[f'mortality_binary_top_{k}_recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        else:
            metrics[f'mortality_binary_top_{k}_auroc'] = 0.5
            metrics[f'mortality_binary_top_{k}_f1'] = 0.0
            metrics[f'mortality_binary_top_{k}_precision'] = 0.0
            metrics[f'mortality_binary_top_{k}_recall'] = 0.0
    
    metrics['total_predictions'] = len(all_predictions)
    metrics['death_events'] = len(death_predictions)
    
    return predictions, metrics

def create_24hr_plots(discharge_results, mortality_results, output_dir, top_k_values=[1, 3, 5]):
    """Create plots for 24hr discharge and mortality predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    model_names = list(discharge_results.keys())
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    # 1. Discharge Accuracy Plot
    plt.figure(figsize=(12, 6))
    
    for k_idx, k in enumerate(top_k_values):
        accuracies = []
        for model_name in model_names:
            acc = discharge_results[model_name]['metrics'].get(f'discharge_top_{k}_accuracy', 0.0)
            accuracies.append(acc)
        
        plt.bar(x_pos + k_idx * width, accuracies, width, 
               label=f'Top-{k}', alpha=0.8)
    
    plt.title('24hr Discharge Prediction Accuracy')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(x_pos + width, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/24hr_discharge_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mortality Accuracy Plot
    plt.figure(figsize=(12, 6))
    
    for k_idx, k in enumerate(top_k_values):
        accuracies = []
        for model_name in model_names:
            acc = mortality_results[model_name]['metrics'].get(f'mortality_top_{k}_accuracy', 0.0)
            accuracies.append(acc)
        
        plt.bar(x_pos + k_idx * width, accuracies, width, 
               label=f'Top-{k}', alpha=0.8)
    
    plt.title('24hr Mortality Prediction Accuracy')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(x_pos + width, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/24hr_mortality_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Combined AUROC Plot
    plt.figure(figsize=(12, 6))
    
    for k_idx, k in enumerate(top_k_values):
        discharge_aurocs = []
        mortality_aurocs = []
        
        for model_name in model_names:
            discharge_auroc = discharge_results[model_name]['metrics'].get(f'discharge_binary_top_{k}_auroc', 0.5)
            mortality_auroc = mortality_results[model_name]['metrics'].get(f'mortality_binary_top_{k}_auroc', 0.5)
            discharge_aurocs.append(discharge_auroc)
            mortality_aurocs.append(mortality_auroc)
        
        plt.bar(x_pos - width/2 + k_idx * width, discharge_aurocs, width/2, 
               label=f'Discharge Top-{k}', alpha=0.8, color=f'C{k_idx}')
        plt.bar(x_pos + width/2 + k_idx * width, mortality_aurocs, width/2, 
               label=f'Mortality Top-{k}', alpha=0.8, color=f'C{k_idx}', hatch='//')
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    plt.title('24hr Prediction AUROC Comparison')
    plt.xlabel('Models')
    plt.ylabel('AUROC')
    plt.xticks(x_pos, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/24hr_auroc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main 24hr prediction evaluation function."""
    parser = argparse.ArgumentParser(description="24hr discharge and mortality prediction for EHR-FM models.")
    parser.add_argument('--config', type=str, default='evaluation_config.yaml',
                        help='Path to the evaluation configuration YAML file.')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = config['output_dir'] + "_24hr"
    os.makedirs(output_dir, exist_ok=True)
    
    discharge_results = {}
    mortality_results = {}
    top_k_values = config.get('top_k_values', [1, 3, 5])
    
    for model_name, model_config in config['models'].items():
        print(f"\n==================================================")
        print(f"Evaluating 24hr predictions for {model_name}")
        print(f"==================================================")
        
        model_path = Path(model_config['path'])
        data_dir = Path(config['data_dir'])
        num_samples = config.get('num_samples', 1000)
        
        # Load model and dataset
        model, vocab, device, model_cfg = load_model_and_config(model_path)
        dataset = load_dataset(data_dir, vocab, n_positions=model_cfg.n_positions)
        
        # Use the new function to find special tokens and predict
        special_results, all_preds = find_special_tokens_and_predict(
            model, dataset, vocab, device, model_cfg, 
            target_tokens=['HOSPITAL_DISCHARGE', 'MEDS_DEATH', 'HOSPITAL_ADMISSION'], 
            max_samples=num_samples, top_k=5
        )
        
        # Extract results for each event type
        discharge_preds = special_results.get('HOSPITAL_DISCHARGE', {}).get('predictions', [])
        mortality_preds = special_results.get('MEDS_DEATH', {}).get('predictions', [])
        admission_preds = special_results.get('HOSPITAL_ADMISSION', {}).get('predictions', [])
        
        discharge_metrics = special_results.get('HOSPITAL_DISCHARGE', {}).get('accuracies', {})
        mortality_metrics = special_results.get('MEDS_DEATH', {}).get('accuracies', {})
        admission_metrics = special_results.get('HOSPITAL_ADMISSION', {}).get('accuracies', {})
        
        discharge_results[model_name] = {
            'metrics': discharge_metrics,
            'predictions': discharge_preds
        }
        
        mortality_results[model_name] = {
            'metrics': mortality_metrics,
            'predictions': mortality_preds
        }
        
        print(f"✅ {model_name} evaluation completed")
        print(f"   Discharge events: {len(discharge_preds)} (acc: {discharge_metrics.get('top_1_accuracy', 0):.3f})")
        print(f"   Mortality events: {len(mortality_preds)} (acc: {mortality_metrics.get('top_1_accuracy', 0):.3f})")
        print(f"   Admission events: {len(admission_preds)} (acc: {admission_metrics.get('top_1_accuracy', 0):.3f})")
        print(f"   Total predictions: {len(discharge_preds) + len(mortality_preds) + len(admission_preds)}")
    
    # Save results as JSON
    results_json = {
        'discharge': {model: result['metrics'] for model, result in discharge_results.items()},
        'mortality': {model: result['metrics'] for model, result in mortality_results.items()}
    }
    
    results_json_path = f"{output_dir}/24hr_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n📊 24hr results saved to: {results_json_path}")
    
    # Save top-k predictions and probabilities as .npy files for each model
    print("Saving top-k predictions and probabilities as .npy files...")
    for model_name in discharge_results.keys():
        model_output_dir = f"{output_dir}/{model_name}"
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Discharge predictions
        discharge_preds = discharge_results[model_name]['predictions']
        if discharge_preds:
            discharge_topk_tokens = np.array([pred['top_predictions'] for pred in discharge_preds])
            discharge_topk_probs = np.array([pred['top_probabilities'] for pred in discharge_preds])
            
            np.save(f"{model_output_dir}/discharge_topk_tokens.npy", discharge_topk_tokens)
            np.save(f"{model_output_dir}/discharge_topk_probs.npy", discharge_topk_probs)
            print(f"  ✅ {model_name} discharge: {len(discharge_preds)} predictions saved")
        
        # Mortality predictions  
        mortality_preds = mortality_results[model_name]['predictions']
        if mortality_preds:
            mortality_topk_tokens = np.array([pred['top_predictions'] for pred in mortality_preds])
            mortality_topk_probs = np.array([pred['top_probabilities'] for pred in mortality_preds])
            
            np.save(f"{model_output_dir}/mortality_topk_tokens.npy", mortality_topk_tokens)
            np.save(f"{model_output_dir}/mortality_topk_probs.npy", mortality_topk_probs)
            print(f"  ✅ {model_name} mortality: {len(mortality_preds)} predictions saved")
    
    # Create plots
    print("Creating 24hr prediction plots...")
    create_24hr_plots(discharge_results, mortality_results, output_dir, top_k_values)
    
    print(f"\n✅ 24hr evaluation complete! Results and .npy files saved to: {output_dir}")

if __name__ == "__main__":
    main()
