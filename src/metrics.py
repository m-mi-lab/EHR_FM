from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from transformers import PreTrainedModel


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, ctx, get_batch: Callable, eval_iters: int, tokens_of_interest: dict
):
    """
    Estimate loss on train and validation datasets.
    Fixed version to avoid hangs and memory issues.
    """
    block_size = model.config.n_positions if hasattr(model, 'config') else 2048
    if block_size > 10000:  # Sanity check
        block_size = 2048
    block_thresh = min(block_size // 2, 1024)  # Cap threshold to avoid memory issues
    
    # Initialize results dictionary
    out = {}
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate on training and validation splits
    for split in ["train", "val"]:
        # Store losses across evaluation iterations
        losses = []
        
        # Store results for tokens
        all_tokens_res = defaultdict(list)
        toi_res = defaultdict(list)
        
        # Run evaluation for specified number of iterations
        for i in range(eval_iters):
            try:
                # Get batch data
                X, Y = get_batch(split)
                
                # Forward pass with context manager (e.g., autocast)
                with ctx:
                    # Handle both encoder-decoder and decoder-only models
                    if isinstance(X, tuple):
                        output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                    else:
                        output = model(input_ids=X, labels=Y)
                    
                    # Get loss and logits from model output
                    if hasattr(output, 'loss'):
                        loss = output.loss
                        logits = output.logits
                    else:
                        # Handle different output formats
                        loss = output[0] if isinstance(output, tuple) else output
                        logits = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                
                # Store loss
                losses.append(float(loss.item()))
                
                # Calculate accuracy metrics for validation set
                if split == "val" and logits is not None:
                    # Top-k accuracy for all tokens
                    for k in [1, 3, 5]:
                        try:
                            acc = top_k_accuracy(logits, Y, k=k, threshold=block_thresh)
                            all_tokens_res[f"acc_top/all/{split}/{block_thresh}/k={k}"].append(acc)
                        except Exception as e:
                            print(f"Error calculating accuracy for k={k}: {e}")
                    
                    # Top-k accuracy for special tokens
                    for stoken, token in tokens_of_interest.items():
                        for k in [1, 3, 10]:
                            try:
                                acc = top_k_acc_special(logits, Y, token, k=k, threshold=block_thresh)
                                toi_res[f"acc_top/{stoken}/{split}/{block_thresh}/k={k}"].append(acc)
                            except Exception as e:
                                print(f"Error calculating special token accuracy for {stoken}, k={k}: {e}")
            
            except Exception as e:
                print(f"Error during evaluation iteration {i} for {split}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next iteration instead of breaking
                continue
        
        # Calculate mean loss
        if losses:
            out[f"loss/{split}"] = sum(losses) / len(losses)
        else:
            out[f"loss/{split}"] = float('nan')
        
        # Calculate metrics for validation set
        if split == "val":
            # Calculate mean accuracy for all tokens
            for test_name, values in all_tokens_res.items():
                if values:
                    out[test_name] = sum(values) / len(values)
                else:
                    out[test_name] = float('nan')
            
            # Calculate weighted mean accuracy for special tokens
            for test_name, values in toi_res.items():
                if values:
                    weights_sum = sum(x[1] for x in values) if isinstance(values[0], tuple) else len(values)
                    if weights_sum > 0:
                        if isinstance(values[0], tuple):
                            out[test_name] = sum(x[0] * x[1] for x in values) / weights_sum
                        else:
                            out[test_name] = sum(values) / len(values)
                    else:
                        out[test_name] = float('nan')
                else:
                    out[test_name] = float('nan')
    
    # Check for MoE losses
    if hasattr(model, 'config') and hasattr(model.config, 'n_experts') and model.config.n_experts > 1:
        # Add MoE-specific losses if available
        if hasattr(output, 'aux_loss') and output.aux_loss is not None:
            out['moe/aux_loss'] = float(output.aux_loss.item())
        if hasattr(output, 'router_z_loss') and output.router_z_loss is not None:
            out['moe/router_z_loss'] = float(output.router_z_loss.item())
    
    # Set model back to training mode
    model.train()
    
    return out



def compute_weighted_mean(v):
    weights = sum(x[1] for x in v)
    if weights == 0:
        return torch.nan
    return sum(x[0] * x[1] for x in v) / weights


@torch.jit.script
def top_k_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int, threshold: int) -> float:
    """Calculate top-k accuracy for all tokens."""
    # Safety checks
    seq_len = min(logits.size(1), y.size(1))
    threshold = min(threshold, seq_len - 1)  # Ensure threshold is valid
    
    # If threshold is negative or sequence is too short, return 0
    if threshold < 0 or seq_len <= threshold:
        return 0.0
    
    # Slice tensors to appropriate sizes
    logits = logits[:, threshold:seq_len, :]
    y_true = y[:, threshold:seq_len]
    
    # Ensure y_true has correct shape for comparison
    y_true = y_true.unsqueeze(-1)
    
    # Cap k to vocab size
    k = min(k, logits.size(-1))
    
    # Get top-k predictions
    _, indices = torch.topk(logits, k, dim=-1)
    
    # Check if true label is in top-k predictions
    correct = (indices == y_true).any(dim=-1)
    
    # Calculate accuracy
    if correct.numel() > 0:
        return float(correct.sum() / correct.numel())
    else:
        return 0.0

@torch.jit.script
def top_k_acc_special(
    logits: torch.Tensor, y: torch.Tensor, token_of_interest: int, k: int, threshold: int
) -> tuple[float, int]:
    """Calculate top-k accuracy for specific tokens of interest."""
    # Safety checks
    seq_len = min(logits.size(1), y.size(1))
    threshold = min(threshold, seq_len - 1)  # Ensure threshold is valid
    
    # If threshold is negative or sequence is too short, return 0
    if threshold < 0 or seq_len <= threshold:
        return 0.0, 0
    
    # Slice tensors to appropriate sizes
    logits = logits[:, threshold:seq_len, :]
    y_true = y[:, threshold:seq_len]
    
    # Add dimension for comparison
    y_true = y_true.unsqueeze(-1)
    
    # Identify positions where token of interest appears
    interested = (y_true == token_of_interest)
    
    # If token doesn't appear, return 0
    weight = interested.sum()
    if weight == 0:
        return 0.0, 0
    
    # Cap k to vocab size
    k = min(k, logits.size(-1))
    
    # Get top-k predictions
    _, indices = torch.topk(logits, k, dim=-1)
    
    # Check if true label is in top-k predictions
    correct = (indices == y_true).any(dim=-1, keepdim=True)
    
    # Calculate accuracy for token of interest
    score = (correct & interested).sum() / weight
    
    return score.item(), weight.item()