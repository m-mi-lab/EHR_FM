#!/usr/bin/env python3
"""
Final Comprehensive Evaluation Script for EHR Foundation Model
(CONFIDENCE-WEIGHTED TRAJECTORY VERSION)

Evaluates model on multiple clinical prediction tasks using
confidence-weighted trajectory rollouts.

Probability is computed as:
    sum(confidence of positive outcome tokens)
    ------------------------------------------------
    sum(confidence of all outcome tokens)

This preserves ETHOS-style trajectory simulation while improving
calibration and AUROC smoothness.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, f1_score, precision_score, recall_score
)
from scipy.stats import norm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer.datasets.hospital_mortality import HospitalMortalityDataset
from src.tokenizer.datasets.mimic_icu import ICUMortalityDataset, ICUReadmissionDataset
from src.tokenizer.datasets.readmission import ReadmissionDataset
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.constants import SpecialToken as ST
from src.utils import load_env, resolve_model_path
from omegaconf import OmegaConf
from transformers import GPT2Config
from src.model import GPT2LMNoBiasModel


# =========================
# Task configuration
# =========================

TASK_CONFIGS = {
    "hosp_mortality": {
        "dataset_class": HospitalMortalityDataset,
        "positive_token": ST.DEATH,
        "negative_tokens": [ST.DISCHARGE],
        "description": "Hospital Mortality Prediction",
        "task_type": "binary",
    },
    "icu_mortality": {
        "dataset_class": ICUMortalityDataset,
        "positive_token": ST.DEATH,
        "negative_tokens": [ST.ICU_DISCHARGE],
        "description": "ICU Mortality Prediction",
        "task_type": "binary",
    },
    "hosp_readmission": {
        "dataset_class": ReadmissionDataset,
        "positive_token": ST.ADMISSION,
        "negative_tokens": [ST.DEATH, ST.TIMELINE_END],
        "description": "Hospital 30-day Readmission",
        "task_type": "binary",
    },
    "icu_readmission": {
        "dataset_class": ICUReadmissionDataset,
        "positive_token": ST.ICU_ADMISSION,
        "negative_tokens": [ST.DISCHARGE, ST.DEATH],
        "description": "ICU Readmission",
        "task_type": "binary",
    },
}


# =========================
# Model loading
# =========================

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


# =========================
# Trajectory generation
# =========================

def generate_single_trajectory_confidence(
    model,
    context,
    vocab_size,
    device,
    outcome_token_ids,
    max_tokens=10000,
    temperature=1.0,
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ctx = context.clone()
    with torch.no_grad():
        for step in range(max_tokens):
            if ctx.size(0) > 2048:
                ctx = ctx[-2048:]

            logits = model(ctx.unsqueeze(0).to(device)).logits[0, -1, :vocab_size]
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)

            token = torch.multinomial(probs, 1).item()
            prob = probs[token].item()

            if token in outcome_token_ids:
                return {
                    "outcome_token": token,
                    "outcome_prob": prob,
                    "outcome_position": step,
                }

            ctx = torch.cat([ctx, torch.tensor([token])])

    return {"outcome_token": None, "outcome_prob": None, "outcome_position": None}


# =========================
# Patient-level prediction
# =========================

def run_prediction_for_patient_confidence(
    model,
    context,
    ground_truth_outcome,
    vocab,
    device,
    positive_token_id,
    negative_token_ids,
    num_trajectories=20,
    max_tokens=10000,
    temperature=1.0,
    base_seed=42,
):
    vocab_size = len(vocab)
    outcome_ids = [positive_token_id] + negative_token_ids

    pos_conf, neg_conf = 0.0, 0.0
    trajectories = []

    for i in range(num_trajectories):
        traj = generate_single_trajectory_confidence(
            model,
            context,
            vocab_size,
            device,
            outcome_ids,
            max_tokens,
            temperature,
            base_seed + i,
        )
        trajectories.append(traj)

        if traj["outcome_token"] == positive_token_id:
            pos_conf += traj["outcome_prob"] or 0.0
        elif traj["outcome_token"] in negative_token_ids:
            neg_conf += traj["outcome_prob"] or 0.0

    total = pos_conf + neg_conf
    positive_probability = pos_conf / total if total > 0 else 0.5
    predicted_outcome = (
        positive_token_id if positive_probability > 0.5 else negative_token_ids[0]
    )

    return {
        "trajectories": trajectories,
        "positive_probability": positive_probability,
        "predicted_outcome": predicted_outcome,
        "ground_truth_outcome": ground_truth_outcome,
        "is_correct": predicted_outcome == ground_truth_outcome,
    }


# =========================
# Evaluation loop
# =========================

def evaluate_task(
    model,
    dataset,
    vocab,
    device,
    task_name,
    task_config,
    num_patients,
    num_trajectories,
    max_tokens,
    temperature,
    base_seed,
):
    pos_id = vocab.stoi[task_config["positive_token"]]
    neg_ids = [vocab.stoi[t] for t in task_config["negative_tokens"]]

    results = []

    for i in tqdm(range(num_patients), desc=f"{task_name}"):
        ctx, meta = dataset[i]
        ctx = ctx[ctx >= 0]

        gt_token = meta["expected"]
        gt_id = pos_id if gt_token == task_config["positive_token"] else vocab.stoi[gt_token]

        res = run_prediction_for_patient_confidence(
            model,
            ctx,
            gt_id,
            vocab,
            device,
            pos_id,
            neg_ids,
            num_trajectories,
            max_tokens,
            temperature,
            base_seed + i * 1000,
        )
        results.append(res)

    return results


# =========================
# Metrics
# =========================

def fit_gaussian_roc(fpr, tpr):
    mask = (fpr > 0) & (fpr < 1) & (tpr > 0) & (tpr < 1)
    if mask.sum() < 3:
        return None, None

    z_fpr = norm.ppf(fpr[mask])
    z_tpr = norm.ppf(tpr[mask])

    a, b = np.polyfit(z_fpr, z_tpr, 1)
    d_prime = b * np.sqrt(1 + a ** 2)
    auroc = norm.cdf(d_prime / np.sqrt(1 + a ** 2))

    return auroc, {"d_prime": d_prime, "sigma_ratio": a}


def calculate_task_statistics(results, positive_token_id):
    y_true = np.array([1 if r["ground_truth_outcome"] == positive_token_id else 0 for r in results])
    y_score = np.array([r["positive_probability"] for r in results])
    y_pred = (y_score > 0.5).astype(int)

    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc_gauss, gauss_params = fit_gaussian_roc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "total_patients": len(results),
        "accuracy": (y_pred == y_true).mean(),
        "auroc": auroc,
        "auroc_gaussian_fit": auroc_gauss,
        "gaussian_params": gauss_params,
        "auprc": auprc,
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval_final_config.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="eval_confidence_results")
    args = parser.parse_args()

    load_env(None)

    # Register resolver only if not already registered
    if not OmegaConf.has_resolver("oc.env"):
        OmegaConf.register_new_resolver(
            "oc.env",
            lambda k, default=None: os.environ.get(k, default)
        )

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Get model path from args or config
    model_spec = args.model or cfg.get("model")
    if model_spec is None:
        raise ValueError(
            "No model specified. Please provide --model argument or ensure "
            "config file has 'model' key, or set BEST_MODEL_PATH environment variable."
        )
    
    model_path = resolve_model_path(model_spec)
    model, vocab, device, model_cfg = load_model(model_path)

    print("Resolved model path:", cfg.get("model"))
    print("Env model path:", os.environ.get("BEST_MODEL_PATH"))


    all_results = {}

    for task_name, task_cfg in TASK_CONFIGS.items():
        dataset = task_cfg["dataset_class"](
            input_dir=Path(args.data_dir or cfg["data_dir"]),
            n_positions=model_cfg.n_positions,
        )

        results = evaluate_task(
            model,
            dataset,
            vocab,
            device,
            task_name,
            task_cfg,
            cfg.get("num_patients", 100),
            cfg.get("num_trajectories_per_patient", 20),
            cfg.get("max_tokens", 10000),
            cfg.get("temperature", 1.0),
            cfg.get("base_seed", 42),
        )

        stats = calculate_task_statistics(
            results, vocab.stoi[task_cfg["positive_token"]]
        )
        all_results[task_name] = stats
        print(task_name, stats["auroc"])

    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
