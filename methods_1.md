# Methodology: Direct Probability Evaluation

## Overview
We evaluate our EHR foundation model on five clinical prediction tasks using a direct probability extraction method. Unlike trajectory-based sampling approaches that require multiple forward passes per patient, our method performs a single forward pass and directly extracts outcome probabilities from the model's softmax distribution, providing both computational efficiency and deterministic results.

## Patient Timeline Construction
For each prediction task, patient timelines are constructed from the complete longitudinal EHR, including static demographics (age, sex, race) and temporal sequences of clinical events (admissions, discharges, diagnoses, procedures, medications, laboratory results, and vital signs). Each timeline is tokenized using a domain-specific vocabulary and truncated to a maximum context window of 2,048 tokens. When patient history exceeds this limit, the most recent tokens are retained.

## Prediction Point Definition
The prediction point varies by task:
- **Hospital Mortality**: Hospital admission token + 2 token offset
- **ICU Mortality**: ICU admission token + 1 token offset  
- **Hospital Readmission**: Hospital discharge token
- **ICU Readmission**: ICU discharge token
- **ICU Length of Stay**: ICU admission token + 1 token offset

At each prediction point, the model receives all historical information up to and including the specified offset, simulating a real-world scenario where clinicians must make predictions upon patient admission or discharge with access to complete prior medical history.

## Ground Truth Matching
For patients with multiple hospital or ICU visits, each admission-discharge cycle is treated as an independent sample. Ground truth outcomes are matched to prediction points using temporal ordering: for each admission token, we identify the next occurring outcome token (DISCHARGE or DEATH) in the timeline using binary search. This ensures each sample has a well-defined, temporally-consistent ground truth label.

## Binary Classification Tasks
For mortality and readmission prediction, we extract the model's next-token prediction logits at the prediction point and compute softmax probabilities. The probability of the positive outcome (e.g., death or readmission) is calculated as:

$$P(\text{positive}) = \frac{\text{softmax}[\text{positive\_token}]}{\text{softmax}[\text{positive\_token}] + \sum_{i} \text{softmax}[\text{negative\_token}_i]}$$

This normalization restricts the probability space to clinically relevant outcomes, excluding irrelevant tokens. We evaluate performance using Area Under the Receiver Operating Characteristic curve (AUROC), Area Under the Precision-Recall curve (AUPRC), and classification accuracy with a 0.5 decision threshold.

## Regression Task: ICU Length of Stay
For ICU LOS prediction, we adopt a simplified approach compatible with single forward passes. We first extract discharge and death probabilities as described above. Following ETHOS methodology, patients with death probability >0.5 are excluded from LOS calculations. For remaining patients, we estimate LOS using a heuristic baseline of 2.5 days (MIMIC-IV median ICU stay) inversely weighted by discharge probability, capped between 0.1 and 30 days. Ground truth LOS is computed from dataset metadata as the time difference between ICU admission and discharge timestamps. We report Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Pearson correlation coefficient.

## Computational Efficiency
The direct probability method requires only one forward pass per patient (O(1) per sample), compared to trajectory-based methods that require 10-20 forward passes for frequency-based probability estimation (O(n) per sample). This results in approximately 10× speedup while maintaining deterministic, reproducible results without sampling variance.

## Tasks Evaluated

### 1. Hospital Mortality Prediction
- **Input**: Patient timeline up to hospital admission + 2 tokens
- **Output**: Probability of in-hospital death vs. discharge
- **Dataset**: All hospital admissions in MIMIC-IV
- **Metrics**: AUROC, AUPRC, Accuracy

### 2. ICU Mortality Prediction
- **Input**: Patient timeline up to ICU admission + 1 token
- **Output**: Probability of ICU death vs. ICU discharge
- **Dataset**: All ICU admissions in MIMIC-IV
- **Metrics**: AUROC, AUPRC, Accuracy

### 3. Hospital 30-day Readmission
- **Input**: Patient timeline up to hospital discharge
- **Output**: Probability of readmission within 30 days
- **Dataset**: Hospital discharges with 30-day follow-up window
- **Metrics**: AUROC, AUPRC, Accuracy

### 4. ICU Readmission
- **Input**: Patient timeline up to ICU discharge
- **Output**: Probability of ICU readmission during same hospitalization
- **Dataset**: ICU discharges within hospital stays
- **Metrics**: AUROC, AUPRC, Accuracy

### 5. ICU Length of Stay
- **Input**: Patient timeline up to ICU admission + 1 token
- **Output**: Predicted length of stay in days
- **Dataset**: ICU admissions ending in discharge (deaths excluded)
- **Metrics**: MAE, RMSE, Pearson correlation

## Implementation Details
- **Model Architecture**: GPT-2 based transformer with domain-specific tokenization
- **Context Window**: 2,048 tokens (including patient demographics)
- **Evaluation Set**: 100-1000 patients per task (configurable)
- **Hardware**: Single GPU evaluation
- **Runtime**: ~20 seconds per 1000 patients (vs. ~200 seconds for trajectory methods)




p_id_x : [pc, adm1, dsc1, adm2, dsc2, adm3, dth3]
samples1: [pc, adm1, dsc1]
samples2: [pc, adm1, dsc1, adm2, dsc2]

mortality -->
input: [pc, adm1, dsc1, adm2, dsc2, < 2048 | adm3 + offset]
gt: dth3

