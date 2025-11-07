# EHR Timeline Creation and Structure

## Overview

Patient timelines are chronologically ordered sequences of medical events, measurements, and metadata that represent a patient's healthcare journey. Each timeline combines:

- **Static patient context** (demographics, age)
- **Dynamic medical events** (lab results, vital signs, admissions)
- **Temporal relationships** (time intervals between events)
- **Quantized values** (binned measurements for standardization)

## Timeline Structure

### 1. Patient Context (Static Data)
Each timeline begins with patient context that remains constant:

```
[BMI//Q7] [GENDER//F] [MARITAL//MARRIED] [Q7] [Q2] [RACE//WHITE]
```

**Components:**
- `BMI//Q[1-10]`: Body Mass Index quantile (Q1=lowest, Q10=highest)
- `GENDER//[M|F]`: Patient gender
- `MARITAL//[STATUS]`: Marital status (MARRIED, SINGLE, DIVORCED, etc.)
- Age tokens: Two quantile tokens representing patient age at timeline start
- `RACE//[CATEGORY]`: Racial/ethnic category

### 2. Dynamic Timeline Events
Medical events are ordered chronologically with embedded time intervals:

```
[12h-18h] [LAB//50965//PG/ML] [Q8] [30d-2mt] [VITAL//BLOOD_PRESSURE] [Q7] [Q7] [7d-12d] [ADMISSION_TYPE//EMERGENCY]
```

## Tokenization Process

### Stage 1: Data Collection and Sorting
1. **Raw MEDS data** is loaded from parquet files
2. Events are **sorted by patient ID and timestamp** (`subject_id`, `time`)
3. **Filtering** removes invalid timestamps (< 1970)

### Stage 2: Code Processing and Quantization
1. **Laboratory values** are quantized into 10 bins (Q1-Q10)
   - Q1: Bottom 10% of values
   - Q10: Top 10% of values
2. **Medical codes** are standardized and mapped to vocabulary
3. **Static demographics** are collected per patient

### Stage 3: Time Interval Injection
Time gaps between consecutive events are converted to interval tokens:

**Time Interval Hierarchy:**
```yaml
5m-15m:     5+ minutes
15m-45m:    15+ minutes  
45m-1h15m:  45+ minutes
1h15m-2h:   1h 15m+
2h-3h:      2+ hours
3h-5h:      3+ hours
5h-8h:      5+ hours
8h-12h:     8+ hours
12h-18h:    12+ hours
18h-1d:     18+ hours
1d-2d:      1+ day
2d-4d:      2+ days
4d-7d:      4+ days
7d-12d:     7+ days
12d-20d:    12+ days
20d-30d:    20+ days
30d-2mt:    30+ days
2mt-6mt:    2+ months
≥6mt:       6+ months
```

**Algorithm:**
- Calculate time difference between consecutive events
- Insert appropriate interval token(s) based on gap duration
- Large gaps get multiple tokens (e.g., 1-year gap = multiple `≥6mt` tokens)

### Stage 4: Timeline Termination
Each timeline ends with special tokens:
- `TIMELINE_END`: Normal end of patient data
- `MEDS_DEATH`: Patient death event
- `HOSPITAL_DISCHARGE`: End at discharge (for specific tasks)

## Token Categories

### Medical Events
- `LAB//[CODE]//[UNIT]`: Laboratory test results
  - Example: `LAB//50965//PG/ML` (lab code 50965, picograms per milliliter)
- `VITAL//[TYPE]`: Vital sign measurements
  - Example: `VITAL//BLOOD_PRESSURE`

### Quantile Values
- `Q[1-10]`: Binned measurement values
  - Always follow the measurement they quantify
  - Enable standardization across different lab ranges

### Administrative Events
- `ADMISSION_TYPE//[TYPE]`: Hospital admission type
  - `EMERGENCY`, `SCHEDULED`, `OBSERVATION`
- `DISCHARGE_LOCATION//[LOCATION]`: Discharge destination
  - `HOME`, `HEALTHCARE_FACILITY`, `DIED`, `HOSPICE`

### Demographics
- `BMI//Q[1-10]` or `BMI//UNKNOWN`: Body mass index
- `GENDER//[M|F]`: Patient gender
- `RACE//[CATEGORY]`: Racial category
- `MARITAL//[STATUS]`: Marital status

## Data Flow Pipeline

```
Raw MEDS Data
    ↓
Filter & Sort by (patient_id, time)
    ↓
Extract Static Demographics
    ↓
Process Medical Codes → Vocabulary Mapping
    ↓
Quantize Lab Values → Q1-Q10 Bins
    ↓
Inject Time Intervals Between Events
    ↓
Add Timeline Termination Tokens
    ↓
Tensorize → SafeTensors Format
    ↓
Final Timeline Dataset
```



## Implementation Details

### Patient Context Creation
```python
def _get_patient_context(self, idx: int) -> torch.Tensor:
    # Extract patient demographics at timeline start
    # Convert age to two quantile tokens
    # Handle missing data with UNKNOWN tokens
```

### Time Interval Injection
```python
def inject_time_intervals(df, time_intervals_spec):
    # Calculate time differences between consecutive events
    # Map time gaps to predefined interval categories  
    # Insert interval tokens into event sequence
    # Handle timeline boundaries
```

### Tensorization
```python
def tensorize(in_fp, out_fp, vocab):
    # Convert codes to token IDs using vocabulary
    # Create patient offset indices for efficient access
    # Store as SafeTensors with metadata
```

## Timeline Properties

- **Context Size**: ~6 tokens (demographics + age)
- **Timeline Size**: `n_positions - context_size` (typically ~2042 tokens)
- **Vocabulary Size**: ~4,432 unique tokens
- **Dataset Size**: ~298M timeline positions (MIMIC-IV)

## Usage in Model Training

1. **Context tokens** provide patient background
2. **Timeline tokens** create chronological sequence
3. **Model learns** to predict next medical event
4. **Attention mechanism** captures temporal dependencies
5. **MoE routing** specializes on different medical domains

## File Outputs

The tokenization process generates:
- `*.safetensors`: Tensorized timeline data
- `vocab_t*.csv`: Token vocabulary
- `static_data.pickle`: Patient demographics
- `quantiles.json`: Value quantization bins
- `interval_estimates.json`: Time interval statistics

This structure enables the foundation model to learn rich temporal patterns in electronic health records while maintaining computational efficiency through quantization and standardized tokenization.
