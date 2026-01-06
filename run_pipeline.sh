#!/usr/bin/env bash
#
# EHR_FM Pipeline Runner
# Executes: Download -> MEDS ETL -> Tokenization -> Training
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   --download            Download MIMIC-IV from PhysioNet (requires credentials)
#   --skip-etl            Skip MEDS ETL step (use existing MEDS data)
#   --skip-tokenization   Skip tokenization step (use existing tokenized data)
#   --skip-training       Skip training step
#   --dry-run             Print commands without executing
#   --help                Show this help message
#
# Environment variables can be set via:
#   1. .env file in the same directory
#   2. CLI override: VAR=value ./run_pipeline.sh
#

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Flags
DO_DOWNLOAD=false
SKIP_ETL=false
SKIP_TOKENIZATION=false
SKIP_TRAINING=false
DRY_RUN=false

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_step() {
    echo ""
    echo "=============================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP: $*"
    echo "=============================================================================="
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        log "[DRY-RUN] $*"
    else
        log "Running: $*"
        eval "$@"
    fi
}

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

# ==============================================================================
# Parse CLI Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --download)
            DO_DOWNLOAD=true
            shift
            ;;
        --skip-etl)
            SKIP_ETL=true
            shift
            ;;
        --skip-tokenization)
            SKIP_TOKENIZATION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
done

# ==============================================================================
# Load Environment Variables
# ==============================================================================

# Load .env file if it exists
if [ -f "$ENV_FILE" ]; then
    log "Loading configuration from $ENV_FILE"
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
else
    log "Warning: No .env file found at $ENV_FILE"
    log "Using environment variables or defaults"
fi

# Set defaults for all variables
# Data paths (relative to script directory)
MIMICIV_RAW_DIR="${MIMICIV_RAW_DIR:-${SCRIPT_DIR}/data/mimic-iv-raw}"
MIMICIV_PRE_MEDS_DIR="${MIMICIV_PRE_MEDS_DIR:-${SCRIPT_DIR}/data/mimic-iv-pre-meds}"
MIMICIV_MEDS_DIR="${MIMICIV_MEDS_DIR:-${SCRIPT_DIR}/data/mimic-iv-meds}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${SCRIPT_DIR}/data/tokenized}"

# PhysioNet credentials (required for --download)
PHYSIONET_USERNAME="${PHYSIONET_USERNAME:-}"
PHYSIONET_PASSWORD="${PHYSIONET_PASSWORD:-}"

# Pipeline options
N_WORKERS="${N_WORKERS:-4}"
DO_UNZIP="${DO_UNZIP:-false}"

# Training configuration
MODEL_CONFIG="${MODEL_CONFIG:-gpt2_small_4exp}"
TRAIN_CONFIG="${TRAIN_CONFIG:-4_exp}"
WORLD_SIZE="${WORLD_SIZE:-auto}"

# MLflow configuration
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-EHR_FM}"

# Advanced options
VAL_SPLIT_FRACTION="${VAL_SPLIT_FRACTION:-0.05}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ==============================================================================
# Validate Required Variables
# ==============================================================================

validate_required() {
    local var_name="$1"
    local var_value="${!var_name:-}"
    
    if [ -z "$var_value" ]; then
        error "Required variable $var_name is not set. Check your .env file."
    fi
}

validate_path_exists() {
    local var_name="$1"
    local path="${!var_name:-}"
    
    if [ -z "$path" ]; then
        error "Required path variable $var_name is not set."
    fi
    
    if [ ! -d "$path" ]; then
        error "Directory $var_name=$path does not exist."
    fi
}

log "Validating configuration..."

# Required for training (unless skipping)
if [ "$SKIP_TRAINING" = false ]; then
    validate_required "MLFLOW_TRACKING_URI"
fi

# Required for download
if [ "$DO_DOWNLOAD" = true ]; then
    validate_required "PHYSIONET_USERNAME"
    validate_required "PHYSIONET_PASSWORD"
fi

# Required unless skipping ETL (and not downloading)
if [ "$SKIP_ETL" = false ] && [ "$DO_DOWNLOAD" = false ]; then
    validate_path_exists "MIMICIV_RAW_DIR"
fi

if [ "$SKIP_TOKENIZATION" = false ] && [ "$SKIP_ETL" = true ]; then
    validate_path_exists "MIMICIV_MEDS_DIR"
fi

if [ "$SKIP_TOKENIZATION" = true ] && [ "$SKIP_TRAINING" = false ]; then
    validate_path_exists "TOKENIZED_DIR"
fi

# ==============================================================================
# Print Configuration
# ==============================================================================

log ""
log "Configuration:"
log "  MIMICIV_RAW_DIR:        ${MIMICIV_RAW_DIR}"
log "  MIMICIV_PRE_MEDS_DIR:   ${MIMICIV_PRE_MEDS_DIR}"
log "  MIMICIV_MEDS_DIR:       ${MIMICIV_MEDS_DIR}"
log "  TOKENIZED_DIR:          ${TOKENIZED_DIR}"
log "  MLFLOW_TRACKING_URI:    ${MLFLOW_TRACKING_URI:-<not set>}"
log "  MLFLOW_EXPERIMENT_NAME: ${MLFLOW_EXPERIMENT_NAME}"
log "  PHYSIONET_USERNAME:     ${PHYSIONET_USERNAME:+<set>}${PHYSIONET_USERNAME:-<not set>}"
log "  MODEL_CONFIG:           ${MODEL_CONFIG}"
log "  TRAIN_CONFIG:           ${TRAIN_CONFIG}"
log "  WORLD_SIZE:             ${WORLD_SIZE}"
log "  N_WORKERS:              ${N_WORKERS}"
log "  DO_UNZIP:               ${DO_UNZIP}"
log ""
log "Flags:"
log "  DO_DOWNLOAD:            ${DO_DOWNLOAD}"
log "  SKIP_ETL:               ${SKIP_ETL}"
log "  SKIP_TOKENIZATION:      ${SKIP_TOKENIZATION}"
log "  SKIP_TRAINING:          ${SKIP_TRAINING}"
log "  DRY_RUN:                ${DRY_RUN}"
log ""

# ==============================================================================
# Step 0: Download MIMIC-IV (Optional)
# ==============================================================================

if [ "$DO_DOWNLOAD" = true ]; then
    log_step "Downloading MIMIC-IV from PhysioNet"
    
    # Create raw data directory
    run_cmd "mkdir -p '$MIMICIV_RAW_DIR'"
    
    log "Downloading MIMIC-IV v2.2 (this may take several hours)..."
    log "Note: You must have credentialed access to PhysioNet"
    
    # Download MIMIC-IV using wget
    if [ "$DRY_RUN" = true ]; then
        log "[DRY-RUN] wget -r -N -c -np --user '\${PHYSIONET_USERNAME}' --password '***' -P '${MIMICIV_RAW_DIR}' --cut-dirs=1 -nH 'https://physionet.org/files/mimiciv/2.2/'"
    else
        log "Running: wget to download MIMIC-IV (credentials hidden)"
        wget -r -N -c -np \
            --user "${PHYSIONET_USERNAME}" \
            --password "${PHYSIONET_PASSWORD}" \
            -P "${MIMICIV_RAW_DIR}" \
            --cut-dirs=1 \
            -nH \
            'https://physionet.org/files/mimiciv/2.2/'
    fi
    
    # Download metadata files from MIT-LCP
    log "Downloading MIMIC-IV metadata files..."
    MIMIC_METADATA_URL="https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map"
    
    METADATA_FILES=(
        "d_labitems_to_loinc.csv"
        "inputevents_to_rxnorm.csv"
        "lab_itemid_to_loinc.csv"
        "meas_chartevents_main.csv"
        "meas_chartevents_value.csv"
        "numerics-summary.csv"
        "outputevents_to_loinc.csv"
        "proc_datetimeevents.csv"
        "proc_itemid.csv"
        "waveforms-summary.csv"
    )
    
    for file in "${METADATA_FILES[@]}"; do
        run_cmd "wget -nc -P '${MIMICIV_RAW_DIR}' '${MIMIC_METADATA_URL}/${file}' || true"
    done
    
    log "MIMIC-IV download completed"
else
    log_step "Skipping MIMIC-IV download (use --download to enable)"
fi

# ==============================================================================
# Step 1: MEDS ETL
# ==============================================================================

if [ "$SKIP_ETL" = false ]; then
    log_step "Running MEDS ETL Pipeline"
    
    # Convert to absolute paths for ETL script (it runs from different directory)
    ABS_MIMICIV_RAW_DIR="$(cd "$SCRIPT_DIR" && realpath "$MIMICIV_RAW_DIR")"
    ABS_MIMICIV_PRE_MEDS_DIR="$(cd "$SCRIPT_DIR" && mkdir -p "$MIMICIV_PRE_MEDS_DIR" && realpath "$MIMICIV_PRE_MEDS_DIR")"
    ABS_MIMICIV_MEDS_DIR="$(cd "$SCRIPT_DIR" && mkdir -p "$MIMICIV_MEDS_DIR" && realpath "$MIMICIV_MEDS_DIR")"
    
    log "Using absolute paths:"
    log "  RAW_DIR:      $ABS_MIMICIV_RAW_DIR"
    log "  PRE_MEDS_DIR: $ABS_MIMICIV_PRE_MEDS_DIR"
    log "  MEDS_DIR:     $ABS_MIMICIV_MEDS_DIR"
    
    # Export for the ETL script
    export N_WORKERS
    export MIMICIV_RAW_DIR="$ABS_MIMICIV_RAW_DIR"
    export MIMICIV_PRE_MEDS_DIR="$ABS_MIMICIV_PRE_MEDS_DIR"
    export MIMICIV_MEDS_COHORT_DIR="$ABS_MIMICIV_MEDS_DIR"
    
    ETL_SCRIPT="${SCRIPT_DIR}/scripts/MEDS_transforms/mimic/run.sh"
    
    if [ ! -f "$ETL_SCRIPT" ]; then
        error "ETL script not found: $ETL_SCRIPT"
    fi
    
    run_cmd "cd '${SCRIPT_DIR}/scripts/MEDS_transforms/mimic' && ./run.sh '$ABS_MIMICIV_RAW_DIR' '$ABS_MIMICIV_PRE_MEDS_DIR' '$ABS_MIMICIV_MEDS_DIR' do_unzip=$DO_UNZIP"
    
    log "MEDS ETL completed successfully"
else
    log_step "Skipping MEDS ETL (--skip-etl)"
fi

# ==============================================================================
# Step 2: Tokenization
# ==============================================================================

if [ "$SKIP_TOKENIZATION" = false ]; then
    log_step "Running Tokenization"
    
    # Determine input directory for tokenization
    MEDS_TRAIN_DIR="${MIMICIV_MEDS_DIR}/data/train"
    
    if [ "$DRY_RUN" = false ] && [ ! -d "$MEDS_TRAIN_DIR" ]; then
        error "MEDS train directory not found: $MEDS_TRAIN_DIR"
    fi
    
    # Create tokenized output directory
    run_cmd "mkdir -p '$TOKENIZED_DIR'"
    
    # Run tokenization with Hydra overrides
    run_cmd "cd '$SCRIPT_DIR' && python3 -m src.tokenizer.run_tokenization \
        input_dir='$MEDS_TRAIN_DIR' \
        output_dir='$TOKENIZED_DIR'"
    
    log "Tokenization completed successfully"
else
    log_step "Skipping Tokenization (--skip-tokenization)"
fi

# ==============================================================================
# Step 3: Determine Tokenized Data Path
# ==============================================================================

# Find the actual tokenized dataset directory
# The tokenization script creates a subdirectory like "mimic_train"
if [ "$DRY_RUN" = true ]; then
    FINAL_TOKENIZED_DIR="${TOKENIZED_DIR}/mimic_train"
    log "[DRY-RUN] Would use tokenized data from: $FINAL_TOKENIZED_DIR"
else
    if [ -d "${TOKENIZED_DIR}/mimic_train" ]; then
        FINAL_TOKENIZED_DIR="${TOKENIZED_DIR}/mimic_train"
    else
        # Try to find any directory with vocab file
        VOCAB_FILE=$(find "$TOKENIZED_DIR" -name "vocab_t*.csv" -type f 2>/dev/null | head -1)
        if [ -n "$VOCAB_FILE" ]; then
            FINAL_TOKENIZED_DIR="$(dirname "$VOCAB_FILE")"
        else
            FINAL_TOKENIZED_DIR="$TOKENIZED_DIR"
        fi
    fi

    log "Using tokenized data from: $FINAL_TOKENIZED_DIR"

    # Validate tokenized data exists (only if training)
    if [ "$SKIP_TRAINING" = false ]; then
        if [ ! -f "${FINAL_TOKENIZED_DIR}/static_data.pickle" ]; then
            error "Tokenized data not found in $FINAL_TOKENIZED_DIR (missing static_data.pickle)"
        fi
    fi
fi

# ==============================================================================
# Step 4: Training
# ==============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    log_step "Running Training"

    # Export MLflow configuration
    export MLFLOW_TRACKING_URI
    export MLFLOW_EXPERIMENT_NAME

    # Export authentication if provided
    if [ -n "${MLFLOW_TRACKING_USERNAME:-}" ]; then
        export MLFLOW_TRACKING_USERNAME
    fi
    if [ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
        export MLFLOW_TRACKING_PASSWORD
    fi
    if [ -n "${MLFLOW_TRACKING_TOKEN:-}" ]; then
        export MLFLOW_TRACKING_TOKEN
    fi

    # Build training command
    TRAIN_CMD="python3 -m src.train"
    TRAIN_CMD+=" model=${MODEL_CONFIG}"
    TRAIN_CMD+=" train=${TRAIN_CONFIG}"
    TRAIN_CMD+=" world_size=${WORLD_SIZE}"
    TRAIN_CMD+=" data.tokenized_dir='${FINAL_TOKENIZED_DIR}'"
    TRAIN_CMD+=" data.val_split_fraction=${VAL_SPLIT_FRACTION}"

    # Add resume checkpoint if specified
    if [ -n "$RESUME_CHECKPOINT" ]; then
        if [ ! -f "$RESUME_CHECKPOINT" ]; then
            error "Resume checkpoint not found: $RESUME_CHECKPOINT"
        fi
        TRAIN_CMD+=" train.resume_from_checkpoint='${RESUME_CHECKPOINT}'"
    fi

    run_cmd "cd '$SCRIPT_DIR' && $TRAIN_CMD"
    
    log "Training completed successfully"
else
    log_step "Skipping Training (--skip-training)"
fi

# ==============================================================================
# Done
# ==============================================================================

log_step "Pipeline Complete"

if [ "$SKIP_TRAINING" = false ]; then
    log "Training outputs saved to: ${SCRIPT_DIR}/outputs/"
    log "Checkpoints logged to MLflow: ${MLFLOW_TRACKING_URI}"
    log ""
    log "To view MLflow experiments:"
    log "  Open ${MLFLOW_TRACKING_URI} in your browser"
fi
log ""

