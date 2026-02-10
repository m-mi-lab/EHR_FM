#!/bin/bash -l
#SBATCH --job-name=ehr_infer
#SBATCH --time=1-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:2
#SBATCH --output=infer_%j.log
#SBATCH --error=infer_%j.err

# Configuration
rep_start=1
rep_stop=20
config_file="${1:-scripts/eval_final_config.yaml}"  # Default config or from command line
output_base="infer_results_2"

# Read tasks from config file
tasks=($(python3 -c "import yaml; cfg=yaml.safe_load(open('${config_file}')); print(' '.join(cfg.get('tasks', [])))"))

echo "========================================================================"
echo "DIRECT PROBABILITY INFERENCE - ALL TASKS, 20 REPETITIONS EACH"
echo "========================================================================"
echo "Config:     ${config_file}"
echo "Output:     ${output_base}"
echo "Tasks:      ${tasks[@]}"
echo "Reps:       ${rep_start} to ${rep_stop}"
echo "========================================================================"
echo ""

# Loop through all tasks
for test_name in "${tasks[@]}"; do
    output_dir="${output_base}/${test_name}"
    mkdir -p ${output_dir}
    
    echo ""
    echo "========================================================================"
    echo "TASK: ${test_name}"
    echo "========================================================================"
    echo ""
    
    # Run 20 repetitions for this task with different seeds
    for i in $(seq ${rep_start} ${rep_stop}); do
        seed=$((42 + i))  # Seeds: 43, 44, 45, ..., 62
        echo "[Rep ${i}/${rep_stop}]: Running inference for ${test_name} (seed=${seed})"
        
        python scripts/infer_trajectory.py \
            --config ${config_file} \
            --test ${test_name} \
            --output ${output_dir} \
            --suffix rep${i} \
            --seed ${seed} \
            --num-trajectories 1 \
            --temperature 1.0 \
            --top-k 50 || exit 1
        
        echo "✅ Repetition ${i} complete"
        echo ""
    done
    
    echo ""
    echo "========================================================================"
    echo "AGGREGATING RESULTS FOR ${test_name}"
    echo "========================================================================"
    echo ""
    
    # Aggregate results using Python
    python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("${output_dir}")
test_name = "${test_name}"

# Load all repetition results
files = sorted(output_dir.glob(f"{test_name}_rep*.json"))
if not files:
    print(f"❌ No results found in {output_dir}")
    exit(1)

data = [json.load(open(f)) for f in files]
print(f"✅ Loaded {len(data)} repetitions")

# Aggregate metrics
aggregated = {"task": test_name, "num_repetitions": len(data)}

# Get all numeric metrics from first result
for key, value in data[0].items():
    if isinstance(value, (int, float)) and key not in ["num_patients"]:
        values = [d[key] for d in data if key in d]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_min"] = float(np.min(values))
        aggregated[f"{key}_max"] = float(np.max(values))

# Save aggregated results
agg_file = output_dir / f"{test_name}_aggregated.json"
with open(agg_file, 'w') as f:
    json.dump(aggregated, f, indent=2)

# Print summary
print("")
print("="*70)
print(f"AGGREGATED RESULTS: {test_name.upper()}")
print("="*70)
for key in sorted(aggregated.keys()):
    if key.endswith("_mean"):
        base = key.replace("_mean", "")
        mean = aggregated[f"{base}_mean"]
        std = aggregated[f"{base}_std"]
        print(f"{base:20s}: {mean:.4f} ± {std:.4f}")
print("="*70)
print(f"✅ Saved to {agg_file}")
EOF
    
    echo ""
    echo "✅ Task ${test_name} complete!"
    echo ""
done

echo ""
echo "========================================================================"
echo "ALL TASKS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to ${output_base}/"
for test_name in "${tasks[@]}"; do
    echo "  - ${test_name}/"
    echo "      ├── ${test_name}_rep*.json (20 repetitions)"
    echo "      ├── ${test_name}_rep*_plot.png (20 plots)"
    echo "      └── ${test_name}_aggregated.json"
done
echo ""
echo "========================================================================"
