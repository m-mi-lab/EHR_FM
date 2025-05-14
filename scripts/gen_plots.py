import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import yaml
import datetime

def generate_config_yaml(log_dir, config_path):
    """Generate a config YAML file with default settings if one doesn't exist."""
    # Find all event files recursively
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    # Extract unique run paths relative to log_dir and sort them for consistent order
    run_paths = sorted(list(set([os.path.relpath(os.path.dirname(file), log_dir) for file in event_files])))
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    
    config = {
        "title": f"Experiment Results ({timestamp})",
        "max_steps": 25000,  # Default maximum steps to process for each run
        "smoothing_weight": 0.8,  # Default smoothing weight
        "runs": {}
    }
    
    # Add each run with a default legend name
    for i, run_path in enumerate(run_paths):
        # Extract date from path if possible
        # (Assumes path components like YYYY-MM-DD are part of the run_path)
        date_parts = [part for part in run_path.split(os.sep) if part.replace("-", "").isdigit() and len(part) >= 8]
        date_str = date_parts[0] if date_parts else ""
        
        default_name = f"Run {i+1}" + (f" ({date_str})" if date_str else "")
        
        config["runs"][run_path] = {
            "legend": default_name,
            "color": None,  # Will be auto-assigned by seaborn if None
            "linestyle": "-",
            "visible": True # By default, all found runs are visible
        }
    
    # Write config to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False) # sort_keys=False to maintain order
    
    print(f"Generated new config file at {config_path}")
    print("Please edit this file to customize legend names, plot settings, max_steps, smoothing_weight, and run visibility.")
    
    return config

def load_config(config_path, log_dir):
    """Load config YAML file or generate a new one if it doesn't exist."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config file not found at {config_path}. Generating a new one.")
        config = generate_config_yaml(log_dir, config_path)
    
    config.setdefault("max_steps", 25000)
    config.setdefault("smoothing_weight", 0.8)
    config.setdefault("runs", {})
        
    return config

def smooth_data(scalars, weight=0.9):
    """Apply exponential moving average smoothing."""
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars)
    if scalars.size == 0:
        return np.array([])
    
    last = scalars[0]  # First point
    smoothed = np.zeros_like(scalars, dtype=float)
    smoothed[0] = last
    for i in range(1, len(scalars)):
        smoothed_val = last * weight + (1 - weight) * scalars[i]
        smoothed[i] = smoothed_val
        last = smoothed_val
    return smoothed

def load_tensorboard_data(log_dir, config, smoothing_weight):
    """Load data from TensorBoard logs based on the provided config."""
    all_runs_data = []
    
    max_steps_global = config.get("max_steps") # Get global max_steps from config

    if "runs" not in config or not config["runs"]:
        print("No runs specified or found in the config file's 'runs' section.")
        return all_runs_data

    for run_path_key, run_specific_config in config["runs"].items():
        if not run_specific_config.get("visible", True):
            print(f"Skipping run '{run_path_key}' as it's marked not visible in config.")
            continue

        # run_path_key is relative to log_dir
        current_run_actual_dir = os.path.join(log_dir, run_path_key)

        # Check if the directory for the run exists
        if not os.path.isdir(current_run_actual_dir):
            print(f"Directory for run '{run_path_key}' not found at '{current_run_actual_dir}'. Skipping.")
            continue

        # EventAccumulator takes the directory containing the event files
        try:
            ea = event_accumulator.EventAccumulator(
                current_run_actual_dir,
                size_guidance=event_accumulator.DEFAULT_SIZE_GUIDANCE # Default guidance
            )
            ea.Reload() # Load events
            
            available_tags = ea.Tags()['scalars']
            
            for tag in available_tags:
                events = ea.Scalars(tag)
                
                # Filter steps and values based on max_steps_global
                processed_steps = []
                processed_values = []
                
                for event in events:
                    if max_steps_global is None or event.step <= max_steps_global:
                        processed_steps.append(event.step)
                        processed_values.append(event.value)
                
                if not processed_values: # Skip if no data points remain after filtering (or if tag had no data)
                    print(f"No data for tag '{tag}' in run '{run_path_key}' (or all data beyond max_steps).")
                    continue

                legend_name = run_specific_config.get("legend", run_path_key) # Use run_path_key as fallback legend
                color = run_specific_config.get("color", None)
                linestyle = run_specific_config.get("linestyle", "-")

                data_entry = {
                    'run_path': run_path_key,
                    'legend_name': legend_name,
                    'tag': tag,
                    'step': processed_steps,
                    'value': processed_values,
                    'smoothed_value': smooth_data(np.array(processed_values), weight=smoothing_weight),
                    'color': color,
                    'linestyle': linestyle
                }
                all_runs_data.append(data_entry)
                
        except Exception as e:
            print(f"Error processing run '{run_path_key}' from directory '{current_run_actual_dir}': {e}")
            # Optionally, list event files to help debug if ea.Reload() fails due to no files:
            event_files_in_dir = glob.glob(os.path.join(current_run_actual_dir, "events.out.tfevents.*"))
            if not event_files_in_dir:
                print(f"  Note: No 'events.out.tfevents.*' files found in '{current_run_actual_dir}'.")
            
    return all_runs_data

def plot_tensorboard_data(all_runs, config=None, save_path=None, smoothing_weight_for_title=0.9):
    """Create seaborn plots from TensorBoard data."""
    if not all_runs:
        return

    # Group data by tag
    tags = sorted(list(set([run['tag'] for run in all_runs]))) # Sort tags for consistent plot order
    
    sns.set_theme(style="darkgrid")
    
    for tag in tags:
        plt.figure(figsize=(12, 7)) # Increased height slightly for legend
        
        tag_runs = [run for run in all_runs if run['tag'] == tag]
        
        for run_data in tag_runs:
            # Ensure there are steps and values to plot
            if not run_data['step'] or not run_data['smoothed_value'].size:
                print(f"Skipping plot for run '{run_data['legend_name']}' on tag '{tag}' due to missing data.")
                continue

            plot_kwargs = {
                'x': run_data['step'],
                'y': run_data['smoothed_value'],
                'label': run_data['legend_name'],
                'alpha': 0.8,
                'linestyle': run_data['linestyle']
            }
            
            if run_data['color']:
                plot_kwargs['color'] = run_data['color']
            
            sns.lineplot(**plot_kwargs)
        
        plot_title_text = f"{tag} (Smoothing={smoothing_weight_for_title})"
        if config and "title" in config:
            plot_title_text = f"{config['title']} - {tag} (Smoothing={smoothing_weight_for_title})"
            
        plt.title(plot_title_text)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Adjusted legend position
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside

        if save_path:
            safe_tag_filename = tag.replace('/', '_').replace(' ', '_')
            output_filename = os.path.join(save_path, f"{safe_tag_filename}.png")
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_filename}")
        else:
            plt.show()
        plt.close() # Close the figure to free memory

def main():
    log_dir = "/workspace/ehr_stuff/temp_/EHR_FM/outputs" 
    save_dir = "plots/" 
    config_path = "plot.yaml"  

    os.makedirs(save_dir, exist_ok=True)
    
    config = load_config(config_path, log_dir)
    smoothing_weight = config.get("smoothing_weight", 0.8)
    
    print("Loading TensorBoard data based on config...")
    all_runs_processed_data = load_tensorboard_data(log_dir, config, smoothing_weight)
    
    if not all_runs_processed_data:
        print("No data loaded. Possible reasons: no runs in config, runs marked not visible, "
              "issues with log directories, or all data filtered by max_steps.")
        return
    
    num_metrics = len(set([run['tag'] for run in all_runs_processed_data]))
    print(f"Creating plots for {num_metrics} metric(s)...")
    plot_tensorboard_data(all_runs_processed_data, config, save_path=save_dir, smoothing_weight_for_title=smoothing_weight)
    
    if num_metrics > 0 :
        print(f"Plots saved to {save_dir}")
    else:
        print("No metrics found to plot.")

if __name__ == "__main__":
    main()