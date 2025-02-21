import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from pathlib import Path
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import os
import pandas as pd
from glob import glob
from pathlib import Path

def extract_run_data(event_file, target_metrics=None):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    run_data = {}
    available_tags = ea.Tags()['scalars']
    metrics_to_extract = target_metrics if target_metrics else available_tags
    
    for tag in metrics_to_extract:
        if tag in available_tags:
            events = ea.Scalars(tag)
            df = pd.DataFrame({
                'step': [e.step for e in events],
                'value': [e.value for e in events],
                'wall_time': [e.wall_time for e in events]
            })
            run_data[tag] = df
    
    return run_data

def process_experiment_group(group_path, output_dir, target_metrics=None):
    group_name = os.path.basename(group_path)
    print(f"Processing {group_name}...")
    
    group_output_dir = os.path.join(output_dir, f"{group_name}_csv")
    os.makedirs(group_output_dir, exist_ok=True)
    
    version_dirs = sorted(glob(os.path.join(group_path, "version_*")))
    for version_dir in version_dirs:
        version = os.path.basename(version_dir)
        
        event_files = []
        for root, _, files in os.walk(version_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, file))
        
        for event_file in event_files:
            try:
                run_data = extract_run_data(event_file, target_metrics)
                
                for metric, df in run_data.items():
                    output_file = os.path.join(group_output_dir, f"{version}_{metric}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Saved {output_file}")
                    
            except Exception as e:
                print(f"Error processing {event_file}: {str(e)}")

def export_all_groups(logdir, output_dir, target_metrics=None):
    experiment_groups = [
        d for d in glob(os.path.join(logdir, "cifar10_*"))
        if os.path.isdir(d) and not d.endswith("_csv")
    ]
    
    print(f"Found {len(experiment_groups)} experiment groups")
    if target_metrics:
        print(f"Extracting metrics: {', '.join(target_metrics)}")
    
    for group in experiment_groups:
        process_experiment_group(group, output_dir, target_metrics)

def generate_distinct_colors(n, experiment_names):
    colors = []
    for i, name in enumerate(experiment_names):
        if 'steg' in name.lower():
            colors.append((1.0, 0.0, 0.0))
        else:
            hue = i / n
            saturation = 0.7 + np.random.normal(0, 0.1)
            saturation = np.clip(saturation, 0.6, 0.8)
            value = 0.9 + np.random.normal(0, 0.1)
            value = np.clip(value, 0.8, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
    return colors

def plot_experiment_metrics(experiment_names, base_path, output_dir, show_std=True):
    all_experiments = {}
    total_versions = 0

    for experiment_name in experiment_names:
        experiment_path = Path(base_path) / experiment_name
        version_dfs = {}

        for csv_file in experiment_path.glob(f"*version_*.csv"):
            try:
                parts = csv_file.stem.split("_")
                version_idx = parts.index("version") + 1
                version_num = int(parts[version_idx])
                
                df = pd.read_csv(csv_file)
                df.columns = ['step', 'value', 'wall_time']
                version_dfs[version_num] = df
            except Exception as e:
                print(f"Error loading file {csv_file} for experiment {experiment_name}: {e}")

        if version_dfs:
            all_experiments[experiment_name] = version_dfs
            total_versions += len(version_dfs)

    if not all_experiments:
        print("No experiment data found.")
        return

    colors = generate_distinct_colors(len(experiment_names), experiment_names)

    plt.figure(figsize=(12, 8))
    
    for color, (experiment_name, version_dfs) in zip(colors, all_experiments.items()):
        all_steps = []
        all_values = []

        for version_num, df in version_dfs.items():
            grouped = df.groupby('step')['value'].mean()
            all_steps.append(grouped.index)
            all_values.append(grouped.values)

        max_steps = max(map(len, all_steps))
        combined_values = np.full((len(all_values), max_steps), np.nan)

        for i, values in enumerate(all_values):
            combined_values[i, :len(values)] = values

        mean_values = np.nanmean(combined_values, axis=0)
        std_values = np.nanstd(combined_values, axis=0)
        steps = np.arange(len(mean_values))

        linewidth = 2.5 if 'steg' in experiment_name.lower() else 2.0
        zorder = 10 if 'steg' in experiment_name.lower() else 1

        plt.plot(steps, mean_values, 
                label=experiment_name.replace('_csv', ''), 
                color=color,
                linewidth=linewidth,
                zorder=zorder)
        
        if show_std:
            plt.fill_between(
                steps, 
                mean_values - std_values, 
                mean_values + std_values, 
                alpha=0.2, 
                color=color,
                zorder=zorder-1
            )

    plt.title("Validation Accuracy Across Experiments", weight="bold", fontsize=16)
    plt.xlabel("Step", weight="bold", fontsize=14)
    plt.ylabel("Accuracy", weight="bold", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "combined_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

def main():
    TENSORBOARD_LOG_DIR = "tensorboard_logs"
    OUTPUT_DIR = "output_csvs"
    METRICS_TO_EXTRACT = ["val_acc"]  
    base_path = "output_csvs"
    output_dir = "plots"
    
    export_all_groups(TENSORBOARD_LOG_DIR, OUTPUT_DIR, METRICS_TO_EXTRACT)

    experiments = [
        "cifar10_steg_hflips",
        "cifar10_steg",
        "cifar10_color", 
        "cifar10_erase",
        "cifar10_none",
        "cifar10_hflip",
        "cifar10_blur",
    ]
    
    for i in range(len(experiments)):
        experiments[i] = experiments[i] + '_csv'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_experiment_metrics(experiments, base_path, output_dir, show_std=False)

if __name__ == "__main__":
    main()