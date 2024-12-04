import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from pathlib import Path


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.normal(0, 0.1)
        saturation = np.clip(saturation, 0.6, 0.8)
        value = 0.9 + np.random.normal(0, 0.1)
        value = np.clip(value, 0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def adjust_colors_for_two(colors):
    if len(colors) == 2:
        colors[0] = (1.0, 0.0, 0.0)
        colors[1] = (0.0, 0.0, 0.5)
    return colors


def plot_experiment_metrics(experiment_names, base_path, output_dir, show_std=True):
    all_experiments = {}
    total_versions = 0

    for experiment_name in experiment_names:
        experiment_path = Path(base_path) / experiment_name
        version_dfs = {}

        for csv_file in experiment_path.glob(f"{experiment_name}_version_*.csv"):
            try:
                version_num = int(csv_file.stem.split("_")[-1])
                df = pd.read_csv(csv_file)
                version_dfs[version_num] = df
            except Exception as e:
                print(f"Error loading file {csv_file} for experiment {experiment_name}: {e}")

        if version_dfs:
            all_experiments[experiment_name] = version_dfs
            total_versions += len(version_dfs)

    if not all_experiments:
        print("No experiment data found.")
        return

    colors = generate_distinct_colors(len(experiment_names))
    colors = adjust_colors_for_two(colors)
    handles, labels = [], []

    individual_plot_path = Path(output_dir) / "individual_metrics.png"
    plt.figure(figsize=(12, 8))
    for color, (experiment_name, version_dfs) in zip(colors, all_experiments.items()):
        all_steps = []
        all_values = []

        for version_num, df in version_dfs.items():
            grouped = df.groupby("Step")["Value"].mean()
            all_steps.append(grouped.index)
            all_values.append(grouped.values)

        max_steps = max(map(len, all_steps))
        combined_values = np.full((len(all_values), max_steps), np.nan)

        for i, values in enumerate(all_values):
            combined_values[i, :len(values)] = values

        mean_values = np.nanmean(combined_values, axis=0)
        std_values = np.nanstd(combined_values, axis=0)
        steps = np.arange(len(mean_values))

        plt.plot(steps, mean_values, label=experiment_name, color=color)
        if show_std:
            plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.2, color=color)

        handles.append(plt.Line2D([], [], color=color))
        labels.append(experiment_name)

    plt.title("Individual Experiment Metrics", weight="bold", fontsize=16)
    plt.xlabel("Step", weight="bold", fontsize=14)
    plt.ylabel("Value", weight="bold", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(handles, labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(individual_plot_path, dpi=300)
    plt.close()

    mean_plot_path = Path(output_dir) / "mean_metrics.png"
    plt.figure(figsize=(12, 8))
    for color, (experiment_name, version_dfs) in zip(colors, all_experiments.items()):
        all_steps = []
        all_values = []

        for version_num, df in version_dfs.items():
            grouped = df.groupby("Step")["Value"].mean()
            all_steps.append(grouped.index)
            all_values.append(grouped.values)

        max_steps = max(map(len, all_steps))
        combined_values = np.full((len(all_values), max_steps), np.nan)

        for i, values in enumerate(all_values):
            combined_values[i, :len(values)] = values

        mean_values = np.nanmean(combined_values, axis=0)
        std_values = np.nanstd(combined_values, axis=0)
        steps = np.arange(len(mean_values))

        plt.plot(steps, mean_values, label=experiment_name, color=color)
        if show_std:
            plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.2, color=color)

    plt.title("Mean Experiment Metrics", weight="bold", fontsize=16)
    plt.xlabel("Step", weight="bold", fontsize=14)
    plt.ylabel("Mean Value", weight="bold", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mean_plot_path, dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    experiments = ["cifar10_d_resnet18", "cifar10_s_resnet18"]
    base_path = "experiment_logs"
    output_dir = "plots"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_experiment_metrics(experiments, base_path, output_dir, show_std=False)


if __name__ == "__main__":
    main()
