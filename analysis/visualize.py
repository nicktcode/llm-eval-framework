"""Generate charts and visual reports from evaluation results.

Produces bar charts, heatmaps, and comparison plots for the
evaluation dimensions and experimental conditions.
"""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-friendly palette (IBM Design Library)
COLORS = {
    "blue": "#648FFF",
    "purple": "#785EF0",
    "magenta": "#DC267F",
    "orange": "#FE6100",
    "yellow": "#FFB000",
}

COLOR_LIST = list(COLORS.values())

DIMENSION_LABELS = {
    "factual": "Factual Accuracy",
    "adherence": "Instruction Adherence",
    "hallucination": "Hallucination (inv.)",
    "consistency": "Consistency",
}


def plot_dimension_scores(
    summary: dict[str, Any],
    output_path: str = "results/charts/dimension_scores.png",
    title: str = "Evaluation Scores by Dimension",
):
    """Create a bar chart of mean scores across dimensions.

    Args:
        summary: Experiment summary from metrics.summarize_experiment.
        output_path: Where to save the chart.
        title: Chart title.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dimensions = summary.get("dimensions", {})
    if not dimensions:
        return

    labels = []
    means = []
    errors = []

    for dim_key, dim_data in dimensions.items():
        label = DIMENSION_LABELS.get(dim_key, dim_key)
        labels.append(label)
        means.append(dim_data.get("mean", 0) or 0)
        std = dim_data.get("std", 0) or 0
        errors.append(std)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=errors, capsize=5,
        color=COLOR_LIST[:len(labels)],
        edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )

    ax.set_ylabel("Mean Score (1-5)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{mean:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temperature_comparison(
    comparison: list[dict[str, Any]],
    output_path: str = "results/charts/temperature_comparison.png",
):
    """Create a grouped bar chart comparing scores across temperatures.

    Args:
        comparison: List of comparison entries from metrics.compare_conditions.
        output_path: Where to save the chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not comparison:
        return

    conditions = [c["condition"] for c in comparison]
    dimension_keys = list(comparison[0].get("dimensions", {}).keys())
    dim_labels = [DIMENSION_LABELS.get(k, k) for k in dimension_keys]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(dim_labels))
    bar_width = 0.25
    offsets = np.linspace(
        -bar_width * (len(conditions) - 1) / 2,
        bar_width * (len(conditions) - 1) / 2,
        len(conditions),
    )

    for idx, (condition, offset) in enumerate(zip(comparison, offsets)):
        values = [
            condition["dimensions"].get(k, 0) or 0
            for k in dimension_keys
        ]
        bars = ax.bar(
            x + offset, values, bar_width,
            label=condition["condition"].replace("_", " ").title(),
            color=COLOR_LIST[idx % len(COLOR_LIST)],
            edgecolor="white", linewidth=0.5,
        )

    ax.set_ylabel("Mean Score (1-5)", fontsize=12)
    ax.set_title("Score Comparison Across Temperature Settings", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_difficulty_breakdown(
    scores_by_difficulty: dict[str, list[float]],
    output_path: str = "results/charts/difficulty_breakdown.png",
):
    """Plot score distribution by difficulty level.

    Args:
        scores_by_difficulty: Dict mapping difficulty to list of scores.
        output_path: Where to save the chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    means = []
    stds = []

    for diff in difficulties:
        vals = scores_by_difficulty.get(diff, [])
        if vals:
            means.append(sum(vals) / len(vals))
            if len(vals) > 1:
                m = sum(vals) / len(vals)
                stds.append((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)
            else:
                stds.append(0)
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [COLORS["blue"], COLORS["orange"], COLORS["magenta"]]
    bars = ax.bar(
        difficulties, means, yerr=stds, capsize=5,
        color=colors, edgecolor="white", linewidth=0.8,
    )

    ax.set_ylabel("Mean Score (1-5)", fontsize=12)
    ax.set_title("Scores by Prompt Difficulty", fontsize=14, pad=15)
    ax.set_ylim(0, 5.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_judge_agreement(
    agreement_data: dict[str, Any],
    output_path: str = "results/charts/judge_agreement.png",
):
    """Visualize judge agreement statistics.

    Args:
        agreement_data: Output from metrics.compute_judge_agreement.
        output_path: Where to save the chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    agreement_rate = agreement_data.get("agreement_rate", 0) or 0
    disagreement_rate = 1 - agreement_rate

    bars = ax.bar(
        ["Exact Agreement", "Disagreement"],
        [agreement_rate * 100, disagreement_rate * 100],
        color=[COLORS["blue"], COLORS["orange"]],
        edgecolor="white", linewidth=0.8,
    )

    for bar, val in zip(bars, [agreement_rate, disagreement_rate]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val * 100:.1f}%", ha="center", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Judge Self-Agreement Rate", fontsize=14, pad=15)
    ax.set_ylim(0, 110)

    mean_dev = agreement_data.get("mean_deviation", 0) or 0
    ax.text(
        0.95, 0.95,
        f"Mean deviation: {mean_dev:.2f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_charts(results_dir: str = "results"):
    """Generate all available charts from saved results files.

    Args:
        results_dir: Directory containing JSON result files.
    """
    results_path = Path(results_dir)
    charts_dir = results_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Full evaluation chart
    full_eval_path = results_path / "full_eval.json"
    if full_eval_path.exists():
        with open(full_eval_path) as f:
            data = json.load(f)
        if "summary" in data:
            plot_dimension_scores(
                data["summary"],
                str(charts_dir / "dimension_scores.png"),
            )

    # Temperature comparison
    temp_path = results_path / "temperature_comparison.json"
    if temp_path.exists():
        with open(temp_path) as f:
            data = json.load(f)
        if "comparison" in data:
            plot_temperature_comparison(
                data["comparison"],
                str(charts_dir / "temperature_comparison.png"),
            )

    # Judge agreement
    agreement_path = results_path / "judge_agreement.json"
    if agreement_path.exists():
        with open(agreement_path) as f:
            data = json.load(f)
        plot_judge_agreement(data, str(charts_dir / "judge_agreement.png"))


if __name__ == "__main__":
    generate_all_charts()
