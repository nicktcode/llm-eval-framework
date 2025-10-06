"""Generate markdown evaluation reports from results.

Produces a formatted report summarizing evaluation outcomes
across all dimensions and experimental conditions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


DIMENSION_LABELS = {
    "factual": "Factual Accuracy",
    "adherence": "Instruction Adherence",
    "hallucination": "Hallucination Detection",
    "consistency": "Consistency",
}


def generate_report(
    results: dict[str, Any],
    output_path: str = "results/report.md",
    title: str = "LLM Evaluation Report",
) -> str:
    """Generate a markdown report from evaluation results.

    Args:
        results: Full evaluation results dictionary.
        output_path: Where to write the report.
        title: Report title.

    Returns:
        The markdown report as a string.
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    summary = results.get("summary", {})

    # Overall summary
    lines.append("## Overall Summary")
    lines.append("")
    if summary.get("overall_mean") is not None:
        lines.append(f"- Overall mean score: {summary['overall_mean']:.2f} / 5.00")
    lines.append(f"- Total evaluations: {summary.get('total_evaluations', 'N/A')}")
    if summary.get("elapsed_seconds"):
        lines.append(f"- Evaluation time: {summary['elapsed_seconds']:.0f}s")
    if summary.get("temperature") is not None:
        lines.append(f"- Temperature: {summary['temperature']}")
    lines.append("")

    # Per-dimension results
    lines.append("## Results by Dimension")
    lines.append("")
    lines.append("| Dimension | Mean | Std Dev | Min | Max | N |")
    lines.append("|-----------|------|---------|-----|-----|---|")

    for dim_key, dim_data in summary.get("dimensions", {}).items():
        label = DIMENSION_LABELS.get(dim_key, dim_key)
        mean = f"{dim_data['mean']:.2f}" if dim_data.get("mean") is not None else "N/A"
        std = f"{dim_data['std']:.2f}" if dim_data.get("std") is not None else "N/A"
        min_s = dim_data.get("min", "N/A")
        max_s = dim_data.get("max", "N/A")
        n = dim_data.get("scored_prompts", 0)
        lines.append(f"| {label} | {mean} | {std} | {min_s} | {max_s} | {n} |")

    lines.append("")

    # Per-prompt details
    raw_results = results.get("results", {})
    if raw_results:
        lines.append("## Detailed Results")
        lines.append("")

        for dim_key, dim_scores in raw_results.items():
            label = DIMENSION_LABELS.get(dim_key, dim_key)
            lines.append(f"### {label}")
            lines.append("")
            lines.append("| Prompt ID | Score | Reasoning |")
            lines.append("|-----------|-------|-----------|")

            for entry in dim_scores:
                pid = entry.get("prompt_id", "?")
                score = entry.get("score", "N/A")
                reasoning = entry.get("reasoning", "")
                # Truncate long reasoning for the table
                if len(reasoning) > 80:
                    reasoning = reasoning[:77] + "..."
                # Escape pipe chars in reasoning
                reasoning = reasoning.replace("|", "\\|")
                lines.append(f"| {pid} | {score} | {reasoning} |")

            lines.append("")

    report = "\n".join(lines)

    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(report)

    return report


def generate_comparison_report(
    comparison_data: dict[str, Any],
    output_path: str = "results/comparison_report.md",
    title: str = "Experiment Comparison Report",
) -> str:
    """Generate a report comparing multiple experimental conditions.

    Args:
        comparison_data: Output from runner temperature or system prompt comparison.
        output_path: Where to write the report.
        title: Report title.

    Returns:
        The markdown report as a string.
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    comparison = comparison_data.get("comparison", [])

    if comparison:
        # Summary table
        dim_keys = list(comparison[0].get("dimensions", {}).keys())
        dim_labels = [DIMENSION_LABELS.get(k, k) for k in dim_keys]

        header = "| Condition | Overall |"
        separator = "|-----------|---------|"
        for label in dim_labels:
            header += f" {label} |"
            separator += "------|"

        lines.append("## Comparison Table")
        lines.append("")
        lines.append(header)
        lines.append(separator)

        for entry in comparison:
            row = f"| {entry['condition']} |"
            overall = entry.get("overall_mean")
            row += f" {overall:.2f} |" if overall is not None else " N/A |"

            for dim_key in dim_keys:
                val = entry["dimensions"].get(dim_key)
                row += f" {val:.2f} |" if val is not None else " N/A |"

            lines.append(row)

        lines.append("")

        # Winner analysis
        if comparison:
            best = comparison[0]
            lines.append("## Analysis")
            lines.append("")
            lines.append(
                f"Best overall condition: **{best['condition']}** "
                f"(mean: {best.get('overall_mean', 'N/A')})"
            )
            lines.append("")

    report = "\n".join(lines)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(report)

    return report
