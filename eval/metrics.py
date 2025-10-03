"""Aggregation and statistical metrics for evaluation results.

Provides functions to compute summary statistics across dimensions,
prompts, and experimental conditions.
"""

import math
from typing import Any


def compute_mean(values: list[float | int]) -> float | None:
    """Compute arithmetic mean, returning None for empty lists."""
    if not values:
        return None
    return sum(values) / len(values)


def compute_std(values: list[float | int]) -> float | None:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_confidence_interval(
    values: list[float | int], confidence: float = 0.95
) -> tuple[float, float] | None:
    """Compute confidence interval using t-distribution approximation.

    For small sample sizes we use a simplified approach with
    approximate t-values rather than pulling in scipy.

    Args:
        values: List of numeric values.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds, or None if insufficient data.
    """
    n = len(values)
    if n < 2:
        return None

    mean = sum(values) / n
    std = compute_std(values)
    if std is None:
        return None

    # Approximate t-values for common confidence levels and small n
    # These are good enough for reporting purposes
    t_approx = {
        0.90: {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 10: 1.833, 30: 1.699},
        0.95: {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262, 30: 2.045},
        0.99: {2: 63.657, 3: 9.925, 4: 5.841, 5: 4.604, 10: 3.250, 30: 2.756},
    }

    t_values = t_approx.get(confidence, t_approx[0.95])

    # Find the closest n in our table
    available_n = sorted(t_values.keys())
    closest_n = min(available_n, key=lambda x: abs(x - n))
    t_val = t_values[closest_n]

    margin = t_val * (std / math.sqrt(n))

    return (round(mean - margin, 4), round(mean + margin, 4))


def summarize_dimension(
    scores: list[dict[str, Any]], dimension_name: str
) -> dict[str, Any]:
    """Create a summary for a single evaluation dimension.

    Args:
        scores: List of score dictionaries for this dimension.
        dimension_name: Name of the dimension being summarized.

    Returns:
        Summary dictionary with statistics and metadata.
    """
    valid_scores = [s["score"] for s in scores if s.get("score") is not None]

    summary = {
        "dimension": dimension_name,
        "total_prompts": len(scores),
        "scored_prompts": len(valid_scores),
        "failed_prompts": len(scores) - len(valid_scores),
    }

    if valid_scores:
        summary["mean"] = round(compute_mean(valid_scores), 3)
        summary["std"] = round(compute_std(valid_scores), 3) if compute_std(valid_scores) else 0.0
        summary["min"] = min(valid_scores)
        summary["max"] = max(valid_scores)
        summary["median"] = sorted(valid_scores)[len(valid_scores) // 2]

        ci = compute_confidence_interval(valid_scores)
        if ci:
            summary["ci_95_lower"] = ci[0]
            summary["ci_95_upper"] = ci[1]
    else:
        summary["mean"] = None
        summary["std"] = None
        summary["min"] = None
        summary["max"] = None
        summary["median"] = None

    return summary


def summarize_experiment(
    results: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Summarize results across all dimensions for an experiment.

    Args:
        results: Dictionary mapping dimension names to their score lists.

    Returns:
        Overall experiment summary with per-dimension breakdowns.
    """
    dimension_summaries = {}
    all_means = []

    for dim_name, scores in results.items():
        dim_summary = summarize_dimension(scores, dim_name)
        dimension_summaries[dim_name] = dim_summary
        if dim_summary["mean"] is not None:
            all_means.append(dim_summary["mean"])

    return {
        "dimensions": dimension_summaries,
        "overall_mean": round(compute_mean(all_means), 3) if all_means else None,
        "total_evaluations": sum(
            s["total_prompts"] for s in dimension_summaries.values()
        ),
    }


def compare_conditions(
    condition_results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare results across experimental conditions (e.g., temperatures).

    Args:
        condition_results: Dictionary mapping condition labels to their
            experiment summaries.

    Returns:
        List of comparison entries sorted by overall performance.
    """
    comparisons = []

    for condition, summary in condition_results.items():
        entry = {
            "condition": condition,
            "overall_mean": summary.get("overall_mean"),
            "dimensions": {},
        }

        for dim_name, dim_summary in summary.get("dimensions", {}).items():
            entry["dimensions"][dim_name] = dim_summary.get("mean")

        comparisons.append(entry)

    comparisons.sort(
        key=lambda x: x["overall_mean"] if x["overall_mean"] is not None else 0,
        reverse=True,
    )

    return comparisons


def compute_judge_agreement(
    trial_scores: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    """Compute agreement between multiple judge trials on the same data.

    Measures how consistent the judge is when scoring the same
    prompt-response pair multiple times.

    Args:
        trial_scores: List of trials, where each trial is a list
            of score dicts aligned by prompt index.

    Returns:
        Agreement statistics including exact match rate and mean deviation.
    """
    if len(trial_scores) < 2:
        return {"agreement_rate": None, "mean_deviation": None}

    num_prompts = min(len(trial) for trial in trial_scores)
    exact_matches = 0
    deviations = []
    valid_comparisons = 0

    for prompt_idx in range(num_prompts):
        scores_for_prompt = []
        for trial in trial_scores:
            if prompt_idx < len(trial) and trial[prompt_idx].get("score") is not None:
                scores_for_prompt.append(trial[prompt_idx]["score"])

        if len(scores_for_prompt) >= 2:
            valid_comparisons += 1

            if len(set(scores_for_prompt)) == 1:
                exact_matches += 1

            mean_s = sum(scores_for_prompt) / len(scores_for_prompt)
            max_dev = max(abs(s - mean_s) for s in scores_for_prompt)
            deviations.append(max_dev)

    if valid_comparisons == 0:
        return {"agreement_rate": None, "mean_deviation": None}

    return {
        "agreement_rate": round(exact_matches / valid_comparisons, 3),
        "mean_deviation": round(sum(deviations) / len(deviations), 3),
        "exact_matches": exact_matches,
        "total_comparisons": valid_comparisons,
    }
