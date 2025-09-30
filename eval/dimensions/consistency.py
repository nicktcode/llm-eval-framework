"""Consistency scoring dimension.

Evaluates how consistent a model's outputs are when given the same
prompt multiple times. Uses pairwise semantic similarity judgments
rather than exact string matching.
"""

import itertools
import json
import logging
from typing import Any

from eval.judges import call_judge

logger = logging.getLogger(__name__)


def score_consistency(
    prompt: str,
    responses: list[str],
    rubric: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    """Score the consistency of multiple responses to the same prompt.

    Computes pairwise consistency scores between all response pairs
    using the LLM judge, then aggregates them into an overall score.

    Args:
        prompt: The original prompt.
        responses: List of responses from multiple trials.
        rubric: The consistency rubric from rubrics.yaml.
        judge_config: Configuration for the judge model.

    Returns:
        Dictionary with overall score, pairwise scores, and variance.
    """
    if len(responses) < 2:
        return {
            "score": None,
            "reasoning": "Need at least 2 responses to measure consistency",
            "pairwise_scores": [],
            "variance": None,
        }

    pairwise_scores = []
    pairs = list(itertools.combinations(range(len(responses)), 2))

    for i, j in pairs:
        judge_prompt = rubric["judge_prompt"].format(
            prompt=prompt,
            response_a=responses[i],
            response_b=responses[j],
        )

        result = call_judge(judge_prompt, judge_config)

        if result is not None:
            pairwise_scores.append(
                {
                    "pair": (i, j),
                    "score": result["score"],
                    "reasoning": result["reasoning"],
                }
            )

    valid_scores = [p["score"] for p in pairwise_scores if p["score"] is not None]

    if not valid_scores:
        return {
            "score": None,
            "reasoning": "No valid pairwise scores obtained",
            "pairwise_scores": pairwise_scores,
            "variance": None,
        }

    mean_score = sum(valid_scores) / len(valid_scores)

    variance = (
        sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores)
        if len(valid_scores) > 1
        else 0.0
    )

    return {
        "score": round(mean_score, 2),
        "reasoning": f"Mean pairwise consistency across {len(valid_scores)} pairs",
        "pairwise_scores": pairwise_scores,
        "variance": round(variance, 4),
    }


def compute_lexical_similarity(response_a: str, response_b: str) -> float:
    """Compute simple lexical similarity between two responses.

    Uses Jaccard similarity on word sets as a quick baseline metric.
    This supplements the LLM-based semantic similarity judgment.

    Args:
        response_a: First response text.
        response_b: Second response text.

    Returns:
        Jaccard similarity coefficient between 0 and 1.
    """
    words_a = set(response_a.lower().split())
    words_b = set(response_b.lower().split())

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


def aggregate_consistency_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate consistency scores across multiple prompts.

    Args:
        scores: List of consistency score dictionaries.

    Returns:
        Summary statistics.
    """
    valid_scores = [s["score"] for s in scores if s.get("score") is not None]
    variances = [s["variance"] for s in scores if s.get("variance") is not None]

    if not valid_scores:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
            "total_prompts": len(scores),
            "mean_variance": None,
        }

    return {
        "mean": sum(valid_scores) / len(valid_scores),
        "min": min(valid_scores),
        "max": max(valid_scores),
        "count": len(valid_scores),
        "total_prompts": len(scores),
        "mean_variance": (
            sum(variances) / len(variances) if variances else None
        ),
    }
