"""Factual accuracy scoring dimension.

Evaluates whether a model's output contains correct, verifiable facts
by using an LLM judge to compare the response against expected facts.
"""

import json
import logging
from typing import Any

from eval.judges import call_judge

logger = logging.getLogger(__name__)


def score_factual_accuracy(
    prompt: str,
    response: str,
    expected_facts: list[str],
    rubric: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    """Score a response for factual accuracy using the LLM judge.

    Args:
        prompt: The original prompt sent to the model.
        response: The model's response text.
        expected_facts: List of facts that should appear in the response.
        rubric: The factual accuracy rubric from rubrics.yaml.
        judge_config: Configuration for the judge model.

    Returns:
        Dictionary with 'score' (int 1-5) and 'reasoning' (str).
    """
    judge_prompt = rubric["judge_prompt"].format(
        prompt=prompt,
        response=response,
        expected_facts=", ".join(expected_facts),
    )

    result = call_judge(judge_prompt, judge_config)

    if result is None:
        logger.warning("Judge returned None for factual accuracy scoring")
        return {"score": None, "reasoning": "Judge failed to return a valid score"}

    return result


def check_facts_present(response: str, expected_facts: list[str]) -> dict[str, bool]:
    """Quick heuristic check for whether expected facts appear in the response.

    This is a simple string-matching check that supplements the LLM judge.
    It's not authoritative but useful for fast sanity checking.

    Args:
        response: The model's response text.
        expected_facts: List of expected fact strings.

    Returns:
        Dictionary mapping each expected fact to whether it was found.
    """
    response_lower = response.lower()
    results = {}
    for fact in expected_facts:
        results[fact] = fact.lower() in response_lower
    return results


def aggregate_factual_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple factual accuracy scores into a summary.

    Args:
        scores: List of score dictionaries from score_factual_accuracy.

    Returns:
        Dictionary with mean, min, max, and count of valid scores.
    """
    valid_scores = [s["score"] for s in scores if s.get("score") is not None]

    if not valid_scores:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
            "total_prompts": len(scores),
        }

    return {
        "mean": sum(valid_scores) / len(valid_scores),
        "min": min(valid_scores),
        "max": max(valid_scores),
        "count": len(valid_scores),
        "total_prompts": len(scores),
    }
