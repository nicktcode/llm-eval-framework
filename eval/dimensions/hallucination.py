"""Hallucination detection scoring dimension.

Evaluates whether a model's response introduces claims not supported
by the provided source context. This is especially important for
retrieval-augmented generation and grounded Q&A tasks.
"""

import json
import logging
from typing import Any

from eval.judges import call_judge

logger = logging.getLogger(__name__)


def score_hallucination(
    prompt: str,
    response: str,
    context: str,
    expected_answer: str,
    rubric: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    """Score a response for hallucination using the LLM judge.

    The scoring is inverted from typical "higher is worse" hallucination
    metrics: here a 5 means no hallucination (best) and a 1 means
    severe hallucination (worst), consistent with the other dimensions.

    Args:
        prompt: The original prompt (includes context in it).
        response: The model's response text.
        context: The source context the model should have grounded on.
        expected_answer: Description of what a correct answer looks like.
        rubric: The hallucination rubric from rubrics.yaml.
        judge_config: Configuration for the judge model.

    Returns:
        Dictionary with 'score', 'reasoning', and 'grounding_check'.
    """
    grounding = check_grounding_heuristic(response, context)

    judge_prompt = rubric["judge_prompt"].format(
        prompt=prompt,
        context=context,
        response=response,
        expected_answer=expected_answer,
    )

    result = call_judge(judge_prompt, judge_config)

    if result is None:
        logger.warning("Judge returned None for hallucination scoring")
        return {
            "score": None,
            "reasoning": "Judge failed to return a valid score",
            "grounding_check": grounding,
        }

    result["grounding_check"] = grounding
    return result


def check_grounding_heuristic(response: str, context: str) -> dict[str, Any]:
    """Basic heuristic check for response grounding.

    Compares word overlap between the response and the source context.
    This is a rough measure and not a substitute for the LLM judge,
    but it can flag cases where the response diverges significantly
    from the provided text.

    Args:
        response: The model's response.
        context: The source context.

    Returns:
        Dictionary with overlap statistics.
    """
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())

    # Remove common stop words to focus on content words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "where",
        "how", "what", "which", "who", "whom", "this", "that", "these",
        "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their",
    }

    context_content = context_words - stop_words
    response_content = response_words - stop_words

    if not response_content:
        return {
            "overlap_ratio": 0.0,
            "novel_words_count": 0,
            "context_coverage": 0.0,
        }

    overlap = response_content & context_content
    novel = response_content - context_content

    overlap_ratio = len(overlap) / len(response_content)
    context_coverage = len(overlap) / len(context_content) if context_content else 0.0

    return {
        "overlap_ratio": round(overlap_ratio, 3),
        "novel_words_count": len(novel),
        "context_coverage": round(context_coverage, 3),
    }


def detect_refusal(response: str) -> bool:
    """Detect whether the model refused to answer or flagged missing information.

    Some prompts are designed to trick the model into hallucinating.
    A good response might refuse to answer or explicitly state that
    the information isn't in the context. This detects that pattern.

    Args:
        response: The model's response text.

    Returns:
        True if the response appears to be a refusal/flag, False otherwise.
    """
    refusal_patterns = [
        "not mentioned",
        "not provided",
        "not included",
        "does not contain",
        "does not mention",
        "doesn't mention",
        "doesn't contain",
        "no information about",
        "not enough information",
        "cannot determine",
        "cannot answer",
        "not stated",
        "not available in",
        "not found in",
        "the text does not",
        "the provided text",
        "based on the text alone",
        "i cannot find",
        "there is no mention",
    ]

    response_lower = response.lower()
    return any(pattern in response_lower for pattern in refusal_patterns)


def aggregate_hallucination_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hallucination scores across multiple prompts.

    Args:
        scores: List of score dictionaries.

    Returns:
        Summary with mean, min, max, refusal rate, and grounding stats.
    """
    valid_scores = [s["score"] for s in scores if s.get("score") is not None]

    grounding_overlaps = []
    for s in scores:
        gc = s.get("grounding_check", {})
        if "overlap_ratio" in gc:
            grounding_overlaps.append(gc["overlap_ratio"])

    if not valid_scores:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
            "total_prompts": len(scores),
            "avg_grounding_overlap": None,
        }

    return {
        "mean": sum(valid_scores) / len(valid_scores),
        "min": min(valid_scores),
        "max": max(valid_scores),
        "count": len(valid_scores),
        "total_prompts": len(scores),
        "avg_grounding_overlap": (
            sum(grounding_overlaps) / len(grounding_overlaps)
            if grounding_overlaps
            else None
        ),
    }
