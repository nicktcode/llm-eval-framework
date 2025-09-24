"""Instruction adherence scoring dimension.

Evaluates whether a model followed specific formatting, length,
and constraint instructions in the prompt.
"""

import json
import logging
import re
from typing import Any

from eval.judges import call_judge

logger = logging.getLogger(__name__)


def score_instruction_adherence(
    prompt: str,
    response: str,
    format_requirements: dict[str, Any],
    rubric: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    """Score a response for instruction adherence using the LLM judge.

    Args:
        prompt: The original prompt sent to the model.
        response: The model's response text.
        format_requirements: Dictionary describing the format constraints.
        rubric: The instruction adherence rubric from rubrics.yaml.
        judge_config: Configuration for the judge model.

    Returns:
        Dictionary with 'score', 'reasoning', and 'heuristic_checks'.
    """
    heuristic_results = run_heuristic_checks(response, format_requirements)

    requirements_str = json.dumps(format_requirements, indent=2)
    judge_prompt = rubric["judge_prompt"].format(
        prompt=prompt,
        response=response,
        format_requirements=requirements_str,
    )

    result = call_judge(judge_prompt, judge_config)

    if result is None:
        logger.warning("Judge returned None for adherence scoring")
        return {
            "score": None,
            "reasoning": "Judge failed to return a valid score",
            "heuristic_checks": heuristic_results,
        }

    result["heuristic_checks"] = heuristic_results
    return result


def run_heuristic_checks(
    response: str, requirements: dict[str, Any]
) -> dict[str, bool]:
    """Run automated heuristic checks against format requirements.

    These checks are deterministic and supplement the LLM judge scoring.
    Not all requirements can be checked heuristically, so only the ones
    that can be reliably automated are included.

    Args:
        response: The model's response text.
        requirements: Format requirements dictionary.

    Returns:
        Dictionary mapping check names to pass/fail booleans.
    """
    checks = {}

    if "line_count" in requirements:
        lines = [l for l in response.strip().split("\n") if l.strip()]
        checks["line_count"] = len(lines) == requirements["line_count"]

    if "valid_json" in requirements and requirements["valid_json"]:
        try:
            # Try to extract JSON from code blocks first
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
            else:
                parsed = json.loads(response.strip())
            checks["valid_json"] = True

            if "required_keys" in requirements:
                checks["has_required_keys"] = all(
                    k in parsed for k in requirements["required_keys"]
                )

            if "hobbies_count" in requirements and "hobbies" in parsed:
                if isinstance(parsed["hobbies"], list):
                    checks["hobbies_count"] = (
                        len(parsed["hobbies"]) == requirements["hobbies_count"]
                    )
        except (json.JSONDecodeError, TypeError):
            checks["valid_json"] = False

    if "word_count_min" in requirements or "word_count_max" in requirements:
        word_count = len(response.split())
        if "word_count_min" in requirements:
            checks["word_count_min"] = word_count >= requirements["word_count_min"]
        if "word_count_max" in requirements:
            checks["word_count_max"] = word_count <= requirements["word_count_max"]

    if "forbidden_words" in requirements:
        response_lower = response.lower()
        checks["no_forbidden_words"] = not any(
            w.lower() in response_lower for w in requirements["forbidden_words"]
        )

    if "sentence_count" in requirements:
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        checks["sentence_count"] = len(sentences) == requirements["sentence_count"]

    if "max_chars" in requirements:
        checks["max_chars"] = len(response.strip()) <= requirements["max_chars"]

    if "bullet_count" in requirements:
        bullet_char = requirements.get("bullet_char", "-")
        bullets = [
            l.strip()
            for l in response.strip().split("\n")
            if l.strip().startswith(bullet_char)
        ]
        checks["bullet_count"] = len(bullets) == requirements["bullet_count"]

    if "numbered_list" in requirements and requirements["numbered_list"]:
        numbered_lines = re.findall(r"^\d+[\.\)]\s", response, re.MULTILINE)
        if "item_count" in requirements:
            checks["numbered_item_count"] = (
                len(numbered_lines) == requirements["item_count"]
            )

    if "paragraph_count" in requirements:
        paragraphs = [p.strip() for p in response.strip().split("\n\n") if p.strip()]
        checks["paragraph_count"] = len(paragraphs) == requirements["paragraph_count"]

    return checks


def aggregate_adherence_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple instruction adherence scores.

    Args:
        scores: List of score dictionaries.

    Returns:
        Summary statistics for adherence scores.
    """
    valid_scores = [s["score"] for s in scores if s.get("score") is not None]

    heuristic_pass_rates = {}
    for score_entry in scores:
        for check_name, passed in score_entry.get("heuristic_checks", {}).items():
            if check_name not in heuristic_pass_rates:
                heuristic_pass_rates[check_name] = {"passed": 0, "total": 0}
            heuristic_pass_rates[check_name]["total"] += 1
            if passed:
                heuristic_pass_rates[check_name]["passed"] += 1

    for check_name in heuristic_pass_rates:
        stats = heuristic_pass_rates[check_name]
        stats["rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0

    if not valid_scores:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
            "total_prompts": len(scores),
            "heuristic_pass_rates": heuristic_pass_rates,
        }

    return {
        "mean": sum(valid_scores) / len(valid_scores),
        "min": min(valid_scores),
        "max": max(valid_scores),
        "count": len(valid_scores),
        "total_prompts": len(scores),
        "heuristic_pass_rates": heuristic_pass_rates,
    }
