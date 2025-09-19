"""LLM-as-judge scoring functions.

Uses Claude as a judge to evaluate model outputs. The judge receives
a rubric-formatted prompt and returns a structured score with reasoning.

Known biases to be aware of:
- Self-preference bias: Claude may rate its own outputs higher
- Verbosity bias: Longer responses tend to receive higher scores
- Position bias: In pairwise comparisons, the first response may be favored

We mitigate these by using structured rubrics, running multiple trials,
and separating the scoring logic from the rubric definitions.
"""

import json
import logging
import os
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client using the API key from environment.

    Returns:
        Configured Anthropic client.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY=your-key-here"
        )
    return anthropic.Anthropic(api_key=api_key)


def call_judge(
    judge_prompt: str,
    judge_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Call the LLM judge with a scoring prompt and parse the response.

    Args:
        judge_prompt: The fully formatted prompt for the judge.
        judge_config: Configuration dict with 'model', 'max_tokens', 'temperature'.

    Returns:
        Parsed dictionary with 'score' and 'reasoning', or None if parsing fails.
    """
    client = get_client()

    try:
        response = client.messages.create(
            model=judge_config["model"],
            max_tokens=judge_config.get("max_tokens", 512),
            temperature=judge_config.get("temperature", 0.0),
            messages=[{"role": "user", "content": judge_prompt}],
        )

        response_text = response.content[0].text
        return parse_judge_response(response_text)

    except anthropic.APIError as e:
        logger.error(f"API error during judge call: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during judge call: {e}")
        return None


def call_model(
    prompt: str,
    model_config: dict[str, Any],
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str | None:
    """Call the model being evaluated and return its response text.

    Args:
        prompt: The prompt to send.
        model_config: Configuration dict with 'name' and 'max_tokens'.
        system_prompt: Optional system prompt to use.
        temperature: Optional temperature override.

    Returns:
        The model's response text, or None on error.
    """
    client = get_client()

    kwargs = {
        "model": model_config["name"],
        "max_tokens": model_config.get("max_tokens", 1024),
        "messages": [{"role": "user", "content": prompt}],
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = client.messages.create(**kwargs)
        return response.content[0].text
    except anthropic.APIError as e:
        logger.error(f"API error during model call: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during model call: {e}")
        return None


def parse_judge_response(response_text: str) -> dict[str, Any] | None:
    """Parse the judge's JSON response into a score dictionary.

    Handles cases where the judge wraps JSON in markdown code blocks
    or includes extra text around the JSON object.

    Args:
        response_text: Raw text from the judge.

    Returns:
        Dictionary with 'score' (int) and 'reasoning' (str), or None.
    """
    # Try to extract JSON from code blocks first
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        text_to_parse = json_match.group(1).strip()
    else:
        text_to_parse = response_text.strip()

    # Try to find a JSON object in the text
    brace_match = re.search(r"\{[^{}]*\}", text_to_parse)
    if brace_match:
        text_to_parse = brace_match.group(0)

    try:
        result = json.loads(text_to_parse)

        if "score" not in result:
            logger.warning(f"Judge response missing 'score' key: {response_text[:200]}")
            return None

        score = result["score"]
        if not isinstance(score, (int, float)) or score < 1 or score > 5:
            logger.warning(f"Judge score out of range: {score}")
            return None

        return {
            "score": int(score),
            "reasoning": result.get("reasoning", "No reasoning provided"),
        }
    except json.JSONDecodeError:
        logger.warning(
            f"Could not parse judge response as JSON: {response_text[:200]}"
        )
        return None
