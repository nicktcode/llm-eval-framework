"""Tests for the judges module.

All tests mock the Anthropic API so they run without an API key.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from eval.judges import (
    parse_judge_response,
    call_judge,
    call_model,
    get_client,
)


class TestParseJudgeResponse:
    """Tests for parsing the judge's JSON responses."""

    def test_parses_clean_json(self):
        response = '{"score": 4, "reasoning": "Mostly correct"}'
        result = parse_judge_response(response)
        assert result["score"] == 4
        assert result["reasoning"] == "Mostly correct"

    def test_parses_json_in_code_block(self):
        response = '```json\n{"score": 3, "reasoning": "Some issues"}\n```'
        result = parse_judge_response(response)
        assert result["score"] == 3

    def test_parses_json_with_surrounding_text(self):
        response = 'Here is my evaluation:\n{"score": 5, "reasoning": "Perfect"}\nDone.'
        result = parse_judge_response(response)
        assert result["score"] == 5

    def test_returns_none_for_invalid_json(self):
        result = parse_judge_response("This is not JSON at all")
        assert result is None

    def test_returns_none_for_missing_score(self):
        response = '{"reasoning": "Good answer but no score"}'
        result = parse_judge_response(response)
        assert result is None

    def test_returns_none_for_score_out_of_range(self):
        response = '{"score": 7, "reasoning": "Invalid score"}'
        result = parse_judge_response(response)
        assert result is None

    def test_returns_none_for_zero_score(self):
        response = '{"score": 0, "reasoning": "Zero is out of range"}'
        result = parse_judge_response(response)
        assert result is None

    def test_returns_none_for_negative_score(self):
        response = '{"score": -1, "reasoning": "Negative"}'
        result = parse_judge_response(response)
        assert result is None

    def test_handles_float_score(self):
        response = '{"score": 4.0, "reasoning": "Float score"}'
        result = parse_judge_response(response)
        assert result["score"] == 4
        assert isinstance(result["score"], int)

    def test_handles_missing_reasoning(self):
        response = '{"score": 3}'
        result = parse_judge_response(response)
        assert result["score"] == 3
        assert result["reasoning"] == "No reasoning provided"

    def test_parses_code_block_without_json_label(self):
        response = '```\n{"score": 2, "reasoning": "Low quality"}\n```'
        result = parse_judge_response(response)
        assert result["score"] == 2


class TestCallJudge:
    """Tests for calling the judge with mocked API."""

    @patch("eval.judges.get_client")
    def test_call_judge_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"score": 4, "reasoning": "Good answer"}')
        ]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_judge(
            "Rate this response",
            {"model": "claude-sonnet-4-20250514", "max_tokens": 512, "temperature": 0.0},
        )

        assert result is not None
        assert result["score"] == 4
        mock_client.messages.create.assert_called_once()

    @patch("eval.judges.get_client")
    def test_call_judge_api_error(self, mock_get_client):
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="Rate limited",
            request=MagicMock(),
            body=None,
        )
        mock_get_client.return_value = mock_client

        result = call_judge(
            "Rate this response",
            {"model": "claude-sonnet-4-20250514", "max_tokens": 512},
        )

        assert result is None

    @patch("eval.judges.get_client")
    def test_call_judge_unparseable_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I cannot rate this.")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_judge(
            "Rate this response",
            {"model": "claude-sonnet-4-20250514", "max_tokens": 512},
        )

        assert result is None


class TestCallModel:
    """Tests for calling the model under evaluation with mocked API."""

    @patch("eval.judges.get_client")
    def test_call_model_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The answer is 42")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_model(
            "What is the answer?",
            {"name": "claude-sonnet-4-20250514", "max_tokens": 1024},
        )

        assert result == "The answer is 42"

    @patch("eval.judges.get_client")
    def test_call_model_with_system_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response with system prompt")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_model(
            "Hello",
            {"name": "claude-sonnet-4-20250514", "max_tokens": 1024},
            system_prompt="You are helpful.",
            temperature=0.5,
        )

        assert result is not None
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs[1]["system"] == "You are helpful."
        assert call_kwargs[1]["temperature"] == 0.5

    @patch("eval.judges.get_client")
    def test_call_model_returns_none_on_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_client

        result = call_model(
            "Hello",
            {"name": "claude-sonnet-4-20250514", "max_tokens": 1024},
        )

        assert result is None


class TestGetClient:
    """Tests for client initialization."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_get_client_with_key(self):
        client = get_client()
        assert client is not None

    @patch.dict("os.environ", {}, clear=True)
    def test_get_client_without_key_raises(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_client()
