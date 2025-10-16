"""Tests for the evaluation runner.

All API calls are mocked so tests run without an API key.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from eval.runner import EvalRunner


@pytest.fixture
def runner(tmp_path):
    """Create an EvalRunner with the real config and prompts files."""
    # The runner expects to find files relative to cwd, so we use
    # the actual project files
    return EvalRunner(config_path="config.yaml")


class TestEvalRunnerInit:
    def test_loads_config(self, runner):
        assert "model" in runner.config
        assert "judge" in runner.config
        assert runner.config["model"]["name"] == "claude-3-5-sonnet-20241022"

    def test_loads_prompts(self, runner):
        assert len(runner.prompts) > 0
        categories = {p["category"] for p in runner.prompts}
        assert "factual" in categories
        assert "adherence" in categories
        assert "hallucination" in categories
        assert "consistency" in categories

    def test_loads_rubrics(self, runner):
        assert "factual_accuracy" in runner.rubrics
        assert "instruction_adherence" in runner.rubrics
        assert "hallucination" in runner.rubrics
        assert "consistency" in runner.rubrics

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            EvalRunner(config_path="nonexistent.yaml")


class TestGetPromptsByCategory:
    def test_factual_prompts(self, runner):
        factual = runner.get_prompts_by_category("factual")
        assert len(factual) > 0
        assert all(p["category"] == "factual" for p in factual)

    def test_adherence_prompts(self, runner):
        adherence = runner.get_prompts_by_category("adherence")
        assert len(adherence) > 0
        assert all("format_requirements" in p for p in adherence)

    def test_hallucination_prompts(self, runner):
        halluc = runner.get_prompts_by_category("hallucination")
        assert len(halluc) > 0
        assert all("context" in p for p in halluc)

    def test_empty_category(self, runner):
        result = runner.get_prompts_by_category("nonexistent")
        assert result == []


class TestRunFactualEvaluation:
    @patch("eval.dimensions.factual.call_judge")
    @patch("eval.judges.get_client")
    def test_runs_factual_eval(self, mock_get_client, mock_call_judge, runner):
        # Mock the model response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Gold has the chemical symbol Au.")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Mock the judge
        mock_call_judge.return_value = {"score": 5, "reasoning": "Correct"}

        results = runner.run_factual_evaluation(temperature=0.0)

        assert len(results) > 0
        assert results[0]["score"] == 5

    @patch("eval.judges.get_client")
    def test_handles_model_failure(self, mock_get_client, runner):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API down")
        mock_get_client.return_value = mock_client

        results = runner.run_factual_evaluation()

        assert len(results) > 0
        assert results[0]["score"] is None


class TestRunAdherenceEvaluation:
    @patch("eval.dimensions.adherence.call_judge")
    @patch("eval.judges.get_client")
    def test_runs_adherence_eval(self, mock_get_client, mock_call_judge, runner):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="France\nGermany\nItaly\nSpain\nPortugal")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        mock_call_judge.return_value = {"score": 5, "reasoning": "Perfect format"}

        results = runner.run_adherence_evaluation(temperature=0.0)

        assert len(results) > 0
        # Adherence results should include heuristic checks
        assert "heuristic_checks" in results[0]


class TestRunHallucinationEvaluation:
    @patch("eval.dimensions.hallucination.call_judge")
    @patch("eval.judges.get_client")
    def test_runs_hallucination_eval(self, mock_get_client, mock_call_judge, runner):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The Eiffel Tower was completed in 1889.")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        mock_call_judge.return_value = {"score": 5, "reasoning": "No hallucination"}

        results = runner.run_hallucination_evaluation(temperature=0.0)

        assert len(results) > 0
        assert "grounding_check" in results[0]


class TestRunConsistencyEvaluation:
    @patch("eval.dimensions.consistency.call_judge")
    @patch("eval.judges.get_client")
    def test_runs_consistency_eval(self, mock_get_client, mock_call_judge, runner):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="4")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        mock_call_judge.return_value = {"score": 5, "reasoning": "Same answer"}

        results = runner.run_consistency_evaluation(
            temperature=0.0, num_trials=3,
        )

        assert len(results) > 0


class TestSaveResults:
    @patch("eval.judges.get_client")
    def test_saves_results_to_json(self, mock_get_client, runner, tmp_path):
        runner.config["output"]["results_dir"] = str(tmp_path)

        results = {
            "results": {"factual": [{"score": 4, "reasoning": "good", "response": "text"}]},
            "summary": {"overall_mean": 4.0, "total_evaluations": 1, "dimensions": {}},
        }

        runner.save_results(results, "test_results.json")

        output_file = tmp_path / "test_results.json"
        assert output_file.exists()

        with open(output_file) as f:
            saved = json.load(f)

        # Response text should be stripped
        assert "response" not in saved["results"]["factual"][0]


class TestStripResponses:
    def test_removes_response_key(self, runner):
        data = {"results": [{"score": 4, "response": "long text here"}]}
        stripped = runner._strip_responses(data)
        assert "response" not in stripped["results"][0]

    def test_removes_responses_key(self, runner):
        data = {"results": [{"score": 4, "responses": ["a", "b", "c"]}]}
        stripped = runner._strip_responses(data)
        assert "responses" not in stripped["results"][0]

    def test_preserves_other_keys(self, runner):
        data = {"results": [{"score": 4, "reasoning": "good", "response": "text"}]}
        stripped = runner._strip_responses(data)
        assert stripped["results"][0]["score"] == 4
        assert stripped["results"][0]["reasoning"] == "good"
