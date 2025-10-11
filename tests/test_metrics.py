"""Tests for the metrics module.

These tests don't need API mocking since metrics are pure computation.
"""

import pytest

from eval.metrics import (
    compute_mean,
    compute_std,
    compute_confidence_interval,
    summarize_dimension,
    summarize_experiment,
    compare_conditions,
    compute_judge_agreement,
)


class TestComputeMean:
    def test_basic_mean(self):
        assert compute_mean([1, 2, 3, 4, 5]) == 3.0

    def test_single_value(self):
        assert compute_mean([4]) == 4.0

    def test_empty_list(self):
        assert compute_mean([]) is None

    def test_float_values(self):
        result = compute_mean([1.5, 2.5, 3.5])
        assert abs(result - 2.5) < 0.001


class TestComputeStd:
    def test_basic_std(self):
        result = compute_std([2, 4, 4, 4, 5, 5, 7, 9])
        assert result is not None
        assert abs(result - 2.138) < 0.01

    def test_identical_values(self):
        result = compute_std([3, 3, 3, 3])
        assert result == 0.0

    def test_single_value(self):
        assert compute_std([5]) is None

    def test_empty_list(self):
        assert compute_std([]) is None


class TestComputeConfidenceInterval:
    def test_basic_ci(self):
        values = [3, 4, 4, 5, 5, 4, 3, 5, 4, 4]
        ci = compute_confidence_interval(values)
        assert ci is not None
        lower, upper = ci
        assert lower < 4.1
        assert upper > 4.1
        assert lower > 0
        assert upper <= 5.5

    def test_insufficient_data(self):
        assert compute_confidence_interval([3]) is None

    def test_empty_list(self):
        assert compute_confidence_interval([]) is None


class TestSummarizeDimension:
    def test_normal_scores(self):
        scores = [
            {"score": 4, "reasoning": "good"},
            {"score": 5, "reasoning": "great"},
            {"score": 3, "reasoning": "ok"},
        ]
        result = summarize_dimension(scores, "factual")
        assert result["dimension"] == "factual"
        assert result["total_prompts"] == 3
        assert result["scored_prompts"] == 3
        assert result["failed_prompts"] == 0
        assert result["mean"] == 4.0
        assert result["min"] == 3
        assert result["max"] == 5

    def test_with_none_scores(self):
        scores = [
            {"score": 4, "reasoning": "good"},
            {"score": None, "reasoning": "failed"},
            {"score": 3, "reasoning": "ok"},
        ]
        result = summarize_dimension(scores, "adherence")
        assert result["scored_prompts"] == 2
        assert result["failed_prompts"] == 1
        assert result["mean"] == 3.5

    def test_all_none_scores(self):
        scores = [
            {"score": None, "reasoning": "failed"},
            {"score": None, "reasoning": "also failed"},
        ]
        result = summarize_dimension(scores, "hallucination")
        assert result["mean"] is None
        assert result["scored_prompts"] == 0


class TestSummarizeExperiment:
    def test_multi_dimension_summary(self):
        results = {
            "factual": [
                {"score": 4, "reasoning": "good"},
                {"score": 5, "reasoning": "great"},
            ],
            "adherence": [
                {"score": 3, "reasoning": "ok"},
                {"score": 4, "reasoning": "good"},
            ],
        }
        summary = summarize_experiment(results)
        assert "dimensions" in summary
        assert summary["overall_mean"] is not None
        assert summary["total_evaluations"] == 4

    def test_empty_results(self):
        summary = summarize_experiment({})
        assert summary["overall_mean"] is None
        assert summary["total_evaluations"] == 0


class TestCompareConditions:
    def test_comparison_sorting(self):
        condition_results = {
            "temp_0.0": {
                "overall_mean": 4.2,
                "dimensions": {"factual": {"mean": 4.5}, "adherence": {"mean": 3.9}},
            },
            "temp_1.0": {
                "overall_mean": 3.8,
                "dimensions": {"factual": {"mean": 3.5}, "adherence": {"mean": 4.1}},
            },
        }
        result = compare_conditions(condition_results)
        assert len(result) == 2
        assert result[0]["condition"] == "temp_0.0"
        assert result[0]["overall_mean"] > result[1]["overall_mean"]


class TestComputeJudgeAgreement:
    def test_perfect_agreement(self):
        trial_1 = [{"score": 4}, {"score": 5}, {"score": 3}]
        trial_2 = [{"score": 4}, {"score": 5}, {"score": 3}]
        result = compute_judge_agreement([trial_1, trial_2])
        assert result["agreement_rate"] == 1.0
        assert result["mean_deviation"] == 0.0

    def test_partial_agreement(self):
        trial_1 = [{"score": 4}, {"score": 5}, {"score": 3}]
        trial_2 = [{"score": 4}, {"score": 4}, {"score": 3}]
        result = compute_judge_agreement([trial_1, trial_2])
        assert 0 < result["agreement_rate"] < 1

    def test_no_agreement(self):
        trial_1 = [{"score": 1}, {"score": 2}, {"score": 3}]
        trial_2 = [{"score": 5}, {"score": 4}, {"score": 1}]
        result = compute_judge_agreement([trial_1, trial_2])
        assert result["agreement_rate"] == 0.0

    def test_single_trial(self):
        result = compute_judge_agreement([[{"score": 4}]])
        assert result["agreement_rate"] is None

    def test_handles_none_scores(self):
        trial_1 = [{"score": 4}, {"score": None}, {"score": 3}]
        trial_2 = [{"score": 4}, {"score": None}, {"score": 3}]
        result = compute_judge_agreement([trial_1, trial_2])
        assert result["agreement_rate"] == 1.0
        assert result["total_comparisons"] == 2

    def test_three_trials(self):
        trial_1 = [{"score": 4}, {"score": 5}]
        trial_2 = [{"score": 4}, {"score": 5}]
        trial_3 = [{"score": 4}, {"score": 4}]
        result = compute_judge_agreement([trial_1, trial_2, trial_3])
        assert result["exact_matches"] == 1
        assert result["total_comparisons"] == 2
