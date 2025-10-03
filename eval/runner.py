"""Main evaluation runner.

Orchestrates the full evaluation pipeline: loads config and prompts,
runs the model, scores across all dimensions, and collects results
for analysis.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from eval.judges import call_model
from eval.dimensions.factual import score_factual_accuracy
from eval.dimensions.adherence import score_instruction_adherence
from eval.dimensions.hallucination import score_hallucination
from eval.dimensions.consistency import score_consistency
from eval.metrics import summarize_experiment, compare_conditions, compute_judge_agreement

logger = logging.getLogger(__name__)


class EvalRunner:
    """Runs the full evaluation pipeline.

    Loads configuration, test prompts, and rubrics, then evaluates
    model outputs across all four dimensions.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self.rubrics = self._load_rubrics()
        self.results = {}

    def _load_config(self, config_path: str) -> dict:
        """Load the YAML configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            return yaml.safe_load(f)

    def _load_prompts(self) -> list[dict]:
        """Load test prompts from the JSON file."""
        prompts_path = Path("prompts/test_prompts.json")
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(prompts_path) as f:
            data = json.load(f)
        return data["prompts"]

    def _load_rubrics(self) -> dict:
        """Load scoring rubrics from YAML."""
        rubrics_path = Path("prompts/rubrics.yaml")
        if not rubrics_path.exists():
            raise FileNotFoundError(f"Rubrics file not found: {rubrics_path}")

        with open(rubrics_path) as f:
            return yaml.safe_load(f)

    def get_prompts_by_category(self, category: str) -> list[dict]:
        """Filter prompts by evaluation category."""
        return [p for p in self.prompts if p["category"] == category]

    def run_factual_evaluation(
        self,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run factual accuracy evaluation on all factual prompts.

        Args:
            temperature: Temperature for the model being evaluated.
            system_prompt: Optional system prompt for the model.

        Returns:
            List of score dictionaries.
        """
        factual_prompts = self.get_prompts_by_category("factual")
        model_config = self.config["model"]
        judge_config = self.config["judge"]
        rubric = self.rubrics["factual_accuracy"]
        results = []

        for prompt_data in factual_prompts:
            logger.info(f"Evaluating factual: {prompt_data['id']}")

            response = call_model(
                prompt_data["prompt"],
                model_config,
                system_prompt=system_prompt,
                temperature=temperature,
            )

            if response is None:
                results.append({
                    "prompt_id": prompt_data["id"],
                    "score": None,
                    "reasoning": "Model failed to respond",
                })
                continue

            score = score_factual_accuracy(
                prompt=prompt_data["prompt"],
                response=response,
                expected_facts=prompt_data["expected_facts"],
                rubric=rubric,
                judge_config=judge_config,
            )
            score["prompt_id"] = prompt_data["id"]
            score["response"] = response
            results.append(score)

        return results

    def run_adherence_evaluation(
        self,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run instruction adherence evaluation."""
        adherence_prompts = self.get_prompts_by_category("adherence")
        model_config = self.config["model"]
        judge_config = self.config["judge"]
        rubric = self.rubrics["instruction_adherence"]
        results = []

        for prompt_data in adherence_prompts:
            logger.info(f"Evaluating adherence: {prompt_data['id']}")

            response = call_model(
                prompt_data["prompt"],
                model_config,
                system_prompt=system_prompt,
                temperature=temperature,
            )

            if response is None:
                results.append({
                    "prompt_id": prompt_data["id"],
                    "score": None,
                    "reasoning": "Model failed to respond",
                    "heuristic_checks": {},
                })
                continue

            score = score_instruction_adherence(
                prompt=prompt_data["prompt"],
                response=response,
                format_requirements=prompt_data["format_requirements"],
                rubric=rubric,
                judge_config=judge_config,
            )
            score["prompt_id"] = prompt_data["id"]
            score["response"] = response
            results.append(score)

        return results

    def run_hallucination_evaluation(
        self,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run hallucination detection evaluation."""
        halluc_prompts = self.get_prompts_by_category("hallucination")
        model_config = self.config["model"]
        judge_config = self.config["judge"]
        rubric = self.rubrics["hallucination"]
        results = []

        for prompt_data in halluc_prompts:
            logger.info(f"Evaluating hallucination: {prompt_data['id']}")

            response = call_model(
                prompt_data["prompt"],
                model_config,
                system_prompt=system_prompt,
                temperature=temperature,
            )

            if response is None:
                results.append({
                    "prompt_id": prompt_data["id"],
                    "score": None,
                    "reasoning": "Model failed to respond",
                    "grounding_check": {},
                })
                continue

            score = score_hallucination(
                prompt=prompt_data["prompt"],
                response=response,
                context=prompt_data["context"],
                expected_answer=prompt_data["expected_answer"],
                rubric=rubric,
                judge_config=judge_config,
            )
            score["prompt_id"] = prompt_data["id"]
            score["response"] = response
            results.append(score)

        return results

    def run_consistency_evaluation(
        self,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        num_trials: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run consistency evaluation by generating multiple responses per prompt."""
        consistency_prompts = self.get_prompts_by_category("consistency")
        model_config = self.config["model"]
        judge_config = self.config["judge"]
        rubric = self.rubrics["consistency"]
        trials = num_trials or self.config["experiments"]["consistency_trials"]
        results = []

        for prompt_data in consistency_prompts:
            logger.info(f"Evaluating consistency: {prompt_data['id']}")

            responses = []
            for trial in range(trials):
                response = call_model(
                    prompt_data["prompt"],
                    model_config,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                if response is not None:
                    responses.append(response)

            if len(responses) < 2:
                results.append({
                    "prompt_id": prompt_data["id"],
                    "score": None,
                    "reasoning": "Not enough successful responses for consistency check",
                    "pairwise_scores": [],
                    "variance": None,
                })
                continue

            score = score_consistency(
                prompt=prompt_data["prompt"],
                responses=responses,
                rubric=rubric,
                judge_config=judge_config,
            )
            score["prompt_id"] = prompt_data["id"]
            score["responses"] = responses
            results.append(score)

        return results

    def run_full_evaluation(
        self,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run evaluation across all four dimensions.

        Args:
            temperature: Temperature setting for the model.
            system_prompt: Optional system prompt.

        Returns:
            Dictionary with results for each dimension and a summary.
        """
        logger.info(f"Starting full evaluation (temp={temperature})")
        start_time = time.time()

        results = {
            "factual": self.run_factual_evaluation(temperature, system_prompt),
            "adherence": self.run_adherence_evaluation(temperature, system_prompt),
            "hallucination": self.run_hallucination_evaluation(temperature, system_prompt),
            "consistency": self.run_consistency_evaluation(temperature, system_prompt),
        }

        elapsed = time.time() - start_time
        summary = summarize_experiment(results)
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["temperature"] = temperature
        summary["system_prompt"] = system_prompt

        return {"results": results, "summary": summary}

    def run_temperature_comparison(self) -> dict[str, Any]:
        """Run the full evaluation at each configured temperature setting.

        Returns:
            Dictionary mapping temperature labels to their results.
        """
        temperatures = self.config["experiments"]["temperatures"]
        system_prompt = self.config["experiments"]["system_prompts"]["minimal"]

        condition_results = {}
        for temp in temperatures:
            label = f"temp_{temp}"
            logger.info(f"Running temperature condition: {label}")
            eval_result = self.run_full_evaluation(
                temperature=temp, system_prompt=system_prompt
            )
            condition_results[label] = eval_result

        comparison = compare_conditions(
            {k: v["summary"] for k, v in condition_results.items()}
        )

        return {
            "conditions": condition_results,
            "comparison": comparison,
        }

    def run_system_prompt_comparison(self) -> dict[str, Any]:
        """Compare evaluation results across different system prompts.

        Returns:
            Dictionary with results for each system prompt condition.
        """
        system_prompts = self.config["experiments"]["system_prompts"]

        condition_results = {}
        for label, system_prompt in system_prompts.items():
            logger.info(f"Running system prompt condition: {label}")
            eval_result = self.run_full_evaluation(
                temperature=0.0, system_prompt=system_prompt
            )
            condition_results[label] = eval_result

        comparison = compare_conditions(
            {k: v["summary"] for k, v in condition_results.items()}
        )

        return {
            "conditions": condition_results,
            "comparison": comparison,
        }

    def run_judge_agreement_test(self) -> dict[str, Any]:
        """Test judge consistency by scoring the same data multiple times.

        Returns:
            Agreement statistics for the judge.
        """
        num_trials = self.config["experiments"]["judge_agreement_trials"]

        # Use factual prompts as a representative sample
        factual_prompts = self.get_prompts_by_category("factual")[:5]
        model_config = self.config["model"]
        judge_config = self.config["judge"]
        rubric = self.rubrics["factual_accuracy"]

        # First, get model responses (only once)
        responses = {}
        for prompt_data in factual_prompts:
            response = call_model(prompt_data["prompt"], model_config)
            if response is not None:
                responses[prompt_data["id"]] = {
                    "prompt_data": prompt_data,
                    "response": response,
                }

        # Then score the same responses multiple times
        trial_scores = []
        for trial_num in range(num_trials):
            logger.info(f"Judge agreement trial {trial_num + 1}/{num_trials}")
            trial = []
            for prompt_id, data in responses.items():
                score = score_factual_accuracy(
                    prompt=data["prompt_data"]["prompt"],
                    response=data["response"],
                    expected_facts=data["prompt_data"]["expected_facts"],
                    rubric=rubric,
                    judge_config=judge_config,
                )
                trial.append(score)
            trial_scores.append(trial)

        agreement = compute_judge_agreement(trial_scores)
        return agreement

    def save_results(self, results: dict[str, Any], filename: str = "results.json"):
        """Save results to a JSON file in the results directory.

        Args:
            results: Results dictionary to save.
            filename: Output filename.
        """
        results_dir = Path(self.config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        output_path = results_dir / filename

        # Remove response text to keep file size manageable
        clean_results = self._strip_responses(results)

        with open(output_path, "w") as f:
            json.dump(clean_results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def _strip_responses(self, obj: Any) -> Any:
        """Remove 'response' and 'responses' keys to reduce output size."""
        if isinstance(obj, dict):
            return {
                k: self._strip_responses(v)
                for k, v in obj.items()
                if k not in ("response", "responses")
            }
        elif isinstance(obj, list):
            return [self._strip_responses(item) for item in obj]
        return obj


def main():
    """Entry point for running evaluations from the command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run LLM evaluation framework")
    parser.add_argument(
        "--experiment",
        choices=["full", "temperature", "system_prompt", "judge_agreement"],
        default="full",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for full evaluation",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()
    runner = EvalRunner(config_path=args.config)

    if args.experiment == "full":
        results = runner.run_full_evaluation(temperature=args.temperature)
        runner.save_results(results, "full_eval.json")
    elif args.experiment == "temperature":
        results = runner.run_temperature_comparison()
        runner.save_results(results, "temperature_comparison.json")
    elif args.experiment == "system_prompt":
        results = runner.run_system_prompt_comparison()
        runner.save_results(results, "system_prompt_comparison.json")
    elif args.experiment == "judge_agreement":
        results = runner.run_judge_agreement_test()
        runner.save_results(results, "judge_agreement.json")

    print("Evaluation complete. Results saved to results/")


if __name__ == "__main__":
    main()
