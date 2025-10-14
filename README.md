# llm-eval-framework

Config-driven framework for evaluating LLM outputs across four dimensions: factual accuracy, instruction adherence, hallucination detection, and output consistency. Uses the LLM-as-judge pattern with Claude evaluating Claude.

## Why I built this

I kept running into the same problem: I'd tweak a system prompt or change the temperature and have no structured way to tell if things got better or worse. Existing benchmarks are great for leaderboard comparisons, but they don't let you define your own rubrics or test your own prompts. I wanted something closer to a unit test suite for model behavior.

The LLM-as-judge approach has real problems (more on that below), but it's fast enough for iterating and the framework is set up to show you where the judge contradicts itself, so you know when to be skeptical.

## What it measures

Four dimensions, each scored 1-5 by a judge model.

*Factual accuracy* checks if the model gets verifiable facts right. I test it with questions that have known answers (chemical symbols, historical dates, that sort of thing) and compare the response against expected facts. A fast string-match heuristic runs alongside the LLM judge for sanity checking.

*Instruction adherence* checks whether the model follows formatting constraints. Things like "respond in exactly 5 lines" or "return valid JSON with these keys." This dimension has both an LLM judge score and automated heuristic checks (counting lines, parsing JSON, checking word counts). The heuristics catch cases where the judge says the response looks fine but the format is technically wrong.

*Hallucination detection* gives the model a context paragraph and asks it questions. Some questions are answerable from the text, some aren't. A good model should say "the text doesn't mention that" instead of making something up. The scorer tracks word overlap between the response and source context as an additional grounding signal.

*Consistency* runs the same prompt 5 times and measures how much the answers vary. I use pairwise semantic similarity judgments (via the LLM judge) rather than exact string matching, since two correct answers can be worded differently. High variance on factual questions usually means the model is guessing.

## Architecture

```
llm-eval-framework/
    config.yaml              # Model, judge, experiment settings
    prompts/
        test_prompts.json    # 42 test prompts across difficulty levels
        rubrics.yaml         # Scoring rubrics for each dimension
    eval/
        runner.py            # Main evaluation orchestrator
        judges.py            # LLM-as-judge API calls and parsing
        metrics.py           # Aggregation and statistics
        dimensions/
            factual.py       # Factual accuracy scoring
            adherence.py     # Instruction following checks
            hallucination.py # Grounding and hallucination detection
            consistency.py   # Multi-trial consistency scoring
    analysis/
        visualize.py         # Chart generation (colorblind-friendly)
        report.py            # Markdown report generation
    tests/
        test_judges.py
        test_metrics.py
        test_runner.py
```

## Setup

```bash
git clone https://github.com/nthommen/llm-eval-framework.git
cd llm-eval-framework
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

Run a full evaluation:
```bash
python -m eval.runner --experiment full --temperature 0.0
```

Compare temperature settings:
```bash
python -m eval.runner --experiment temperature
```

Compare system prompts (minimal vs. detailed):
```bash
python -m eval.runner --experiment system_prompt
```

Test judge self-agreement:
```bash
python -m eval.runner --experiment judge_agreement
```

Run tests (no API key needed):
```bash
python -m pytest tests/ -v
```

## Design decisions

Rubric definitions are separate from scoring logic. The rubrics live in `prompts/rubrics.yaml` and the code that applies them lives in the dimension modules. You can change scoring criteria without touching Python, and different teams can maintain their own rubric files for their use cases.

Everything configurable goes in `config.yaml`. Model name, temperatures to compare, system prompts, number of consistency trials. I didn't want magic numbers buried in source files.

For instruction adherence I run deterministic heuristic checks (line counts, JSON validation, word counts) in parallel with the LLM judge. When they disagree, that's useful data about the judge's reliability on that particular check type.

The visualization module uses the IBM Design Library color palette, which is designed for accessibility across common forms of color vision deficiency.

Some prompts are designed to bait the model into hallucinating by asking about things not in the provided context. The hallucination detector recognizes a correct refusal ("the text doesn't mention that") as a positive signal rather than penalizing it.

## Known limitations

I want to be clear about the problems with LLM-as-judge:

- Claude judging Claude will probably inflate scores. A more honest test would cross-evaluate across model families.
- Longer responses tend to score higher even when a shorter answer is just as good. The rubrics push against this but don't fully solve it.
- In pairwise consistency comparisons, the response listed first might be favored. I haven't added position randomization yet.
- The 1-5 scale is coarse. Some evaluations would benefit from finer-grained scoring, but the judge gets unreliable when you ask it to distinguish between adjacent points on a wider scale.

## Sample results

Results below are from a run using `claude-sonnet-4-20250514` with the minimal system prompt. Single run, not averaged across sessions.

### Scores at temperature 0.0

| Dimension | Mean | Std Dev | Min | Max | N |
|-----------|------|---------|-----|-----|---|
| Factual Accuracy | 4.58 | 0.67 | 3 | 5 | 12 |
| Instruction Adherence | 4.17 | 0.94 | 2 | 5 | 12 |
| Hallucination Detection | 4.75 | 0.46 | 4 | 5 | 8 |
| Consistency | 4.23 | 0.71 | 3 | 5 | 8 |

### Temperature comparison

| Condition | Factual | Adherence | Hallucination | Consistency | Overall |
|-----------|---------|-----------|---------------|-------------|---------|
| temp 0.0 | 4.58 | 4.17 | 4.75 | 4.23 | 4.43 |
| temp 0.5 | 4.42 | 3.92 | 4.50 | 3.81 | 4.16 |
| temp 1.0 | 4.17 | 3.58 | 4.13 | 3.19 | 3.77 |

No surprise: lower temperature gives more consistent and accurate outputs. Consistency shows the biggest drop at high temperature, which is what you'd expect since temperature directly controls sampling randomness.

The interesting thing is that instruction adherence degrades faster than factual accuracy. I think factual knowledge is baked into the weights and doesn't move much with sampling noise. But following format constraints requires holding instructions in working memory across the full generation, and that's more fragile.

### System prompt comparison

| Condition | Factual | Adherence | Hallucination | Consistency | Overall |
|-----------|---------|-----------|---------------|-------------|---------|
| detailed | 4.67 | 4.33 | 4.88 | 4.15 | 4.51 |
| minimal | 4.58 | 4.17 | 4.75 | 4.23 | 4.43 |

The detailed system prompt helps a bit, but less than I expected. Hallucination detection benefits the most from explicit grounding instructions.

### Judge self-agreement

Over 3 trials scoring the same 5 prompt-response pairs:

- Exact agreement rate: 80%
- Mean score deviation: 0.20

When the judge disagreed with itself it was always by 1 point (a 4 vs a 5, that kind of thing). Never saw a 2-point swing, which is encouraging for calibration even though it's not perfectly deterministic.

## What I'd build next

If I were taking this further:

1. Position randomization for pairwise consistency judgments to control for ordering effects.
2. Cross-model evaluation. Use a different model family as judge to quantify self-preference bias.
3. A web UI for browsing individual prompt results instead of reading markdown tables.
4. RAG-specific evaluation where retrieval quality is scored separately from generation quality.
5. Rubric versioning to track how scoring criteria change over time and whether those changes actually improve signal.

## License

MIT
