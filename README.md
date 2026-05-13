# Evaluating Small-Model Agent Networks Against Large LLMs on Persistent Information Streams

## Research Question

Can a heterogeneous network of small LLM agents match or outperform a single larger LLM on persistent-stream reasoning tasks?

This prototype compares:

- Baseline A: one larger LLM acting as a monolithic agent.
- Method D: a heterogeneous network of smaller role-specialized agents.

The goal is not to prove that a small swarm wins on a toy benchmark. The goal is to build a fair evaluation framework for studying when small-agent networks can match a larger model, measuring accuracy, latency, token usage, model calls, and failure modes.

## Fairness Note

An earlier adaptive swarm used task-specific prompting and deterministic canonicalization tuned to this synthetic benchmark. That was useful for debugging, but it is not fair evidence against the large baseline because the same task-specific rules were not given to the baseline.

The main comparison now uses the fair 3-agent pipeline:

- FactExtractorAgent
- StateAndContradictionAgent
- AnswerAndVerifierAgent

Both baseline and swarm receive the same shared answer-format contract. Generic answer cleanup and scoring are applied equally to both methods. The adaptive architecture still exists as an exploratory ablation only and prints a warning when used.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Option 1: Ollama

Recommended for local experiments. It is free, private, and avoids hosted API rate limits.

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
```

Use this `.env`:

```env
SMALL_MODEL=ollama/gemma3:4b
LARGE_MODEL=ollama/gemma3:12b
```

Default fair pilot:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --run-name fair_pipeline_gemma4b_vs_gemma12b_5tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

Full fair synthetic run:

```bash
python -m src.run_experiment \
  --mode both \
  --run-name fair_pipeline_gemma4b_vs_gemma12b_30tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

On a 24GB Mac, avoid `gemma3:27b` for routine runs. Prefer `gemma3:12b` or another model that fits comfortably in memory.

## Option 2: OpenRouter

OpenRouter gives one API for many hosted models. Free models can be rate-limited or temporarily unavailable. Paid models are usually more reliable and require OpenRouter credits.

Use this `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
SMALL_MODEL=openrouter/qwen/qwen3-4b:free
LARGE_MODEL=openrouter/google/gemma-4-31b-it:free
```

Run:

```bash
python -m src.run_experiment --mode both --limit 1 --run-name openrouter_free_1task
```

## Data

The current benchmark is:

```text
data/synthetic_streams.jsonl
```

Each line is one task with:

- `stream`: 3-7 timestamped messages.
- `question`: the final question to answer.
- `gold_answer`: the primary short answer.
- `aliases`: optional acceptable short-answer variants.

The tasks simulate persistent email, Slack, article, memo, conference, and lab-update streams where later messages may update or contradict earlier ones.

## Metrics

The evaluator reports:

- exact match against gold or aliases
- contains match in either direction
- token F1
- aggregate score
- token usage when available
- latency
- number of model calls

Scoring:

- `1.0` for normalized exact match against gold or aliases.
- `0.8` if the gold/alias contains the prediction or the prediction contains the gold/alias.
- `0.5` if token F1 is at least `0.5`.
- `0.0` otherwise.

Generic answer cleanup is applied equally to baseline and swarm. It only strips formatting noise such as whitespace, surrounding quotes/backticks, trailing periods, and leading `the`. It does not use task type, gold answer, or benchmark-specific mappings.

## Outputs

Each run writes:

- JSON records: `results/run_TIMESTAMP.json` or `results/TIMESTAMP_RUNNAME.json`
- Markdown summary: `results/TIMESTAMP_RUNNAME_summary.md`

The Markdown report includes config, summary metrics, a per-task comparison table, and non-perfect cases.

## Code Layout

- `src/run_experiment.py`: command-line runner; chooses baseline, swarm, model names, output paths, and execution order.
- `src/agents/baseline_large.py`: Baseline A, one large model call over the full stream.
- `src/agents/swarm_pipeline.py`: fair Method D default, a role-specialized small-model pipeline.
- `src/agents/swarm_memory.py`: memory/retrieval Method D variant that selects evidence before answering.
- `src/agents/swarm_adaptive.py`: exploratory ablation with task-specific rules; not the main fair comparison.
- `src/prompts.py`: shared answer-format contract plus role prompts.
- `src/postprocess.py`: generic answer cleanup applied equally to both methods.
- `src/evaluator.py`: shared scoring logic for baseline and swarm.
- `src/report.py`: Markdown report generation.
- `data/synthetic_streams.jsonl`: synthetic stream tasks with gold answers and optional aliases.
- `tools/download_longmemeval.py`: downloads official LongMemEval cleaned JSON files.
- `tools/convert_longmemeval.py`: converts LongMemEval JSON into this project's JSONL task format.

## LongMemEval Adapter

LongMemEval is an external ICLR 2025 benchmark for long-term memory in chat assistants. It is much closer to this project than GAIA because it contains timestamped interaction histories, questions, and gold answers.

Download the small oracle file first:

```bash
python tools/download_longmemeval.py \
  --out-dir ../LongMemEval/data \
  --files oracle
```

Convert it into this project's `StreamTask` JSONL format:

```bash
python tools/convert_longmemeval.py \
  --input ../LongMemEval/data/longmemeval_oracle.json \
  --output data/longmemeval_oracle_useronly.jsonl \
  --user-only
```

Run a small smoke test:

```bash
python -m src.run_experiment \
  --data data/longmemeval_oracle_useronly.jsonl \
  --mode both \
  --limit 3 \
  --run-name longmemeval_oracle_smoke_gemma3_4b_vs_12b_3tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --warmup-models \
  --ollama-keep-alive 10m \
  --swarm-agents 3 \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

The `--user-only` conversion strips assistant turns to reduce local context size. This is useful for first local experiments, but it is not the full official LongMemEval setting.

## Useful Commands

Sanity run:

```bash
python -m src.run_experiment --mode both --limit 1
```

Fair 5-task pilot:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --run-name fair_pipeline_gemma4b_vs_gemma12b_5tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --swarm-agents 3
```

Run only the large baseline:

```bash
python -m src.run_experiment --mode baseline --limit 5 --run-name baseline_5tasks
```

Run only the fair 3-agent swarm:

```bash
python -m src.run_experiment --mode swarm --limit 5 --swarm-agents 3 --run-name fair_swarm_5tasks
```

Run the memory/retrieval swarm:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --run-name memory_swarm_gemma4b_vs_gemma12b_5tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --warmup-models \
  --ollama-keep-alive 10m \
  --ollama-num-ctx 8192 \
  --swarm-architecture memory \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

## Exploratory Ablations Only

The adaptive architecture is available for debugging and ablation studies:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --run-name ablation_adaptive_gemma4b_vs_gemma12b_5tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --swarm-architecture adaptive
```

This mode prints a warning because it uses task-specific prompts and canonicalization. Do not treat adaptive results as the main fair Baseline A vs Method D comparison.

## Benchmark Plan

Current:

- Synthetic persistent-stream benchmark with gold answers and aliases.
- LongMemEval oracle adapter for external long-memory smoke tests.

Next:

- Add full LongMemEval_S runs with retrieval/truncation strategies.
- Add GAIA Level 1 loader later.
- Add LLM-as-judge evaluation.
- Add real streams such as email, Slack, articles, and lab updates.
- Plug into Professor Chandy's distributed message-passing agent system.

GAIA is not implemented yet. The current code includes a loader stub so a future public benchmark can be added without rewriting the experiment pipeline.

## Current Limitations

- synthetic benchmark only
- aliases are manually curated
- no live streams yet
- no budget optimization yet
- no integration with Professor Chandy's message-passing system yet
