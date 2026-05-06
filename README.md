# Evaluating Small-Model Agent Networks Against Large LLMs on Persistent Information Streams

## Research Question

Can a heterogeneous network of SLM agents match or outperform a single large LLM on stream reasoning tasks?

This prototype is for Jeffrey Zhou's research direction with Professor K. Mani Chandy: comparing a monolithic large-model agent against an adaptive network of smaller role-specialized agents.

## Simplified Comparison

Baseline A = single large LLM.

Method D = heterogeneous SLM agent network.

The prototype intentionally skips the single-small-model baseline, homogeneous swarm, and budget optimization for now. The first question is whether the small-agent network can recover comparable accuracy on persistent information streams.

## Why Synthetic Streams

Synthetic streams give controlled gold answers while matching the shape of realistic persistent streams: email threads, Slack channels, article updates, lab memos, paper logistics, conference planning, and experiment status messages.

Each task contains 3-7 timestamped messages. Later messages may update, correct, or contradict earlier messages. The model must answer a final question about the latest state, a contradiction, an assignment, a priority, or a multi-hop dependency.

## Benchmark Plan

Current: `data/synthetic_streams.jsonl`.

Next: GAIA Level 1 validation subset.

GAIA is a benchmark for general AI assistants with reasoning, tool use, browsing, and short unambiguous answers. The pipeline is organized so a future GAIA loader can be added without rewriting the model runners, result schema, or summary code.

TODOs for GAIA:

- Add a GAIA Level 1 loader that emits a compatible benchmark task object.
- Add tool-use and browsing adapters where GAIA requires them.
- Keep the same `MethodResult` contract so baseline and swarm methods remain comparable.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `OPENROUTER_API_KEY` in `.env`.

## OpenRouter Notes

You need an OpenRouter account and API key.

Free models may have rate limits. Paid models require credits.

Model names can be configured through `.env` or CLI flags:

```bash
SMALL_MODEL=openrouter/google/gemma-3-4b-it:free
LARGE_MODEL=openrouter/google/gemma-4-31b-it:free
```

## Run

```bash
python -m src.run_experiment --mode both --limit 5
```

Other useful runs:

```bash
python -m src.run_experiment --mode baseline
python -m src.run_experiment --mode swarm
python -m src.run_experiment --mode both --limit 10 --small-model openrouter/google/gemma-3-4b-it:free --large-model openrouter/google/gemma-4-31b-it:free
```

Results are saved to `results/run_TIMESTAMP.json` by default.

## Metrics

The evaluator reports:

- exact match
- contains gold
- score
- token usage
- latency
- number of model calls

Scoring is intentionally simple:

- `1.0` for normalized exact match
- `0.75` if the normalized gold answer appears inside the normalized prediction
- `0.0` otherwise

Token counts are collected from LiteLLM usage fields when available. If the provider response does not include token usage, token fields are saved as `null` and the run continues.

## Current Limitations

- synthetic benchmark
- simple automatic scoring
- no live streams yet
- no integration with Professor Chandy's message-passing system yet
- one small model and one large model at a time
- no budget optimization yet

## Next Steps

- plug into distributed message-passing agents
- add GAIA loader
- add LLM-as-judge evaluation
- add real streams
- compare more models later
