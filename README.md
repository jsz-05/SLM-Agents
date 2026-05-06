# Evaluating Small-Model Agent Networks Against Large LLMs on Persistent Information Streams

## Research Question

Can a heterogeneous network of small LLM agents match or outperform a single larger LLM on stream reasoning tasks?

This prototype compares:

- Baseline A: one larger LLM acting as a monolithic agent.
- Method D: a heterogeneous network of smaller role-specialized agents.

The first benchmark is synthetic and controlled: each task is a stream of messages where later messages may update, correct, or contradict earlier ones. The model must answer a final question about the current state, contradiction, priority, entity assignment, or multi-hop dependency.

## What Runs

The baseline sends the full stream and question to one model.

The swarm uses the same small model in five roles:

- FactExtractorAgent
- StateTrackerAgent
- ContradictionDetectorAgent
- AnswerAgent
- VerifierAgent

Results are scored against the task's gold answer and saved to `results/run_TIMESTAMP.json`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then choose one of the two model options below.

## Option 1: Ollama

This is the recommended local path. It is free, private, and does not need an API key.

Install/open Ollama, then pull models:

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
```

Use this `.env`:

```env
SMALL_MODEL=ollama/gemma3:4b
LARGE_MODEL=ollama/gemma3:12b
```

Run:

```bash
python -m src.run_experiment --mode both --limit 5
```

Equivalent explicit command:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

Other local pairs you can try:

```bash
python -m src.run_experiment --mode both --limit 5 --small-model ollama/qwen3:8b --large-model ollama/qwen3:14b
python -m src.run_experiment --mode both --limit 5 --small-model ollama/gemma3:4b --large-model ollama/gemma3:27b
```

## Option 2: OpenRouter

OpenRouter gives one API for many hosted models. It is convenient, but free models can be rate-limited or temporarily unavailable. Paid models are usually more reliable and require OpenRouter credits.

Use this `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
SMALL_MODEL=openrouter/qwen/qwen3-4b:free
LARGE_MODEL=openrouter/google/gemma-4-31b-it:free
```

Free OpenRouter run:

```bash
python -m src.run_experiment --mode both --limit 1
```

For more reliable experiments, choose paid model IDs from OpenRouter and set them in `.env` or pass them as CLI flags:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --small-model openrouter/some-small-model \
  --large-model openrouter/some-large-model
```

## Data

The current benchmark is:

```text
data/synthetic_streams.jsonl
```

Each line is one task with:

- `stream`: 3-7 timestamped messages.
- `question`: the final question to answer.
- `gold_answer`: the short answer used for scoring.

Example task shape:

```json
{
  "id": "syn_001",
  "task_type": "state_tracking",
  "stream": [
    {"t": 1, "source": "lab email", "text": "The meeting is Friday."},
    {"t": 2, "source": "Slack", "text": "Friday no longer works."},
    {"t": 3, "source": "calendar", "text": "The meeting moved to Monday."}
  ],
  "question": "What day is the meeting now?",
  "gold_answer": "Monday"
}
```

## Metrics

The evaluator reports:

- exact match
- contains gold
- score
- token usage when available
- latency
- number of model calls

Scoring:

- `1.0` for normalized exact match.
- `0.75` if the normalized gold answer appears inside the normalized prediction.
- `0.0` otherwise.

Token counts come from LiteLLM/provider usage fields when available. If a provider does not return token usage, token fields are saved as `null`.

## Useful Commands

Run only the large baseline:

```bash
python -m src.run_experiment --mode baseline --limit 5
```

Run only the small swarm:

```bash
python -m src.run_experiment --mode swarm --limit 5
```

Run both methods:

```bash
python -m src.run_experiment --mode both --limit 5
```

## Benchmark Plan

Current:

- Synthetic persistent-stream benchmark with gold answers.

Next:

- Add GAIA Level 1 loader.
- Add LLM-as-judge evaluation.
- Add real streams such as email, Slack, articles, and lab updates.
- Plug into Professor Chandy's distributed message-passing agent system.

GAIA is not implemented yet. The current code includes a loader stub so a future public benchmark can be added without rewriting the experiment pipeline.

## Current Limitations

- synthetic benchmark only
- simple automatic scoring
- no live streams yet
- no budget optimization yet
- no integration with Professor Chandy's message-passing system yet
