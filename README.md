# Evaluating Small-Model Agent Networks Against Large LLMs on Persistent Information Streams

## Research Question

Can a heterogeneous network of small LLM agents match or outperform a single larger LLM on stream reasoning tasks?

This prototype compares:

- Baseline A: one larger LLM acting as a monolithic agent.
- Method D: a heterogeneous network of smaller role-specialized agents.

The first benchmark is synthetic and controlled: each task is a stream of messages where later messages may update, correct, or contradict earlier ones. The model must answer a final question about the current state, contradiction, priority, entity assignment, or multi-hop dependency.

## What Runs

The baseline sends the full stream and question to one model.

There are two swarm architectures.

Pipeline swarm (`--swarm-architecture pipeline`) uses the same small model in five roles:

- FactExtractorAgent
- StateTrackerAgent
- ContradictionDetectorAgent
- AnswerAgent
- VerifierAgent

You can also run a compact 3-agent swarm with `--swarm-agents 3`:

- FactExtractorAgent
- StateAndContradictionAgent
- AnswerAndVerifierAgent

Adaptive swarm (`--swarm-architecture adaptive`) uses:

- TaskSpecialistAgent
- StrictVerifierAndNormalizerAgent
- a deterministic short-answer canonicalizer

The adaptive swarm is the strongest current architecture on the synthetic benchmark. It uses task type metadata, small-model reasoning, and a symbolic final formatting layer. It does not use the gold answer.

Results are scored against the task's gold answer and saved to `results/run_TIMESTAMP.json` by default. Use `--run-name` to create readable filenames such as `results/20260505_235000_sanity_1task.json`.

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
python -m src.run_experiment --mode both --limit 5 --swarm-architecture adaptive
```

Equivalent explicit command:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --run-name adaptive_gemma4b_vs_gemma12b_5tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --swarm-architecture adaptive \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

Other local pairs you can try:

```bash
python -m src.run_experiment --mode both --limit 5 --run-name qwen_8b_vs_14b_5tasks --small-model ollama/qwen3:8b --large-model ollama/qwen3:14b
```

On a 24GB Mac, avoid `gemma3:27b` for routine runs. A safer comparison is either:

- same family: `gemma3:4b` compact swarm vs `gemma3:12b` baseline
- roughly 14B baseline: `gemma3:4b` compact swarm vs `qwen3:14b` baseline

Safe 3-agent run against Qwen 14B:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 1 \
  --run-name sanity_gemma_4b_3agent_swarm_vs_qwen_14b_1task \
  --execution-order method \
  --stop-ollama-between-methods \
  --swarm-agents 3 \
  --small-model ollama/gemma3:4b \
  --large-model ollama/qwen3:14b
```

`gemma3:27b` can exceed memory on 24GB machines, especially in `--mode both`. Prefer `gemma3:12b` or `qwen3:14b` for routine runs on a 24GB Mac.

If you do test `gemma3:27b`, use method-ordered execution and explicitly unload Ollama between methods:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 1 \
  --run-name sanity_lowmem_gemma_4b_swarm_vs_27b_1task \
  --execution-order method \
  --stop-ollama-between-methods \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:27b
```

This runs the 27B baseline first, stops `gemma3:27b`, then runs the 4B swarm. It is slower but much safer on 24GB machines.

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
python -m src.run_experiment --mode both --limit 1 --run-name openrouter_free_1task
```

For more reliable experiments, choose paid model IDs from OpenRouter and set them in `.env` or pass them as CLI flags:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 5 \
  --run-name openrouter_paid_5tasks \
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
python -m src.run_experiment --mode baseline --limit 5 --run-name baseline_5tasks
```

Run only the small swarm:

```bash
python -m src.run_experiment --mode swarm --limit 5 --run-name swarm_5tasks
```

Run only the adaptive swarm:

```bash
python -m src.run_experiment --mode swarm --limit 5 --swarm-architecture adaptive --run-name adaptive_swarm_5tasks
```

Run only the compact 3-agent swarm:

```bash
python -m src.run_experiment --mode swarm --limit 5 --swarm-agents 3 --run-name compact_swarm_3agents_5tasks
```

Run both methods:

```bash
python -m src.run_experiment --mode both --limit 5 --swarm-architecture adaptive --run-name both_adaptive_5tasks
```

Suggested first experiment for 24GB Macs:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 1 \
  --run-name sanity_adaptive_gemma4b_swarm_vs_gemma12b_1task \
  --execution-order method \
  --stop-ollama-between-methods \
  --swarm-architecture adaptive \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

Full synthetic run:

```bash
python -m src.run_experiment \
  --mode both \
  --run-name final_adaptive_gemma4b_swarm_vs_gemma12b_30tasks \
  --execution-order method \
  --stop-ollama-between-methods \
  --swarm-architecture adaptive \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:12b
```

Latest local result on the 30-task synthetic benchmark:

```text
Gemma 12B baseline:        avg_score 0.808, exact 19/30, avg_latency 3.233s, tokens 7,784
Adaptive Gemma 4B swarm:   avg_score 1.000, exact 30/30, avg_latency 2.942s, tokens 26,393
```

Saved result:

```text
results/20260506_013155_final_adaptive_v3_gemma4b_swarm_vs_gemma12b_30tasks.json
```

Only try `gemma3:27b` after closing memory-heavy apps, starting with `--limit 1`, and watching Activity Monitor memory pressure.

Low-memory 27B sanity check:

```bash
python -m src.run_experiment \
  --mode both \
  --limit 1 \
  --run-name sanity_lowmem_gemma_4b_swarm_vs_27b_1task \
  --execution-order method \
  --stop-ollama-between-methods \
  --small-model ollama/gemma3:4b \
  --large-model ollama/gemma3:27b
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
