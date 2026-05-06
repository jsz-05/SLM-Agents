from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.dataset import load_stream_tasks
from src.evaluator import evaluate_task_result, summarize_results
from src.model_names import normalize_model_name
from src.schemas import MethodResult, StreamTask


DEFAULT_SMALL_MODEL = "ollama/gemma3:4b"
DEFAULT_LARGE_MODEL = "ollama/gemma3:12b"


def _model_to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return repr(obj)


def _safe_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "run"


def _default_output_path(run_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        return f"results/{timestamp}_{_safe_run_name(run_name)}.json"
    return f"results/run_{timestamp}.json"


def _run_method_safely(func, task: StreamTask) -> dict[str, Any]:
    try:
        result: MethodResult = func(task)
        metrics = evaluate_task_result(task, result)
        return {
            "result": _model_to_dict(result),
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
        }


def _print_summary(summary: dict[str, Any], output: str) -> None:
    print("\nSummary")
    print("=======")
    print(f"Records: {summary['n']}")
    print("")
    print(f"{'method':<10} {'avg_score':>10} {'exact':>8} {'contains':>10} {'avg_latency':>12} {'tokens':>10} {'calls':>8}")
    print("-" * 78)
    for method in ("baseline", "swarm"):
        avg_score = summary.get(f"{method}_avg_score")
        avg_latency = summary.get(f"{method}_avg_latency")
        tokens = summary.get(f"{method}_total_tokens")
        print(
            f"{method:<10} "
            f"{_fmt(avg_score):>10} "
            f"{summary.get(f'{method}_exact_count', 0):>8} "
            f"{summary.get(f'{method}_contains_count', 0):>10} "
            f"{_fmt(avg_latency):>12} "
            f"{_fmt(tokens):>10} "
            f"{summary.get(f'{method}_total_calls', 0):>8}"
        )
    print(f"\nSaved: {output}")


def _print_errors(records: list[dict[str, Any]]) -> None:
    errors: list[tuple[str, str, str]] = []
    for record in records:
        task_id = record.get("task", {}).get("id", "unknown")
        for method in ("baseline", "swarm"):
            method_record = record.get(method)
            if isinstance(method_record, dict) and "error" in method_record:
                message = method_record["error"].get("message", "")
                errors.append((task_id, method, message))

    if not errors:
        return

    print("\nErrors")
    print("======")
    for task_id, method, message in errors[:3]:
        print(f"{task_id} / {method}: {message}")
    if len(errors) > 3:
        print(f"... plus {len(errors) - 3} more error(s). See the saved JSON for details.")


def _ollama_model_name(model: str) -> str | None:
    if not model.startswith("ollama/"):
        return None
    return model.removeprefix("ollama/")


def _stop_ollama_model(model: str) -> None:
    ollama_model = _ollama_model_name(model)
    if not ollama_model:
        return
    print(f"\nStopping Ollama model: {ollama_model}")
    try:
        subprocess.run(
            ["ollama", "stop", ollama_model],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("Could not stop model because the ollama command was not found.")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare a large LLM baseline with a small-model agent swarm on stream tasks."
    )
    parser.add_argument("--data", default="data/synthetic_streams.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["baseline", "swarm", "both"], default="both")
    parser.add_argument(
        "--large-model",
        default=normalize_model_name(os.getenv("LARGE_MODEL", DEFAULT_LARGE_MODEL)),
    )
    parser.add_argument(
        "--small-model",
        default=normalize_model_name(os.getenv("SMALL_MODEL", DEFAULT_SMALL_MODEL)),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional readable label for the output filename, e.g. sanity_1task.",
    )
    parser.add_argument(
        "--execution-order",
        choices=["task", "method"],
        default="task",
        help="Run both methods per task, or run all baseline tasks before all swarm tasks.",
    )
    parser.add_argument(
        "--stop-ollama-between-methods",
        action="store_true",
        help="Unload Ollama models between baseline and swarm runs to reduce memory pressure.",
    )
    parser.add_argument(
        "--swarm-agents",
        type=int,
        choices=[3, 5],
        default=5,
        help="Use the full 5-agent swarm or a compact 3-agent swarm.",
    )
    parser.add_argument(
        "--swarm-architecture",
        choices=["pipeline", "adaptive"],
        default="pipeline",
        help="Use the original pipeline swarm or the task-adaptive specialist swarm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.large_model = normalize_model_name(args.large_model)
    args.small_model = normalize_model_name(args.small_model)

    if args.mode in ("baseline", "both"):
        from src.baseline_large import run_large_baseline
    if args.mode in ("swarm", "both"):
        if args.swarm_architecture == "adaptive":
            from src.swarm_adaptive import run_adaptive_swarm
        else:
            from src.swarm_small import run_small_swarm

    output = args.output or _default_output_path(args.run_name)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    tasks = load_stream_tasks(args.data, limit=args.limit)
    records: list[dict[str, Any]] = [{"task": _model_to_dict(task)} for task in tasks]

    def run_baseline(record: dict[str, Any], task: StreamTask) -> None:
        record["baseline"] = _run_method_safely(
            lambda t: run_large_baseline(
                t,
                large_model=args.large_model,
                temperature=args.temperature,
            ),
            task,
        )

    def run_swarm(record: dict[str, Any], task: StreamTask) -> None:
        if args.swarm_architecture == "adaptive":
            record["swarm"] = _run_method_safely(
                lambda t: run_adaptive_swarm(
                    t,
                    small_model=args.small_model,
                    temperature=args.temperature,
                ),
                task,
            )
            return

        record["swarm"] = _run_method_safely(
            lambda t: run_small_swarm(
                t,
                small_model=args.small_model,
                temperature=args.temperature,
                swarm_agents=args.swarm_agents,
            ),
            task,
        )

    if args.execution_order == "method" and args.mode == "both":
        for record, task in tqdm(
            zip(records, tasks),
            total=len(tasks),
            desc="Running baseline",
        ):
            run_baseline(record, task)

        if args.stop_ollama_between_methods:
            _stop_ollama_model(args.large_model)

        for record, task in tqdm(
            zip(records, tasks),
            total=len(tasks),
            desc="Running swarm",
        ):
            run_swarm(record, task)

        if args.stop_ollama_between_methods:
            _stop_ollama_model(args.small_model)
    else:
        for record, task in tqdm(zip(records, tasks), total=len(tasks), desc="Running tasks"):
            if args.mode in ("baseline", "both"):
                run_baseline(record, task)

            if (
                args.mode == "both"
                and args.stop_ollama_between_methods
                and args.execution_order == "task"
            ):
                _stop_ollama_model(args.large_model)

            if args.mode in ("swarm", "both"):
                run_swarm(record, task)

            if (
                args.mode == "both"
                and args.stop_ollama_between_methods
                and args.execution_order == "task"
            ):
                _stop_ollama_model(args.small_model)

        if args.stop_ollama_between_methods and args.mode == "baseline":
            _stop_ollama_model(args.large_model)
        if args.stop_ollama_between_methods and args.mode == "swarm":
            _stop_ollama_model(args.small_model)

    summary = summarize_results(records)
    payload = {
        "config": {
            "data": args.data,
            "limit": args.limit,
            "mode": args.mode,
            "large_model": args.large_model,
            "small_model": args.small_model,
            "temperature": args.temperature,
            "run_name": args.run_name,
            "execution_order": args.execution_order,
            "stop_ollama_between_methods": args.stop_ollama_between_methods,
            "swarm_agents": args.swarm_agents,
            "swarm_architecture": args.swarm_architecture,
        },
        "summary": summary,
        "records": records,
        "todos": [
            "Add a GAIA Level 1 loader that emits the same StreamTask-like interface or a compatible benchmark task schema.",
            "Add tool-use and browsing adapters for GAIA without changing the baseline/swarm result and evaluator contract.",
        ],
    }

    with Path(output).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, default=_json_default)

    _print_summary(summary, output)
    _print_errors(records)


if __name__ == "__main__":
    main()
