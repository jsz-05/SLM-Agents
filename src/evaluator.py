from __future__ import annotations

import re
import string
from typing import Any

from src.schemas import MethodResult, StreamTask


def normalize_text(s: str) -> str:
    lowered = s.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punctuation).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def contains_gold(pred: str, gold: str) -> bool:
    normalized_pred = normalize_text(pred)
    normalized_gold = normalize_text(gold)
    return bool(normalized_gold) and normalized_gold in normalized_pred


def score_answer(pred: str, gold: str) -> float:
    if exact_match(pred, gold):
        return 1.0
    if contains_gold(pred, gold):
        return 0.75
    return 0.0


def evaluate_task_result(task: StreamTask, method_result: MethodResult) -> dict[str, Any]:
    return {
        "gold_answer": task.gold_answer,
        "prediction": method_result.answer,
        "exact_match": exact_match(method_result.answer, task.gold_answer),
        "contains_gold": contains_gold(method_result.answer, task.gold_answer),
        "score": score_answer(method_result.answer, task.gold_answer),
    }


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sum_optional(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _method_entries(records: list[dict], method: str) -> list[dict]:
    entries = []
    for record in records:
        method_record = record.get(method)
        if isinstance(method_record, dict) and "result" in method_record and "metrics" in method_record:
            entries.append(method_record)
    return entries


def summarize_results(records: list[dict]) -> dict[str, Any]:
    baseline_entries = _method_entries(records, "baseline")
    swarm_entries = _method_entries(records, "swarm")

    def scores(entries: list[dict]) -> list[float]:
        return [float(entry["metrics"]["score"]) for entry in entries]

    def exact_count(entries: list[dict]) -> int:
        return sum(1 for entry in entries if entry["metrics"].get("exact_match"))

    def contains_count(entries: list[dict]) -> int:
        return sum(1 for entry in entries if entry["metrics"].get("contains_gold"))

    def latencies(entries: list[dict]) -> list[float]:
        return [float(entry["result"]["latency_seconds"]) for entry in entries]

    def tokens(entries: list[dict]) -> int | None:
        return _sum_optional([entry["result"].get("total_tokens") for entry in entries])

    def calls(entries: list[dict]) -> int:
        return sum(int(entry["result"].get("model_calls", 0)) for entry in entries)

    return {
        "n": len(records),
        "baseline_avg_score": _avg(scores(baseline_entries)),
        "swarm_avg_score": _avg(scores(swarm_entries)),
        "baseline_exact_count": exact_count(baseline_entries),
        "swarm_exact_count": exact_count(swarm_entries),
        "baseline_contains_count": contains_count(baseline_entries),
        "swarm_contains_count": contains_count(swarm_entries),
        "baseline_avg_latency": _avg(latencies(baseline_entries)),
        "swarm_avg_latency": _avg(latencies(swarm_entries)),
        "baseline_total_tokens": tokens(baseline_entries),
        "swarm_total_tokens": tokens(swarm_entries),
        "baseline_total_calls": calls(baseline_entries),
        "swarm_total_calls": calls(swarm_entries),
    }
