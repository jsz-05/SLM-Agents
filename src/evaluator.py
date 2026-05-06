from __future__ import annotations

import re
import string
from typing import Any

from src.postprocess import clean_answer
from src.schemas import MethodResult, StreamTask


def normalize_text(s: str) -> str:
    lowered = clean_answer(s).lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punctuation).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def contains_gold(pred: str, gold: str) -> bool:
    normalized_pred = normalize_text(pred)
    normalized_gold = normalize_text(gold)
    return bool(normalized_gold) and normalized_gold in normalized_pred


def gold_contains_prediction(pred: str, gold: str) -> bool:
    normalized_pred = normalize_text(pred)
    normalized_gold = normalize_text(gold)
    return bool(normalized_pred) and normalized_pred in normalized_gold


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = sum(min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _labels(task: StreamTask) -> list[str]:
    labels = [task.gold_answer, *task.aliases]
    seen = set()
    deduped = []
    for label in labels:
        normalized = normalize_text(label)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(label)
    return deduped


def exact_match_any(pred: str, labels: list[str]) -> bool:
    return any(exact_match(pred, label) for label in labels)


def contains_match_any(pred: str, labels: list[str]) -> bool:
    return any(contains_gold(pred, label) or gold_contains_prediction(pred, label) for label in labels)


def best_token_f1(pred: str, labels: list[str]) -> float:
    if not labels:
        return 0.0
    return max(token_f1(pred, label) for label in labels)


def score_answer(pred: str, gold: str, aliases: list[str] | None = None) -> float:
    labels = [gold, *(aliases or [])]
    if exact_match_any(pred, labels):
        return 1.0
    if contains_match_any(pred, labels):
        return 0.8
    if best_token_f1(pred, labels) >= 0.5:
        return 0.5
    return 0.0


def evaluate_task_result(task: StreamTask, method_result: MethodResult) -> dict[str, Any]:
    labels = _labels(task)
    prediction = clean_answer(method_result.answer)
    best_f1 = best_token_f1(prediction, labels)
    alias_match = bool(task.aliases) and (
        exact_match_any(prediction, task.aliases)
        or contains_match_any(prediction, task.aliases)
    )
    return {
        "gold_answer": task.gold_answer,
        "aliases": task.aliases,
        "prediction": prediction,
        "exact_match": exact_match_any(prediction, labels),
        "alias_match": alias_match,
        "contains_gold": any(contains_gold(prediction, label) for label in labels),
        "gold_contains_prediction": any(gold_contains_prediction(prediction, label) for label in labels),
        "token_f1": best_f1,
        "score": score_answer(prediction, task.gold_answer, task.aliases),
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
        return sum(
            1
            for entry in entries
            if entry["metrics"].get("contains_gold")
            or entry["metrics"].get("gold_contains_prediction")
        )

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
