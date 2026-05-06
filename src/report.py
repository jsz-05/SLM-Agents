from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _cell(value: Any) -> str:
    text = _fmt(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _method(record: dict[str, Any], method: str) -> dict[str, Any]:
    method_record = record.get(method)
    return method_record if isinstance(method_record, dict) else {}


def _result_value(record: dict[str, Any], method: str, key: str) -> Any:
    return _method(record, method).get("result", {}).get(key)


def _metric_value(record: dict[str, Any], method: str, key: str) -> Any:
    return _method(record, method).get("metrics", {}).get(key)


def _score(record: dict[str, Any], method: str) -> float | None:
    value = _metric_value(record, method, "score")
    return float(value) if value is not None else None


def write_markdown_summary(
    output_path: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    markdown_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_summary.md"))
    lines: list[str] = []

    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key, value in config.items():
        lines.append(f"| {_cell(key)} | {_cell(value)} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| method | avg_score | exact | contains | avg_latency | tokens | calls |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in ("baseline", "swarm"):
        lines.append(
            f"| {method} | "
            f"{_cell(summary.get(f'{method}_avg_score'))} | "
            f"{_cell(summary.get(f'{method}_exact_count'))} | "
            f"{_cell(summary.get(f'{method}_contains_count'))} | "
            f"{_cell(summary.get(f'{method}_avg_latency'))} | "
            f"{_cell(summary.get(f'{method}_total_tokens'))} | "
            f"{_cell(summary.get(f'{method}_total_calls'))} |"
        )

    lines.append("")
    lines.append("## Per-Task Comparison")
    lines.append("")
    lines.append(
        "| id | task_type | gold | baseline_answer | baseline_score | "
        "swarm_answer | swarm_score | baseline_tokens | swarm_tokens |"
    )
    lines.append("|---|---|---|---|---:|---|---:|---:|---:|")
    for record in records:
        task = record.get("task", {})
        lines.append(
            f"| {_cell(task.get('id'))} | "
            f"{_cell(task.get('task_type'))} | "
            f"{_cell(task.get('gold_answer'))} | "
            f"{_cell(_metric_value(record, 'baseline', 'prediction'))} | "
            f"{_cell(_score(record, 'baseline'))} | "
            f"{_cell(_metric_value(record, 'swarm', 'prediction'))} | "
            f"{_cell(_score(record, 'swarm'))} | "
            f"{_cell(_result_value(record, 'baseline', 'total_tokens'))} | "
            f"{_cell(_result_value(record, 'swarm', 'total_tokens'))} |"
        )

    lines.append("")
    lines.append("## Failure Cases")
    lines.append("")
    lines.append("| id | method | task_type | gold | prediction | score |")
    lines.append("|---|---|---|---|---|---:|")
    any_failure = False
    for record in records:
        task = record.get("task", {})
        for method in ("baseline", "swarm"):
            score = _score(record, method)
            if score is not None and score < 1.0:
                any_failure = True
                lines.append(
                    f"| {_cell(task.get('id'))} | "
                    f"{method} | "
                    f"{_cell(task.get('task_type'))} | "
                    f"{_cell(task.get('gold_answer'))} | "
                    f"{_cell(_metric_value(record, method, 'prediction'))} | "
                    f"{_cell(score)} |"
                )
    if not any_failure:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
