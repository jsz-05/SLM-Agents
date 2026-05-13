from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _answer_to_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (int, float, bool)):
        return str(answer)
    if isinstance(answer, list):
        return "; ".join(_answer_to_text(item) for item in answer)
    return json.dumps(answer, ensure_ascii=True, sort_keys=True)


def _turns_to_text(session: list[dict[str, Any]], user_only: bool) -> str:
    lines = []
    for turn in session:
        role = str(turn.get("role", "unknown"))
        if user_only and role != "user":
            continue
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _sort_key(item: tuple[str, str, list[dict[str, Any]]]) -> tuple[str, str]:
    session_id, date, _session = item
    return str(date), str(session_id)


def _convert_record(record: dict[str, Any], sort_by_date: bool, user_only: bool) -> dict[str, Any]:
    question_id = str(record.get("question_id", "unknown"))
    question_type = str(record.get("question_type", "unknown"))
    question_date = str(record.get("question_date", "")).strip()

    session_ids = record.get("haystack_session_ids") or []
    dates = record.get("haystack_dates") or []
    sessions = record.get("haystack_sessions") or []
    packed = list(zip(session_ids, dates, sessions))
    if sort_by_date:
        packed = sorted(packed, key=_sort_key)

    stream = []
    for idx, (session_id, date, session) in enumerate(packed, start=1):
        stream.append(
            {
                "t": idx,
                "source": f"LongMemEval session {session_id} @ {date}",
                "text": _turns_to_text(session, user_only=user_only),
            }
        )

    task_type = "abstention" if question_id.endswith("_abs") else question_type
    question = str(record.get("question", "")).strip()
    if question_date:
        question = f"{question}\nQuestion date: {question_date}"

    return {
        "id": f"lme_{question_id}",
        "task_type": task_type,
        "stream": stream,
        "question": question,
        "gold_answer": _answer_to_text(record.get("answer", "")),
        "aliases": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LongMemEval JSON into the StreamTask JSONL format used by this project."
    )
    parser.add_argument("--input", required=True, help="Path to a LongMemEval JSON file.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to convert.")
    parser.add_argument(
        "--sort-by-date",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort sessions chronologically. Useful because oracle sessions are not guaranteed to be sorted.",
    )
    parser.add_argument(
        "--user-only",
        action="store_true",
        help="Keep only user turns from each session. This reduces local context size and mirrors a LongMemEval run option.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if args.limit is not None:
        records = records[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            converted = _convert_record(record, sort_by_date=args.sort_by_date, user_only=args.user_only)
            f.write(json.dumps(converted, ensure_ascii=True, separators=(",", ":")) + "\n")

    print(f"Converted {len(records)} records")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
