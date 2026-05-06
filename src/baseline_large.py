from __future__ import annotations

import json
import re
from typing import Any

from src.models import call_model
from src.prompts import LARGE_BASELINE_PROMPT
from src.schemas import MethodResult, StreamTask


def _stream_to_text(task: StreamTask) -> str:
    return "\n".join(f"[t={m.t}] {m.source}: {m.text}" for m in task.stream)


def _parse_json_loose(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            pass

    return {"answer": text.strip(), "parse_error": "Could not parse JSON."}


def _coerce_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def run_large_baseline(
    task: StreamTask,
    large_model: str,
    temperature: float = 0.0,
) -> MethodResult:
    user_content = (
        f"Stream:\n{_stream_to_text(task)}\n\n"
        f"Question: {task.question}\n"
    )
    call = call_model(
        model=large_model,
        messages=[
            {"role": "system", "content": LARGE_BASELINE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    parsed = _parse_json_loose(call.content)

    answer = str(parsed.get("answer") or parsed.get("final_answer") or call.content).strip()
    rationale = parsed.get("rationale")
    if rationale is not None:
        rationale = str(rationale)

    return MethodResult(
        answer=answer,
        confidence=_coerce_confidence(parsed.get("confidence")),
        rationale=rationale,
        raw={
            "raw_output": call.content,
            "parsed_output": parsed,
            "model_response": call.raw_response,
        },
        model_calls=1,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        total_tokens=call.total_tokens,
        latency_seconds=call.latency_seconds,
    )
