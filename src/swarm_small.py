from __future__ import annotations

import json
import re
from typing import Any

from src.models import call_model
from src.prompts import (
    ANSWER_AGENT_PROMPT,
    CONTRADICTION_DETECTOR_PROMPT,
    FACT_EXTRACTOR_PROMPT,
    STATE_TRACKER_PROMPT,
    VERIFIER_PROMPT,
)
from src.schemas import MethodResult, ModelCallResult, StreamTask


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

    return {"raw_text": text.strip(), "parse_error": "Could not parse JSON."}


def _json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _sum_optional(results: list[ModelCallResult], attr: str) -> int | None:
    values = [getattr(r, attr) for r in results]
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _coerce_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _call_agent(
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
) -> tuple[ModelCallResult, dict[str, Any]]:
    call = call_model(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return call, _parse_json_loose(call.content)


def run_small_swarm(
    task: StreamTask,
    small_model: str,
    temperature: float = 0.0,
) -> MethodResult:
    stream_text = _stream_to_text(task)
    calls: list[ModelCallResult] = []

    facts_call, facts = _call_agent(
        small_model,
        FACT_EXTRACTOR_PROMPT,
        f"Stream:\n{stream_text}",
        temperature,
    )
    calls.append(facts_call)

    state_call, state = _call_agent(
        small_model,
        STATE_TRACKER_PROMPT,
        f"Extracted facts:\n{_json_text(facts)}",
        temperature,
    )
    calls.append(state_call)

    contradictions_call, contradictions = _call_agent(
        small_model,
        CONTRADICTION_DETECTOR_PROMPT,
        (
            f"Stream:\n{stream_text}\n\n"
            f"Extracted facts:\n{_json_text(facts)}\n\n"
            f"Tracked state:\n{_json_text(state)}"
        ),
        temperature,
    )
    calls.append(contradictions_call)

    answer_call, answer_agent = _call_agent(
        small_model,
        ANSWER_AGENT_PROMPT,
        (
            f"Question: {task.question}\n\n"
            f"Extracted facts:\n{_json_text(facts)}\n\n"
            f"Tracked state:\n{_json_text(state)}\n\n"
            f"Contradictions:\n{_json_text(contradictions)}"
        ),
        temperature,
    )
    calls.append(answer_call)

    proposed_answer = str(
        answer_agent.get("answer")
        or answer_agent.get("final_answer")
        or answer_call.content
    ).strip()

    verification_call, verification = _call_agent(
        small_model,
        VERIFIER_PROMPT,
        (
            f"Stream:\n{stream_text}\n\n"
            f"Question: {task.question}\n\n"
            f"Proposed answer: {proposed_answer}"
        ),
        temperature,
    )
    calls.append(verification_call)

    final_answer = str(
        verification.get("final_answer")
        or verification.get("answer")
        or proposed_answer
    ).strip()
    confidence = _coerce_confidence(
        verification.get("confidence", answer_agent.get("confidence"))
    )
    rationale = answer_agent.get("rationale")
    if rationale is not None:
        rationale = str(rationale)

    return MethodResult(
        answer=final_answer,
        confidence=confidence,
        rationale=rationale,
        raw={
            "facts": {
                "raw_output": facts_call.content,
                "parsed_output": facts,
            },
            "state": {
                "raw_output": state_call.content,
                "parsed_output": state,
            },
            "contradictions": {
                "raw_output": contradictions_call.content,
                "parsed_output": contradictions,
            },
            "answer_agent": {
                "raw_output": answer_call.content,
                "parsed_output": answer_agent,
            },
            "verification": {
                "raw_output": verification_call.content,
                "parsed_output": verification,
            },
        },
        model_calls=len(calls),
        input_tokens=_sum_optional(calls, "input_tokens"),
        output_tokens=_sum_optional(calls, "output_tokens"),
        total_tokens=_sum_optional(calls, "total_tokens"),
        latency_seconds=sum(call.latency_seconds for call in calls),
    )
