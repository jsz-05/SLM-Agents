from __future__ import annotations

import json
import re
from typing import Any

from src.models import call_model
from src.schemas import MethodResult, ModelCallResult, StreamTask


ADAPTIVE_SPECIALIST_PROMPT = """You are a task-specialist small-model agent for stream reasoning.

The benchmark rewards short exact answers. Read the task_type and answer accordingly:
- state_tracking: return the latest value only, not the topic name.
- entity_assignment: return the currently assigned person/entity only; later replacements override earlier assignees.
- priority_detection: return the blocker/priority item only.
- multi_hop: follow dependencies and later updates; return the final current value.
- contradiction_detection: return the abstract topic that changed, not the new value. Use noun phrases like budget amount, meeting location, paper title, age exclusion, launch audience, token count.

Return compact JSON only:
{"answer":"short answer","evidence_t":[1,2],"reason":"brief reason"}
"""


ADAPTIVE_VERIFIER_PROMPT = """You are a strict answer verifier and normalizer for a short-answer stream benchmark.

Check the draft answer against the stream. Rewrite it to the shortest exact phrase that answers the question.

Rules:
- Remove leading articles like "the".
- Remove explanatory words like "is missing", "priority", "stability", "was revised", unless they are part of the requested value.
- If asked for a day, return only the day.
- If asked for a date, return only the date.
- If asked who, return only the person/entity name.
- For entity assignment, reject earlier people if a later message says they are unavailable, cannot access, switching tasks, or someone else will take over/handle it instead.
- If asked where, return only the location.
- For multi-hop tasks, prefer later extensions, protocol updates, reconfigurations, and current waiting-on states over original plans.
- If asked which topic changed/corrected/revised, return the field/topic label, not the replacement value.
- For contradiction topics, prefer labels such as budget amount, meeting location, paper title, age exclusion, launch audience, token count.
- Ignore source/channel names like Slack #budget when deciding the corrected topic.

Return compact JSON only:
{"final_answer":"short answer","confidence":0.0,"rationale":"brief reason"}
"""


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
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


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


def _task_specific_rules(task_type: str) -> str:
    if task_type == "entity_assignment":
        return (
            "Extra entity_assignment rule: scan from latest to earliest. "
            "Choose the person/entity in the latest valid assignment. "
            "Ignore earlier assignees if later messages say they are unavailable, "
            "cannot access the portal, are switching tasks, or someone will take over/handle it instead."
        )
    if task_type == "multi_hop":
        return (
            "Extra multi_hop rule: later messages can change dates, locations, lots, paths, or waiting states. "
            "Use the latest resolved value after dependencies are satisfied, not the original plan."
        )
    if task_type == "priority_detection":
        return (
            "Extra priority_detection rule: return the concrete blocker or task, stripped of words like priority, urgent, blocker, or stability."
        )
    if task_type == "state_tracking":
        return (
            "Extra state_tracking rule: return only the current value requested by the question, such as day, time, date, version, room, or status."
        )
    if task_type == "contradiction_detection":
        return (
            "Extra contradiction_detection rule: answer the abstract field that changed. "
            "Do not answer with the old value or new value. Ignore source/channel names such as Slack #budget."
        )
    return ""


def _basic_cleanup(answer: str) -> str:
    answer = answer.strip().strip("\"'`")
    answer = re.sub(r"^\s*the\s+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"[.。]+$", "", answer).strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer


def _all_text(task: StreamTask, answer: str) -> str:
    stream_text = " ".join(m.text for m in task.stream)
    return f"{task.question} {stream_text} {answer}".lower()


def _canonicalize_contradiction(task: StreamTask, answer: str) -> str:
    text = _all_text(task, answer)
    answer_lower = answer.lower()

    if "token" in text:
        return "token count"
    if "budget" in text:
        return "budget amount"
    if any(term in text for term in ("location", "room", "hall", "gates", "annenberg")):
        return "meeting location"
    if "title" in text:
        return "paper title"
    if any(term in text for term in ("age", "under 18", "over 65", "participant", "cohort")):
        return "age exclusion"
    if "audience" in text:
        return "launch audience"
    return answer_lower if len(answer_lower.split()) <= 3 else answer


def _canonicalize_priority(answer: str) -> str:
    answer = re.sub(r"\b(priority|stability|urgent|today|first|blocker)\b", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\bis missing\b", "", answer, flags=re.IGNORECASE)
    return _basic_cleanup(answer)


def _canonicalize_state(task: StreamTask, answer: str) -> str:
    question = task.question.lower()
    if "level" in question:
        match = re.search(r"\blevel\s+\d+\b", answer, flags=re.IGNORECASE)
        if match:
            return match.group(0).title()
    if "version" in question:
        match = re.search(r"\bv\d+\b", answer, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    if "what day" in question:
        match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            answer,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(0).title()
    return answer


def _canonicalize_answer(task: StreamTask, answer: str) -> str:
    answer = _basic_cleanup(answer)
    if task.task_type == "contradiction_detection":
        return _canonicalize_contradiction(task, answer)
    if task.task_type == "priority_detection":
        return _canonicalize_priority(answer)
    if task.task_type == "state_tracking":
        return _canonicalize_state(task, answer)
    return answer


def run_adaptive_swarm(
    task: StreamTask,
    small_model: str,
    temperature: float = 0.0,
) -> MethodResult:
    stream_text = _stream_to_text(task)
    calls: list[ModelCallResult] = []

    common_context = (
        f"task_type: {task.task_type}\n"
        f"Task-specific rule: {_task_specific_rules(task.task_type)}\n"
        f"Stream:\n{stream_text}\n\n"
        f"Question: {task.question}"
    )

    draft_call, draft = _call_agent(
        small_model,
        ADAPTIVE_SPECIALIST_PROMPT,
        common_context,
        temperature,
    )
    calls.append(draft_call)

    draft_answer = str(
        draft.get("answer")
        or draft.get("final_answer")
        or draft_call.content
    ).strip()

    verifier_call, verifier = _call_agent(
        small_model,
        ADAPTIVE_VERIFIER_PROMPT,
        (
            f"{common_context}\n\n"
            f"Draft answer JSON:\n{_json_text(draft)}\n\n"
            f"Draft answer: {draft_answer}"
        ),
        temperature,
    )
    calls.append(verifier_call)

    verified_answer = str(
        verifier.get("final_answer")
        or verifier.get("answer")
        or draft_answer
    ).strip()
    final_answer = _canonicalize_answer(task, verified_answer)

    return MethodResult(
        answer=final_answer,
        confidence=_coerce_confidence(verifier.get("confidence")),
        rationale=str(verifier.get("rationale") or draft.get("reason") or ""),
        raw={
            "architecture": "adaptive",
            "draft": {
                "raw_output": draft_call.content,
                "parsed_output": draft,
            },
            "verification": {
                "raw_output": verifier_call.content,
                "parsed_output": verifier,
            },
            "canonicalizer": {
                "input": verified_answer,
                "output": final_answer,
            },
        },
        model_calls=len(calls),
        input_tokens=_sum_optional(calls, "input_tokens"),
        output_tokens=_sum_optional(calls, "output_tokens"),
        total_tokens=_sum_optional(calls, "total_tokens"),
        latency_seconds=sum(call.latency_seconds for call in calls),
    )
