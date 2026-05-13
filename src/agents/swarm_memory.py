"""Method D memory/retrieval swarm.

This architecture keeps Experiment 1's same-small-model condition, but changes
the small-agent network from "everyone reads the whole stream" to:
- deterministic memory cards
- small-model evidence retrieval
- small-model answer generation
- small-model verification
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

from src.models import call_model
from src.postprocess import clean_answer
from src.prompts import (
    MEMORY_ANSWER_PROMPT,
    MEMORY_RETRIEVER_PROMPT,
    MEMORY_VERIFIER_PROMPT,
)
from src.schemas import MethodResult, ModelCallResult, StreamMessage, StreamTask


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


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


def _split_message(message: StreamMessage, max_chars: int) -> list[str]:
    text = message.text.strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars].strip())
            continue

        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _build_memory_cards(task: StreamTask, max_chars_per_card: int = 900) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for message in task.stream:
        chunks = _split_message(message, max_chars=max_chars_per_card)
        for chunk_index, chunk in enumerate(chunks, start=1):
            cards.append(
                {
                    "id": f"m{len(cards) + 1:03d}",
                    "t": message.t,
                    "chunk": chunk_index,
                    "source": message.source,
                    "text": chunk,
                }
            )
    return cards


def _tokens(text: str) -> list[str]:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [token for token in lowered.split() if len(token) > 2 and token not in STOPWORDS]


def _rank_cards(question: str, cards: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    if len(cards) <= max_candidates:
        return cards

    question_tokens = set(_tokens(question))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    max_t = max((int(card.get("t", 0)) for card in cards), default=0)
    for index, card in enumerate(cards):
        card_text = f"{card.get('source', '')} {card.get('text', '')}"
        card_tokens = _tokens(card_text)
        token_set = set(card_tokens)
        overlap = len(question_tokens & token_set)

        phrase_bonus = 0
        normalized_card = " ".join(card_tokens)
        for token in question_tokens:
            if token in normalized_card:
                phrase_bonus += 0.05

        # A tiny recency nudge helps persistent streams where later updates
        # supersede earlier messages, without hard-coding any task type.
        recency = int(card.get("t", 0)) / max_t if max_t else 0
        score = overlap + phrase_bonus + (0.15 * recency)
        scored.append((score, -index, card))

    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [card for score, _index, card in scored[:max_candidates] if score > 0]
    if not selected:
        selected = [card for _score, _index, card in scored[:max_candidates]]
    selected.sort(key=lambda card: (int(card.get("t", 0)), str(card.get("id", ""))))
    return selected


def _compact_cards(cards: list[dict[str, Any]], max_text_chars: int = 700) -> list[dict[str, Any]]:
    compact = []
    for card in cards:
        text = str(card.get("text", ""))
        if len(text) > max_text_chars:
            text = text[:max_text_chars].rstrip() + " ..."
        compact.append(
            {
                "id": card.get("id"),
                "t": card.get("t"),
                "source": card.get("source"),
                "text": text,
            }
        )
    return compact


def _selected_ids(parsed: dict[str, Any]) -> list[str]:
    value = parsed.get("selected_ids") or parsed.get("ids") or parsed.get("relevant_ids")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return re.findall(r"m\d{3}", value)
    return []


def _cards_by_id(cards: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    lookup = {str(card.get("id")): card for card in cards}
    selected = [lookup[card_id] for card_id in ids if card_id in lookup]
    return selected


def _evidence_text(cards: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{card['id']} t={card['t']} {card['source']}]\n{card['text']}" for card in cards
    )


def _cards_text_chars(cards: list[dict[str, Any]]) -> int:
    return sum(len(str(card.get("text", ""))) for card in cards)


def _merge_cards(primary: list[dict[str, Any]], fallback: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in [*primary, *fallback]:
        card_id = str(card.get("id"))
        if card_id in seen:
            continue
        seen.add(card_id)
        merged.append(card)
        if len(merged) >= limit:
            break
    merged.sort(key=lambda card: (int(card.get("t", 0)), str(card.get("id", ""))))
    return merged


def run_memory_swarm(
    task: StreamTask,
    small_model: str,
    temperature: float = 0.0,
    max_candidates: int = 14,
    max_selected: int = 6,
    evidence_budget_chars: int = 5000,
) -> MethodResult:
    memory_cards = _build_memory_cards(task)
    candidate_cards = _rank_cards(task.question, memory_cards, max_candidates=max_candidates)
    calls: list[ModelCallResult] = []

    retrieval_call, retrieval = _call_agent(
        small_model,
        MEMORY_RETRIEVER_PROMPT,
        (
            f"Question:\n{task.question}\n\n"
            f"Candidate memory cards:\n{_json_text(_compact_cards(candidate_cards))}"
        ),
        temperature,
    )
    calls.append(retrieval_call)

    retrieved_cards = _cards_by_id(candidate_cards, _selected_ids(retrieval))
    selection_strategy = "retrieved"
    if _cards_text_chars(memory_cards) <= evidence_budget_chars:
        selected = memory_cards
        selection_strategy = "all_cards_fit_budget"
    else:
        if not retrieved_cards:
            retrieved_cards = candidate_cards[:max_selected]
            selection_strategy = "fallback_candidates"
        selected = _merge_cards(retrieved_cards, candidate_cards, max_selected)
    evidence = _evidence_text(selected)

    answer_call, answer_agent = _call_agent(
        small_model,
        MEMORY_ANSWER_PROMPT,
        (
            f"Question:\n{task.question}\n\n"
            f"Selected memory evidence:\n{evidence}"
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
        MEMORY_VERIFIER_PROMPT,
        (
            f"Question:\n{task.question}\n\n"
            f"Selected memory evidence:\n{evidence}\n\n"
            f"Proposed answer: {proposed_answer}"
        ),
        temperature,
    )
    calls.append(verification_call)

    final_answer = clean_answer(str(
        verification.get("answer")
        or verification.get("final_answer")
        or proposed_answer
    ))
    rationale = verification.get("rationale") or answer_agent.get("rationale")
    if rationale is not None:
        rationale = str(rationale)

    return MethodResult(
        answer=final_answer,
        confidence=_coerce_confidence(
            verification.get("confidence", answer_agent.get("confidence"))
        ),
        rationale=rationale,
        raw={
            "architecture": "memory",
            "memory_cards_count": len(memory_cards),
            "candidate_cards": _compact_cards(candidate_cards),
            "selection_strategy": selection_strategy,
            "retrieval": {
                "raw_output": retrieval_call.content,
                "parsed_output": retrieval,
            },
            "selected_cards": selected,
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
