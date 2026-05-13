"""Method D memory/retrieval swarm V2.

V2 keeps the same-small-model condition, but changes the memory architecture
for long-memory QA:
- deterministic memory cards and temporal hints
- TemporalEvidenceAgent extracts the relevant event ledger
- TemporalReasonerAgent answers from the ledger/evidence
- TemporalVerifierAgent checks common temporal failure modes

The deterministic layer does not use gold labels. It only packages source
dates, question terms, and date-like snippets so the small agents spend less
capacity searching and more capacity reasoning.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any

from src.models import call_model
from src.postprocess import clean_answer
from src.prompts import (
    MEMORY_V2_EVENT_EXTRACTOR_PROMPT,
    MEMORY_V2_REASONER_PROMPT,
    MEMORY_V2_VERIFIER_PROMPT,
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
    "before",
    "between",
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

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

TEMPORAL_PATTERN = re.compile(
    r"\b("
    r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
    r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)|"
    r"mid-(?:January|February|March|April|May|June|July|August|September|October|November|December)|"
    r"early\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)|"
    r"late\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)|"
    r"last\s+(?:weekend|week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"next\s+(?:weekend|week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"\d+(?:\.\d+)?\s+(?:day|days|week|weeks|month|months|year|years)|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|days|week|weeks|month|months|year|years)|"
    r"first|earlier|later|ago|before|after|combined"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryCard:
    id: str
    t: int
    source: str
    source_date: str | None
    text: str


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
    max_tokens: int = 512,
) -> tuple[ModelCallResult, dict[str, Any]]:
    call = call_model(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return call, _parse_json_loose(call.content)


def _source_date(source: str) -> str | None:
    match = re.search(r"@\s*(\d{4}/\d{2}/\d{2}\s*\([^)]+\)|\d{4}/\d{2}/\d{2})", source)
    if match:
        return match.group(1).strip()
    return None


def _split_message(message: StreamMessage, max_chars: int) -> list[str]:
    text = message.text.strip()
    if not text:
        return []

    # LongMemEval converted sessions often contain repeated "user:" turns in
    # one line. Split those turns so temporal anchors stay close to their event.
    parts = [part.strip() for part in re.split(r"(?=\buser:\s)", text) if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        parts = [text]

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(part), max_chars):
                chunks.append(part[start : start + max_chars].strip())
            continue

        candidate = f"{current}\n{part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def _build_memory_cards(task: StreamTask, max_chars_per_card: int = 1200) -> list[MemoryCard]:
    cards: list[MemoryCard] = []
    for message in task.stream:
        for chunk in _split_message(message, max_chars=max_chars_per_card):
            cards.append(
                MemoryCard(
                    id=f"m{len(cards) + 1:03d}",
                    t=message.t,
                    source=message.source,
                    source_date=_source_date(message.source),
                    text=chunk,
                )
            )
    return cards


def _normalize_tokens(text: str) -> list[str]:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [token for token in lowered.split() if len(token) > 2 and token not in STOPWORDS]


def _quoted_phrases(question: str) -> list[str]:
    phrases = []
    for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", question):
        phrase = match.group(1) or match.group(2)
        if phrase:
            phrases.append(phrase.strip())
    return phrases


def _question_options(question: str) -> list[str]:
    options = _quoted_phrases(question)

    # Capture simple A-or-B candidates when they are not quoted.
    match = re.search(r"\b(?:the\s+)?([^?,]+?)\s+or\s+(?:the\s+)?([^?,]+?)(?:\?|$)", question, flags=re.IGNORECASE)
    if match:
        for group in match.groups():
            candidate = group.strip(" .")
            if 1 <= len(candidate.split()) <= 6:
                options.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for option in options:
        key = option.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(option)
    return deduped


def _is_temporal_question(question: str) -> bool:
    temporal_words = {
        "before",
        "after",
        "first",
        "date",
        "time",
        "days",
        "weeks",
        "months",
        "years",
        "ago",
        "long",
        "passed",
        "between",
        "combined",
    }
    return bool(temporal_words & set(_normalize_tokens(question.lower())))


def _card_score(question: str, options: list[str], card: MemoryCard) -> float:
    question_tokens = set(_normalize_tokens(question))
    card_text = f"{card.source} {card.text}"
    card_tokens = set(_normalize_tokens(card_text))
    score = float(len(question_tokens & card_tokens))

    lower_text = card_text.lower()
    for option in options:
        option_lower = option.lower()
        option_tokens = set(_normalize_tokens(option))
        if option_lower and option_lower in lower_text:
            score += 6.0
        elif option_tokens:
            score += 1.5 * len(option_tokens & card_tokens)

    if TEMPORAL_PATTERN.search(card.text):
        score += 1.0 if _is_temporal_question(question) else 0.25
    return score


def _rank_cards(question: str, cards: list[MemoryCard], limit: int) -> list[MemoryCard]:
    if len(cards) <= limit:
        return cards

    options = _question_options(question)
    scored = [(_card_score(question, options, card), -index, card) for index, card in enumerate(cards)]
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [card for score, _index, card in scored[:limit] if score > 0]
    if not selected:
        selected = [card for _score, _index, card in scored[:limit]]
    return sorted(selected, key=lambda card: (card.t, card.id))


def _card_to_dict(card: MemoryCard, max_text_chars: int | None = None) -> dict[str, Any]:
    text = card.text
    if max_text_chars is not None and len(text) > max_text_chars:
        text = text[:max_text_chars].rstrip() + " ..."
    return {
        "id": card.id,
        "t": card.t,
        "source": card.source,
        "source_date": card.source_date,
        "text": text,
    }


def _evidence_text(cards: list[MemoryCard]) -> str:
    lines = []
    for card in cards:
        source_date = f" source_date={card.source_date}" if card.source_date else ""
        lines.append(f"[{card.id} t={card.t}{source_date} {card.source}]\n{card.text}")
    return "\n".join(lines)


def _cards_text_chars(cards: list[MemoryCard]) -> int:
    return sum(len(card.text) for card in cards)


def _temporal_hints(cards: list[MemoryCard], max_hints: int = 40) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for card in cards:
        for match in TEMPORAL_PATTERN.finditer(card.text):
            start = max(0, match.start() - 130)
            end = min(len(card.text), match.end() + 130)
            snippet = re.sub(r"\s+", " ", card.text[start:end]).strip()
            hints.append(
                {
                    "id": card.id,
                    "source_date": card.source_date,
                    "cue": match.group(0),
                    "snippet": snippet,
                }
            )
            if len(hints) >= max_hints:
                return hints
    return hints


def _looks_like_temporal_value(answer: str) -> bool:
    normalized = answer.strip()
    if not normalized:
        return False
    return bool(TEMPORAL_PATTERN.search(normalized)) or bool(
        re.fullmatch(r"\d{1,4}(?:[/.-]\d{1,2}){0,2}", normalized)
    )


def _question_expects_temporal_value(question: str) -> bool:
    normalized = question.lower()
    temporal_value_phrases = (
        "what date",
        "which date",
        "what time",
        "which time",
        "how many",
        "how long",
        "how much time",
        "how old",
    )
    return any(phrase in normalized for phrase in temporal_value_phrases)


def _clean_event_name(name: str) -> str:
    cleaned = clean_answer(name)
    cleaned = re.sub(r"\b(issue|problem|event|item|task)\b$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or clean_answer(name)


def _first_event_name(event_ledger: dict[str, Any]) -> str | None:
    events = event_ledger.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "").strip()
        if name and not _looks_like_temporal_value(name):
            return _clean_event_name(name)
    return None


def _repair_answer_type(question: str, answer: str, event_ledger: dict[str, Any]) -> tuple[str, str | None]:
    if _question_expects_temporal_value(question) or not _looks_like_temporal_value(answer):
        return answer, None

    replacement = _first_event_name(event_ledger)
    if replacement:
        return replacement, "date_time_answer_replaced_with_event_name"
    return answer, None


def _selected_evidence(
    task: StreamTask,
    cards: list[MemoryCard],
    full_evidence_budget_chars: int,
    max_ranked_cards: int,
) -> tuple[list[MemoryCard], str]:
    if _cards_text_chars(cards) <= full_evidence_budget_chars:
        return cards, "full_stream_under_budget"
    return _rank_cards(task.question, cards, limit=max_ranked_cards), "ranked_cards_over_budget"


def run_memory_v2_swarm(
    task: StreamTask,
    small_model: str,
    temperature: float = 0.0,
    full_evidence_budget_chars: int = 16000,
    max_ranked_cards: int = 16,
) -> MethodResult:
    memory_cards = _build_memory_cards(task)
    selected_cards, selection_strategy = _selected_evidence(
        task,
        memory_cards,
        full_evidence_budget_chars=full_evidence_budget_chars,
        max_ranked_cards=max_ranked_cards,
    )
    evidence = _evidence_text(selected_cards)
    hints = _temporal_hints(selected_cards)
    options = _question_options(task.question)
    calls: list[ModelCallResult] = []

    extractor_call, event_ledger = _call_agent(
        small_model,
        MEMORY_V2_EVENT_EXTRACTOR_PROMPT,
        (
            f"Question:\n{task.question}\n\n"
            f"Question candidate phrases:\n{_json_text(options)}\n\n"
            f"Deterministic temporal hints:\n{_json_text(hints)}\n\n"
            f"Selected memory evidence:\n{evidence}"
        ),
        temperature,
    )
    calls.append(extractor_call)

    reasoner_call, reasoner = _call_agent(
        small_model,
        MEMORY_V2_REASONER_PROMPT,
        (
            f"Question:\n{task.question}\n\n"
            f"Question candidate phrases:\n{_json_text(options)}\n\n"
            f"Extracted evidence ledger:\n{_json_text(event_ledger)}\n\n"
            f"Selected memory evidence:\n{evidence}"
        ),
        temperature,
    )
    calls.append(reasoner_call)

    proposed_answer = str(
        reasoner.get("answer")
        or reasoner.get("final_answer")
        or reasoner_call.content
    ).strip()

    verifier_call, verification = _call_agent(
        small_model,
        MEMORY_V2_VERIFIER_PROMPT,
        (
            f"Question:\n{task.question}\n\n"
            f"Question candidate phrases:\n{_json_text(options)}\n\n"
            f"Extracted evidence ledger:\n{_json_text(event_ledger)}\n\n"
            f"Selected memory evidence:\n{evidence}\n\n"
            f"Proposed answer: {proposed_answer}"
        ),
        temperature,
    )
    calls.append(verifier_call)

    final_answer = clean_answer(str(
        verification.get("answer")
        or verification.get("final_answer")
        or proposed_answer
    ))
    final_answer, answer_type_repair = _repair_answer_type(task.question, final_answer, event_ledger)
    rationale = verification.get("rationale") or reasoner.get("rationale")
    if rationale is not None:
        rationale = str(rationale)

    return MethodResult(
        answer=final_answer,
        confidence=_coerce_confidence(
            verification.get("confidence", reasoner.get("confidence"))
        ),
        rationale=rationale,
        raw={
            "architecture": "memory_v2",
            "memory_cards_count": len(memory_cards),
            "selection_strategy": selection_strategy,
            "selected_cards": [_card_to_dict(card, max_text_chars=900) for card in selected_cards],
            "question_candidate_phrases": options,
            "temporal_hints": hints,
            "event_extractor": {
                "raw_output": extractor_call.content,
                "parsed_output": event_ledger,
            },
            "reasoner": {
                "raw_output": reasoner_call.content,
                "parsed_output": reasoner,
            },
            "verification": {
                "raw_output": verifier_call.content,
                "parsed_output": verification,
            },
            "answer_type_repair": answer_type_repair,
        },
        model_calls=len(calls),
        input_tokens=_sum_optional(calls, "input_tokens"),
        output_tokens=_sum_optional(calls, "output_tokens"),
        total_tokens=_sum_optional(calls, "total_tokens"),
        latency_seconds=sum(call.latency_seconds for call in calls),
    )
