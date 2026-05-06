from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StreamMessage(BaseModel):
    t: int
    source: str
    text: str


class StreamTask(BaseModel):
    id: str
    task_type: str
    stream: list[StreamMessage]
    question: str
    gold_answer: str
    aliases: list[str] = Field(default_factory=list)


class ModelCallResult(BaseModel):
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float
    raw_response: Any | None = None


class MethodResult(BaseModel):
    answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    raw: Any | None = None
    model_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float
