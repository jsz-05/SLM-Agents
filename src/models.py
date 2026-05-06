from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from litellm import completion

from src.model_names import normalize_model_name
from src.schemas import ModelCallResult


load_dotenv()

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
if _openrouter_key:
    os.environ["OPENROUTER_API_KEY"] = _openrouter_key


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _serialize_response(response: Any) -> Any:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    if hasattr(response, "dict"):
        try:
            return response.dict()
        except Exception:
            pass
    try:
        return dict(response)
    except Exception:
        return repr(response)


def _extract_content(response: Any) -> str:
    choices = _get_attr_or_key(response, "choices", [])
    if not choices:
        return ""
    first = choices[0]
    message = _get_attr_or_key(first, "message", {})
    content = _get_attr_or_key(message, "content", "")
    return content or ""


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = _get_attr_or_key(response, "usage")
    input_tokens = _get_attr_or_key(usage, "prompt_tokens")
    output_tokens = _get_attr_or_key(usage, "completion_tokens")
    total_tokens = _get_attr_or_key(usage, "total_tokens")
    return input_tokens, output_tokens, total_tokens


def call_model(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> ModelCallResult:
    model = normalize_model_name(model)
    start = time.perf_counter()
    extra_params: dict[str, Any] = {}
    if model.startswith("ollama/"):
        # Safer for local comparison runs that switch between small and large
        # models. Ollama's default keep-alive can leave both models resident.
        extra_params["keep_alive"] = os.getenv("OLLAMA_KEEP_ALIVE", "0")
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra_params,
        )
    except Exception as exc:
        latency = time.perf_counter() - start
        raise RuntimeError(
            f"LiteLLM call failed for model '{model}' after {latency:.2f}s: {exc}"
        ) from exc

    latency = time.perf_counter() - start
    input_tokens, output_tokens, total_tokens = _extract_usage(response)
    return ModelCallResult(
        content=_extract_content(response),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency,
        raw_response=_serialize_response(response),
    )
