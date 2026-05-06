def normalize_model_name(model: str) -> str:
    """Accept native OpenRouter model ids and adapt them for LiteLLM.

    OpenRouter's own docs use model ids like qwen/qwen3-4b:free, while
    LiteLLM needs the provider prefix: openrouter/qwen/qwen3-4b:free.
    """
    if model.startswith("openrouter/"):
        return model
    if model == "openrouter/free":
        return model
    if ":free" in model and "/" in model:
        return f"openrouter/{model}"
    return model
