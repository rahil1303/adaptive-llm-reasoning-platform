"""
Unified LLM provider service.
All models are called via OpenAI-compatible API format.
"""

import time
import httpx
from app.config import AVAILABLE_MODELS, ModelConfig


async def call_llm(
    model_key: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """
    Call any supported LLM. Returns dict with response text, token count, and latency.
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")

    config = AVAILABLE_MODELS[model_key]
    api_key = config.api_key

    if not api_key:
        raise ValueError(f"API key not set for {model_key}. Set env var: {config.api_key_env}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config.model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{config.api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})

    return {
        "text": choice["message"]["content"],
        "tokens_used": usage.get("total_tokens", 0),
        "latency_ms": round(elapsed_ms, 1),
        "model_id": model_key,
        "model_name": config.display_name,
        "finish_reason": choice.get("finish_reason", "unknown"),
    }


async def call_multiple(
    model_keys: list[str],
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> list[dict]:
    """
    Call multiple models concurrently with the same prompt.
    Returns list of results (including errors as dicts).
    """
    import asyncio

    async def _safe_call(key: str) -> dict:
        try:
            return await call_llm(key, messages, temperature, max_tokens)
        except Exception as e:
            return {
                "text": f"Error: {str(e)}",
                "tokens_used": 0,
                "latency_ms": 0,
                "model_id": key,
                "model_name": AVAILABLE_MODELS.get(key, {}).display_name if key in AVAILABLE_MODELS else key,
                "finish_reason": "error",
            }

    results = await asyncio.gather(*[_safe_call(k) for k in model_keys])
    return list(results)
