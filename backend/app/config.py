"""
Configuration management — loads API keys and settings from environment.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    display_name: str
    api_base: str
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.7

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)


# All supported models — add new ones here
AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "llama-3.3-70b": ModelConfig(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        display_name="LLaMA 3.3 70B",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "llama-3.1-8b": ModelConfig(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        display_name="LLaMA 3.1 8B",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "qwen3-32b": ModelConfig(
        provider="groq",
        model_id="qwen/qwen3-32b",
        display_name="Qwen 3 32B",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
}

# Interaction mode system prompts
INTERACTION_MODES = {
    "direct": {
        "name": "Direct Response",
        "description": "Standard direct answer generation",
        "system_prompt": "You are a helpful assistant. Answer the user's question directly and thoroughly, using the provided context when available.",
    },
    "hint_first": {
        "name": "Hint-First",
        "description": "Provides hints before full answers to encourage reasoning",
        "system_prompt": (
            "You are a reasoning coach. When answering questions:\n"
            "1. First provide 2-3 hints that guide the user toward the answer\n"
            "2. Then provide the full answer clearly separated\n"
            "Format: start with '**Hints:**' then '**Answer:**'"
        ),
    },
    "guided_reasoning": {
        "name": "Guided Reasoning",
        "description": "Walks through reasoning step by step",
        "system_prompt": (
            "You are a structured reasoning assistant. For every question:\n"
            "1. Break down the problem into sub-questions\n"
            "2. Address each sub-question with evidence from context\n"
            "3. Synthesize into a final answer\n"
            "4. Rate your confidence (low/medium/high)\n"
            "Show your reasoning chain explicitly."
        ),
    },
}

# Rate limiting defaults
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_RPM", "30"))
