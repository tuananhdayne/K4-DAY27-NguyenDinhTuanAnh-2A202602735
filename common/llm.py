"""LLM factory for an OpenAI-compatible provider."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set — copy .env.example to .env and add your "
            "provider API key"
        )
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )
