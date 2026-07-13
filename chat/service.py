"""LLM chat service — wraps the OpenAI client for both streaming and non-streaming calls."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .history import DEFAULT_SYSTEM_PROMPT, SessionData

# ---------------------------------------------------------------------------
# Client initialisation (reads from .env or environment variables)
# ---------------------------------------------------------------------------
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Lazy-initialise the AsyncOpenAI client (called once per process)."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "sk-placeholder")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_default_model() -> str:
    return os.getenv("MODEL_NAME", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------

async def chat(
    session: SessionData,
    message: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Send a user message and return the assistant's full reply."""
    # Record user message
    session.add_message("user", message)

    # Build context
    system = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    messages = session.get_context(system_prompt=system)
    model_name = model or get_default_model()

    client = _get_client()
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )

    reply = response.choices[0].message.content or ""
    session.add_message("assistant", reply)
    return reply


# ---------------------------------------------------------------------------
# Streaming chat (async generator yielding text deltas)
# ---------------------------------------------------------------------------

async def chat_stream(
    session: SessionData,
    message: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Send a user message and yield reply chunks as they arrive."""
    session.add_message("user", message)

    system = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    messages = session.get_context(system_prompt=system)
    model_name = model or get_default_model()

    client = _get_client()
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        stream=True,
    )

    full_reply = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_reply += delta
            yield delta

    # Save the complete assistant reply to session history
    session.add_message("assistant", full_reply)