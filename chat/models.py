"""Pydantic models for the chatbot API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    message: str = Field(..., description="The user's message", min_length=1)
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID to maintain conversation context. "
        "A new session is created if not provided.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to override the default behavior.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model name override.",
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 - 2.0).",
    )


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    session_id: str = Field(..., description="The session ID for follow-up messages")
    message: str = Field(..., description="The assistant's reply")
    model: str = Field(..., description="The model used for generation")
    history_length: int = Field(..., description="Number of messages in session history")


class SessionInfo(BaseModel):
    """Info about a chat session."""

    session_id: str
    message_count: int
    created_at: str
    last_active: str


class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""

    session_id: str
    delta: str
    done: bool = False


class ApiError(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None