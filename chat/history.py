"""Conversation history manager with per-session storage."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

# In-memory store: session_id -> SessionData
# For production, replace with Redis / database
_sessions: OrderedDict[str, "SessionData"] = OrderedDict()

MAX_SESSIONS = 10_000  # LRU eviction threshold
MAX_HISTORY_PER_SESSION = 50  # Keep last N messages per session

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Provide clear, concise, and accurate responses. "
    "If you're unsure about something, say so honestly."
)


@dataclass
class SessionData:
    """Holds conversation history and metadata for a single chat session."""

    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim history if it exceeds the limit."""
        self.messages.append({"role": role, "content": content})
        self.last_active = time.time()
        # Keep only the most recent messages
        if len(self.messages) > MAX_HISTORY_PER_SESSION:
            self.messages = self.messages[-MAX_HISTORY_PER_SESSION:]

    def get_context(self, system_prompt: Optional[str] = None) -> list[dict[str, str]]:
        """Return the full message list including an optional system prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.messages)
        return messages


def _evict_if_needed() -> None:
    """Remove the oldest sessions when the store exceeds MAX_SESSIONS."""
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


def create_session() -> SessionData:
    """Create a new chat session and return it."""
    session_id = uuid.uuid4().hex[:16]
    session = SessionData(session_id=session_id)
    _sessions[session_id] = session
    _evict_if_needed()
    return session


def get_or_create_session(session_id: Optional[str] = None) -> SessionData:
    """Return an existing session or create a new one."""
    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        session.last_active = time.time()
        # Move to end (most recently used)
        _sessions.move_to_end(session_id)
        return session
    return create_session()


def get_session(session_id: str) -> Optional[SessionData]:
    """Return a session by ID, or None if not found."""
    return _sessions.get(session_id)


def list_sessions(limit: int = 20) -> list[SessionData]:
    """Return the most recent sessions."""
    items = list(reversed(_sessions.values()))
    return items[:limit]


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def clear_all_sessions() -> int:
    """Clear all sessions and return the count of deleted sessions."""
    count = len(_sessions)
    _sessions.clear()
    return count