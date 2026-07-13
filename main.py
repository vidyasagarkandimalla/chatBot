"""
AI-Assisted Chatbot — FastAPI Application
==========================================

Endpoints
---------
GET  /                          → Chat UI (HTML page)
POST /api/chat                  → Non-streaming chat
WS   /api/chat/stream           → Streaming chat over WebSocket
GET  /api/sessions              → List recent sessions
GET  /api/sessions/{id}         → Get session info
DELETE /api/sessions/{id}       → Delete a session
DELETE /api/sessions            → Clear all sessions
GET  /api/health                → Health check
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from chat.history import (
    clear_all_sessions,
    delete_session,
    get_session,
    get_or_create_session,
    list_sessions,
)
from chat.models import (
    ApiError,
    ChatRequest,
    ChatResponse,
    SessionInfo,
    StreamChunk,
)
from chat.service import chat, chat_stream, get_default_model

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print("🚀 Chatbot API is starting up…")
    yield
    print("👋 Chatbot API is shutting down.")


app = FastAPI(
    title="AI Chatbot API",
    description="A FastAPI-based AI-assisted chatbot with streaming support.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (CSS, JS, images) from /static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# HTML Chat Page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat_page():
    """Serve the single-page chat UI."""
    html_path = STATIC_DIR / "chat.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>chat.html not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model": get_default_model(),
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Chat Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse, responses={500: {"model": ApiError}})
async def handle_chat(request: ChatRequest):
    """Non-streaming chat endpoint. Returns the full assistant reply."""
    try:
        session = get_or_create_session(request.session_id)
        reply = await chat(
            session,
            request.message,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature or 0.7,
        )
        return ChatResponse(
            session_id=session.session_id,
            message=reply,
            model=request.model or get_default_model(),
            history_length=len(session.messages),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/api/chat/stream")
async def handle_chat_stream(ws: WebSocket):
    """Streaming chat endpoint over WebSocket.

    Client sends:
        {"message": "...", "session_id": "...?", "system_prompt": "...?", "model": "...?", "temperature": 0.7}

    Server sends back a series of JSON objects:
        {"session_id": "...", "delta": "...", "done": false}
        {"session_id": "...", "delta": "", "done": true}
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
                continue

            message = data.get("message", "").strip()
            if not message:
                await ws.send_json({"error": "Empty message"})
                continue

            session_id = data.get("session_id")
            system_prompt = data.get("system_prompt")
            model = data.get("model")
            temperature = float(data.get("temperature", 0.7))

            session = get_or_create_session(session_id)

            async for delta in chat_stream(
                session,
                message,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
            ):
                await ws.send_json(
                    StreamChunk(
                        session_id=session.session_id,
                        delta=delta,
                        done=False,
                    ).model_dump()
                )

            # Send final "done" signal
            await ws.send_json(
                StreamChunk(
                    session_id=session.session_id,
                    delta="",
                    done=True,
                ).model_dump()
            )
    except WebSocketDisconnect:
        pass  # Client closed the connection


# ---------------------------------------------------------------------------
# Session Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_all_sessions(limit: int = 20):
    """List recent chat sessions."""
    sessions = list_sessions(limit=limit)
    return [
        SessionInfo(
            session_id=s.session_id,
            message_count=len(s.messages),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.created_at)),
            last_active=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.last_active)),
        )
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get info about a specific session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session.session_id,
        message_count=len(session.messages),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at)),
        last_active=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.last_active)),
    )


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a specific session."""
    if delete_session(session_id):
        return {"detail": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/api/sessions")
async def remove_all_sessions():
    """Clear all sessions."""
    count = clear_all_sessions()
    return {"detail": f"Deleted {count} sessions"}


# ---------------------------------------------------------------------------
# Entry Point (for direct execution: python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )