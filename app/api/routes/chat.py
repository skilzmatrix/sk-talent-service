"""Chat endpoint routes."""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain_core.messages import HumanMessage

from app.agent import agent_app, load_chat_history, save_chat_history

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str


router = APIRouter()


def _extract_text_from_chunk(content: object) -> str:
    """Extract plain text from a streaming chunk's content field.

    Gemini 3.x with ``include_thoughts=True`` emits content as a list of typed
    parts instead of a plain string:
      - {"type": "thinking", "thinking": "..."}  ← internal thought; skip
      - {"type": "text",     "text":    "..."}   ← actual response; emit

    Older models / plain completions emit a bare string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    chunks.append(part.get("text", ""))
                elif part_type is None and "text" in part:
                    # Fallback: dict has "text" but no "type" discriminator
                    chunks.append(part.get("text", ""))
                # "thinking" / "tool_use" / etc. are intentionally skipped
        return "".join(chunks)
    return ""


async def stream_agent(conversation_id: str, message: str) -> AsyncGenerator[str, None]:
    """Run the ReAct agent and stream SSE events to the client.

    Event types emitted:
      {"type": "text",       "content": str}           – streamed token
      {"type": "tool_start", "tool": str, "input": …}  – tool call begins
      {"type": "tool_end",   "tool": str, "output": …} – tool call finished
      {"type": "done"}                                  – stream complete
      {"type": "error",      "content": str}            – unhandled exception
    """
    history = load_chat_history(conversation_id)
    history.append(HumanMessage(content=message))
    state = {"messages": history}

    final_messages: list | None = None

    try:
        async for event in agent_app.astream_events(state, version="v2"):
            kind: str = event["event"]

            # ── Streamed text tokens ──────────────────────────────────────────
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                text = _extract_text_from_chunk(chunk.content)
                if text:
                    yield json.dumps({"type": "text", "content": text})

            # ── Tool lifecycle ────────────────────────────────────────────────
            elif kind == "on_tool_start":
                yield json.dumps(
                    {
                        "type": "tool_start",
                        "tool": event["name"],
                        "input": event["data"].get("input"),
                    }
                )

            elif kind == "on_tool_end":
                raw_out = event["data"].get("output")
                yield json.dumps(
                    {
                        "type": "tool_end",
                        "tool": event["name"],
                        "output": str(raw_out) if raw_out is not None else "",
                    }
                )

            # ── Capture final graph state for persistence ─────────────────────
            elif kind == "on_chain_end":
                # Do NOT filter by event["name"]: the compiled graph name varies
                # across LangGraph versions ("LangGraph", "agent", etc.).
                # Instead, take the last on_chain_end whose output carries messages.
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict) and "messages" in output:
                    final_messages = output["messages"]

        # ── Persist after the run completes (single run, no double-invoke) ──
        if final_messages is not None:
            save_chat_history(conversation_id, list(final_messages))
        else:
            logger.warning(
                "No final messages captured for conversation %s; history not saved.",
                conversation_id,
            )

        yield json.dumps({"type": "done"})

    except Exception as exc:
        logger.error(
            "Agent stream error for conversation %s: %s",
            conversation_id,
            exc,
            exc_info=True,
        )
        yield json.dumps({"type": "error", "content": str(exc)})


@router.post("/api/chat/{conversation_id}/stream")
async def chat_stream(conversation_id: str, request: ChatRequest):
    """Stream the agentic response back to the client using Server-Sent Events."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return EventSourceResponse(stream_agent(conversation_id, request.message.strip()))


@router.get("/api/chat")
async def get_conversations():
    """Return all chat histories from Supabase, ordered newest first."""
    try:
        from app.supabase_operations import _client

        client = _client()
        response = (
            client.table("chat_histories")
            .select("conversation_id, messages, updated_at")
            .order("updated_at", desc=True)
            .execute()
        )
        return {"conversations": response.data}
    except Exception as exc:
        logger.error("Failed to fetch conversations: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")


@router.delete("/api/chat/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a chat history from Supabase."""
    try:
        from app.supabase_operations import _client

        client = _client()
        client.table("chat_histories").delete().eq("conversation_id", conversation_id).execute()
        return {"success": True}
    except Exception as exc:
        logger.error(
            "Failed to delete conversation %s: %s", conversation_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
