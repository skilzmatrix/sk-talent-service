"""Chat endpoint routes."""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import agent_app, load_chat_history, save_chat_history
from app.services import persistence_service

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    attachments: list[dict[str, object]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


router = APIRouter()

CHAT_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024
CHAT_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
CHAT_ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}


def _bad_request(detail: str) -> JSONResponse:
    return JSONResponse(status_code=400, content=[{"detail": detail}])


def _build_chat_attachment_name(filename: str | None) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    if ext not in CHAT_ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(f".{item}" for item in CHAT_ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Allowed extensions: {allowed}.")
    return f"{uuid.uuid4().hex}.{ext}"


def _merge_last_human_attachments(messages: list | None, current_human: HumanMessage) -> list | None:
    """Ensure the latest user turn keeps attachment metadata before persistence.

    Some agent runtimes may recreate HumanMessage objects and drop
    ``additional_kwargs``. This merge keeps the latest turn's attachments stable
    for history reload and chip rendering.
    """
    if not messages:
        return messages

    expected_content = current_human.content
    expected_kwargs = current_human.additional_kwargs or {}
    expected_attachments = expected_kwargs.get("attachments")
    if not expected_attachments:
        return messages

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, HumanMessage):
            continue
        if msg.content != expected_content:
            continue

        merged_kwargs = dict(msg.additional_kwargs or {})
        merged_kwargs.update(expected_kwargs)
        messages[idx] = HumanMessage(content=msg.content, additional_kwargs=merged_kwargs)
        return messages

    return messages


def _inject_message_attachments_for_client(messages: object) -> object:
    """Add ``data.attachments`` derived from ``data.additional_kwargs.attachments``.

    This keeps backward compatibility for chat UIs that render chips from a
    direct attachments field when loading historical messages.
    """
    if not isinstance(messages, list):
        return messages

    out: list[object] = []
    for item in messages:
        if not isinstance(item, dict):
            out.append(item)
            continue

        msg_copy = dict(item)
        data = msg_copy.get("data")
        if isinstance(data, dict):
            data_copy = dict(data)
            kwargs = data_copy.get("additional_kwargs")
            if isinstance(kwargs, dict) and "attachments" in kwargs and "attachments" not in data_copy:
                data_copy["attachments"] = kwargs.get("attachments")
            msg_copy["data"] = data_copy
        out.append(msg_copy)
    return out


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
            elif hasattr(part, "text"):
                # Some SDK revisions emit typed objects instead of plain dicts.
                chunks.append(str(getattr(part, "text") or ""))
                # "thinking" / "tool_use" / etc. are intentionally skipped
        return "".join(chunks)
    return ""


def _extract_last_ai_text(messages: list | None) -> str:
    """Return the most recent AI message text from final graph messages."""
    if not messages:
        return ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return _extract_text_from_chunk(msg.content)
    return ""


def _build_human_message(request: ChatRequest) -> HumanMessage:
    attachment_metadata: dict[str, object] = {}
    if request.attachments:
        attachment_metadata["attachments"] = request.attachments

    extra_payload = getattr(request, "model_extra", None) or {}
    if extra_payload:
        attachment_metadata.update(extra_payload)

    if attachment_metadata:
        return HumanMessage(content=request.message, additional_kwargs=attachment_metadata)
    return HumanMessage(content=request.message)


async def stream_agent(conversation_id: str, request: ChatRequest) -> AsyncGenerator[str, None]:
    """Run the ReAct agent and stream SSE events to the client.

    Event types emitted:
      {"type": "text",       "content": str}           – streamed token
      {"type": "tool_start", "tool": str, "input": …}  – tool call begins
      {"type": "tool_end",   "tool": str, "output": …} – tool call finished
      {"type": "done"}                                  – stream complete
      {"type": "error",      "content": str}            – unhandled exception
    """
    history = load_chat_history(conversation_id)
    current_human = _build_human_message(request)
    history.append(current_human)
    state = {"messages": history}

    final_messages: list | None = None
    emitted_text = False

    try:
        async for event in agent_app.astream_events(state, version="v2"):
            kind: str = event["event"]

            # ── Streamed text tokens ──────────────────────────────────────────
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                text = _extract_text_from_chunk(chunk.content)
                if text:
                    emitted_text = True
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
            final_messages = _merge_last_human_attachments(final_messages, current_human)
            save_chat_history(conversation_id, list(final_messages))
        else:
            logger.warning(
                "No final messages captured for conversation %s; history not saved.",
                conversation_id,
            )

        # Fallback for model integrations that do not emit token stream events.
        if not emitted_text:
            final_text = _extract_last_ai_text(final_messages)
            if final_text:
                yield json.dumps({"type": "text", "content": final_text})

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
    request.message = request.message.strip()
    return EventSourceResponse(stream_agent(conversation_id, request))


@router.post("/api/chat/attachments/upload", status_code=201, response_model=None)
async def upload_chat_attachment(
    file: UploadFile | None = File(default=None),
    conversation_id: str | None = Form(default=None),
) -> dict[str, object] | JSONResponse:
    if file is None:
        return _bad_request("Field 'file' is required.")

    content_type = (file.content_type or "").strip().lower()
    if content_type not in CHAT_ALLOWED_CONTENT_TYPES:
        return _bad_request("Unsupported file content type.")

    normalized_conversation_id: str | None = None
    if conversation_id and conversation_id.strip():
        try:
            normalized_conversation_id = str(uuid.UUID(conversation_id.strip()))
        except ValueError:
            return _bad_request("Field 'conversation_id' must be a valid UUID.")

    try:
        unique_name = _build_chat_attachment_name(file.filename)
    except ValueError as exc:
        return _bad_request(str(exc))

    file_bytes = await file.read()
    if len(file_bytes) > CHAT_ATTACHMENT_MAX_BYTES:
        return _bad_request("File size exceeds 25 MB limit.")

    try:
        attachment_record = await asyncio.to_thread(
            persistence_service.upload_chat_attachment,
            unique_name,
            file_bytes,
            content_type,
            len(file_bytes),
            normalized_conversation_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    storage_key = str(attachment_record.get("storage_key") or "")
    file_url = ""
    try:
        if storage_key:
            file_url = await asyncio.to_thread(persistence_service.get_signed_resume_url, storage_key)
    except Exception:
        file_url = ""

    return {
        "id": attachment_record.get("id"),
        "file_url": file_url,
        "file_name": attachment_record.get("file_name") or file.filename or unique_name,
        "mime_type": attachment_record.get("mime_type") or content_type,
        "storage_key": storage_key,
        "size": attachment_record.get("size_bytes", len(file_bytes)),
        "conversation_id": attachment_record.get("conversation_id") or normalized_conversation_id,
        "created_at": attachment_record.get("created_at"),
        # Backward compatible aliases used by older frontend clients.
        "path": storage_key,
        "name": attachment_record.get("file_name") or file.filename or unique_name,
        "content_type": attachment_record.get("mime_type") or content_type,
    }


@router.get("/api/chat/{conversation_id}/attachments")
async def list_chat_attachments(conversation_id: str):
    try:
        normalized_conversation_id = str(uuid.UUID(conversation_id))
    except ValueError:
        return _bad_request("Path param 'conversation_id' must be a valid UUID.")

    try:
        rows = await asyncio.to_thread(
            persistence_service.list_chat_attachments,
            normalized_conversation_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    attachments: list[dict[str, object]] = []
    for row in rows:
        storage_key = str(row.get("storage_key") or "")
        file_url = ""
        try:
            if storage_key:
                file_url = await asyncio.to_thread(
                    persistence_service.get_signed_resume_url,
                    storage_key,
                )
        except Exception:
            file_url = ""

        attachments.append(
            {
                "id": row.get("id"),
                "conversation_id": row.get("conversation_id"),
                "file_name": row.get("file_name"),
                "mime_type": row.get("mime_type"),
                "size": row.get("size_bytes", 0),
                "storage_key": storage_key,
                "file_url": file_url,
                "created_at": row.get("created_at"),
            }
        )

    return {"attachments": attachments}


@router.delete("/api/chat/attachments/{attachment_id}")
async def delete_chat_attachment(attachment_id: str):
    try:
        normalized_attachment_id = str(uuid.UUID(attachment_id))
    except ValueError:
        return _bad_request("Path param 'attachment_id' must be a valid UUID.")

    try:
        deleted = await asyncio.to_thread(
            persistence_service.delete_chat_attachment,
            normalized_attachment_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if deleted is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    return {"success": True, "attachment_id": normalized_attachment_id}


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
        conversations = response.data or []
        normalized: list[dict[str, object]] = []
        for row in conversations:
            row_copy = dict(row)
            row_copy["messages"] = _inject_message_attachments_for_client(row_copy.get("messages"))
            normalized.append(row_copy)
        return {"conversations": normalized}
    except Exception as exc:
        logger.error("Failed to fetch conversations: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")


@router.delete("/api/chat/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a chat history from Supabase."""
    try:
        await asyncio.to_thread(
            persistence_service.delete_conversation_with_attachments,
            conversation_id,
        )
        return {"success": True}
    except Exception as exc:
        logger.error(
            "Failed to delete conversation %s: %s", conversation_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
