import hashlib
import hmac
import logging
import time
from collections import defaultdict
from io import BytesIO
from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Request, Response
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph import graph_builder
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText
from ai_companion.settings import settings

logger = logging.getLogger(__name__)

# Global module instances
speech_to_text = SpeechToText()
image_to_text = ImageToText()

# Router for WhatsApp response
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = settings.WHATSAPP_TOKEN
WHATSAPP_PHONE_NUMBER_ID = settings.WHATSAPP_PHONE_NUMBER_ID

# Maximum media file size in bytes (25 MB — WhatsApp's own limit)
MAX_MEDIA_SIZE_BYTES = 25 * 1024 * 1024

# ──── Rate Limiter ───────────────────────────────────────
# Simple per-user in-memory rate limiter
_rate_limit_store: Dict[str, list] = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 20


def _is_rate_limited(user_id: str) -> bool:
    """Check if a user has exceeded the rate limit.

    Allows RATE_LIMIT_MAX_REQUESTS messages per RATE_LIMIT_WINDOW_SECONDS.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    # Prune old entries
    _rate_limit_store[user_id] = [t for t in _rate_limit_store[user_id] if t > window_start]

    if len(_rate_limit_store[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return True

    _rate_limit_store[user_id].append(now)
    return False


# ──── Webhook Signature Verification (V2) ────────────────
def _verify_webhook_signature(payload: bytes, signature_header: Optional[str]) -> bool:
    """Verify the X-Hub-Signature-256 HMAC from Meta/WhatsApp.

    Returns True if the signature is valid, or if WHATSAPP_APP_SECRET
    is not configured (allows graceful opt-in during development).
    """
    app_secret = getattr(settings, "WHATSAPP_APP_SECRET", None)
    if not app_secret:
        # Not configured — log a warning but don't block (dev mode)
        logger.warning(
            "WHATSAPP_APP_SECRET is not set — webhook signature verification is DISABLED. "
            "Set it before going to production!"
        )
        return True

    if not signature_header:
        logger.warning("Missing X-Hub-Signature-256 header — rejecting request")
        return False

    # Header format: "sha256=<hex_digest>"
    if not signature_header.startswith("sha256="):
        return False

    expected_sig = signature_header[7:]
    computed_sig = hmac.new(
        app_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed_sig, expected_sig)


# ──── Main Handler ───────────────────────────────────────
@whatsapp_router.api_route("/whatsapp_response", methods=["GET", "POST"])
async def whatsapp_handler(request: Request) -> Response:
    """Handles incoming messages and status updates from the WhatsApp Cloud API."""

    if request.method == "GET":
        params = request.query_params
        if params.get("hub.verify_token") == settings.WHATSAPP_VERIFY_TOKEN:
            return Response(content=params.get("hub.challenge"), status_code=200)
        return Response(content="Verification token mismatch", status_code=403)

    # ── V2: Verify webhook signature before processing ──
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not _verify_webhook_signature(raw_body, signature):
        logger.warning("Webhook signature verification failed — rejecting request")
        return Response(content="Invalid signature", status_code=403)

    try:
        data = await request.json()
        change_value = data["entry"][0]["changes"][0]["value"]
        if "messages" in change_value:
            message = change_value["messages"][0]
            from_number = message["from"]
            session_id = from_number

            # ── V5: Rate limiting ──
            if _is_rate_limited(session_id):
                logger.warning(f"Rate limit exceeded for user {session_id}")
                return Response(content="Rate limit exceeded", status_code=429)

            # Get user message and handle different message types
            content = ""
            if message["type"] == "audio":
                content = await process_audio_message(message)
            elif message["type"] == "image":
                # Get image caption if any
                content = message.get("image", {}).get("caption", "")
                # Download and analyze image
                image_bytes = await download_media(message["image"]["id"])
                try:
                    description = await image_to_text.analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of our conversation.",
                    )
                    content += f"\n[Image Analysis: {description}]"
                except Exception as e:
                    logger.warning(f"Failed to analyze image: {e}")
            else:
                content = message["text"]["body"]

            # ── V5: Enforce text length limit ──
            if len(content) > settings.WHATSAPP_MAX_TEXT_LENGTH:
                content = content[: settings.WHATSAPP_MAX_TEXT_LENGTH]
                logger.warning(f"Truncated oversized message from {session_id} to {settings.WHATSAPP_MAX_TEXT_LENGTH} chars")

            # Process message through the graph agent
            async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
                graph = graph_builder.compile(checkpointer=short_term_memory)
                await graph.ainvoke(
                    {"messages": [HumanMessage(content=content)], "user_id": session_id},
                    {"configurable": {"thread_id": session_id}},
                )

                # Get the workflow type and response from the state
                output_state = await graph.aget_state(config={"configurable": {"thread_id": session_id}})

            workflow = output_state.values.get("workflow", "conversation")
            response_message = output_state.values["messages"][-1].content

            # Handle different response types based on workflow
            if workflow == "audio":
                audio_buffer = output_state.values["audio_buffer"]
                success = await send_response(from_number, response_message, "audio", audio_buffer)
            elif workflow == "image":
                image_path = output_state.values["image_path"]
                with open(image_path, "rb") as f:
                    image_data = f.read()
                success = await send_response(from_number, response_message, "image", image_data)
            else:
                success = await send_response(from_number, response_message, "text")

            if not success:
                return Response(content="Failed to send message", status_code=500)

            return Response(content="Message processed", status_code=200)

        elif "statuses" in change_value:
            return Response(content="Status update received", status_code=200)

        else:
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)


# ──── Media helpers ──────────────────────────────────────
async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp with size validation (V4)."""
    media_metadata_url = f"{settings.WHATSAPP_GRAPH_API_URL}/{settings.WHATSAPP_API_VERSION}/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

        # ── V4: Check file size before downloading ──
        file_size = metadata.get("file_size")
        if file_size and int(file_size) > MAX_MEDIA_SIZE_BYTES:
            raise ValueError(
                f"Media file too large ({int(file_size)} bytes). "
                f"Max allowed: {MAX_MEDIA_SIZE_BYTES} bytes."
            )

        # Stream the download with a size guard
        downloaded = bytearray()
        async with client.stream("GET", download_url, headers=headers) as stream:
            stream.raise_for_status()
            async for chunk in stream.aiter_bytes(chunk_size=8192):
                downloaded.extend(chunk)
                if len(downloaded) > MAX_MEDIA_SIZE_BYTES:
                    raise ValueError(
                        f"Media download exceeded {MAX_MEDIA_SIZE_BYTES} bytes — aborting."
                    )
        return bytes(downloaded)


async def process_audio_message(message: Dict) -> str:
    """Download and transcribe audio message."""
    audio_id = message["audio"]["id"]
    audio_data = await download_media(audio_id)

    audio_buffer = BytesIO(audio_data)
    audio_buffer.seek(0)

    return await speech_to_text.transcribe(audio_buffer.read())


async def send_response(
    from_number: str,
    response_text: str,
    message_type: str = "text",
    media_content: bytes = None,
) -> bool:
    """Send response to user via WhatsApp API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    if message_type in ["audio", "image"]:
        try:
            mime_type = "audio/mpeg" if message_type == "audio" else "image/png"
            media_buffer = BytesIO(media_content)
            media_id = await upload_media(media_buffer, mime_type)
            json_data = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": message_type,
                message_type: {"id": media_id},
            }

            # Add caption for images
            if message_type == "image":
                json_data["image"]["caption"] = response_text
        except Exception as e:
            logger.error(f"Media upload failed, falling back to text: {e}")
            message_type = "text"

    if message_type == "text":
        json_data = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "type": "text",
            "text": {"body": response_text},
        }

    logger.debug(f"Sending {message_type} to {from_number}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.WHATSAPP_GRAPH_API_URL}/{settings.WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=json_data,
        )

    return response.status_code == 200


async def upload_media(media_content: BytesIO, mime_type: str) -> str:
    """Upload media to WhatsApp servers."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    filename = settings.WHATSAPP_AUDIO_FILENAME if "audio" in mime_type else settings.WHATSAPP_IMAGE_FILENAME
    files = {"file": (filename, media_content, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.WHATSAPP_GRAPH_API_URL}/{settings.WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files,
            data=data,
        )
        result = response.json()

    if "id" not in result:
        raise Exception("Failed to upload media")
    return result["id"]
