import base64
import io
import time
import wave
import asyncio
import json
import re
import os
import uuid
from livekit import api


from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from contextlib import suppress
from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

from cartesia import AsyncCartesia

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
)

from httpx2 import websocket
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from google import genai

from graph import graph, initialize_mcp, extract_text

from dotenv import load_dotenv


load_dotenv()

# ============================================================
# CARTESIA TTS CONFIGURATION
# ============================================================

CARTESIA_MODEL_ID = os.getenv("CARTESIA_MODEL_ID", "sonic-3.6")

CARTESIA_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"

CARTESIA_SAMPLE_RATE = 44100

cartesia_client = AsyncCartesia(
    api_key=os.getenv("CARTESIA_API_KEY")
)

# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client()


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n============================================================")
    print("APPLICATION STARTUP")
    print("============================================================")

    print("\nInitializing MCP agent...")

    await initialize_mcp()

    print("\nApplication startup complete.")

    yield

    print("\nApplication shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str
    thread_id: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Backend is running"
    }

@app.post("/livekit/token")
async def create_livekit_token():
    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([livekit_url, api_key, api_secret]):
        raise HTTPException(
            status_code=500,
            detail="LiveKit configuration is missing",
        )

    room_name = f"voice-{uuid.uuid4().hex[:12]}"
    participant_identity = f"user-{uuid.uuid4().hex[:8]}"

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name("Voice User")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )

    return {
        "serverUrl": livekit_url,
        "roomName": room_name,
        "participantIdentity": participant_identity,
        "participantToken": token.to_jwt(),
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    async def generate():
        start_time = time.perf_counter()

        streamed_text = ""
        final_response = ""

        def send_event(event_type, data):
            return (
                f"data: {json.dumps({'type': event_type, **data})}\n\n"
            )

        try:
            async for mode, chunk in graph.astream(
                {
                    "message": request.message,
                    "route": "",
                    "response": "",
                    "messages": [],
                },
                config={
                    "configurable": {
                        "thread_id": request.thread_id
                    }
                },
                stream_mode=["updates", "messages"],
            ):

                # 1. Handle graph node updates
                if mode == "updates":

                    for node_name, update in chunk.items():

                        response = update.get("response", "")

                        print(f"\nDEBUG STREAM NODE: {node_name}")
                        print(f"DEBUG NODE RESPONSE: {response!r}")

                        if response:
                            final_response = response

                        # RAG and unsupported-question answers
                        if (
                            node_name == "rag"
                            and response
                            and not streamed_text
                        ):
                            streamed_text = response

                            yield send_event(
                                "token",
                                {"text": response}
                            )

                        # Customer ID validation response
                        elif (
                            node_name == "blocked"
                            and response
                            and not streamed_text
                        ):
                            streamed_text = response

                            yield send_event(
                                "token",
                                {"text": response}
                            )

                # 2. Handle genuine LLM text chunks
                elif mode == "messages":

                    message_chunk, metadata = chunk

                    node_name = metadata.get(
                        "langgraph_node", ""
                    )

                    # Ignore RAG judge output and tool results
                    if node_name != "mcp_agent":
                        continue

                    text = extract_text(
                        getattr(message_chunk, "content", "")
                    )

                    if text:
                        streamed_text += text

                        yield send_event(
                            "token",
                            {"text": text}
                        )

            # 3. Send a complete answer if no chunks were streamed
            if final_response and not streamed_text:
                streamed_text = final_response

                yield send_event(
                    "token",
                    {"text": final_response}
                )

            print("\nDEBUG FINAL RESPONSE:", repr(final_response))
            print("DEBUG STREAMED TEXT:", repr(streamed_text))
            print("DEBUG DONE ANSWER:", repr(final_response or streamed_text))
            

            # 4. Tell the frontend that the response is complete
            yield send_event(
                "done",
                {
                    "answer": final_response or streamed_text,
                    "elapsed_seconds": round(
                        time.perf_counter() - start_time, 3
                    ),
                },
            )

        except Exception as e:
            yield send_event(
                "error",
                {"message": str(e)}
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
