import base64
import io
import time
import wave
import asyncio
import json
import re
import os


from fastapi.responses import StreamingResponse
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


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    total_start = time.perf_counter()

    result = await graph.ainvoke(
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
        }
    )

    total_end = time.perf_counter()

    print("\n========================================")
    print(
        f"TOTAL /chat REQUEST: "
        f"{total_end - total_start:.3f}s"
    )
    print("========================================")

    return {
        "answer": result["response"]
    }


# ============================================================
# VOICE
# ============================================================

@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    thread_id: str = Form(...)
):

    total_start = time.perf_counter()

    print("\n============================================================")
    print("VOICE REQUEST")
    print("============================================================")

    # --------------------------------------------------------
    # READ AUDIO
    # --------------------------------------------------------

    audio_data = await audio.read()

    print(
        f"Received audio: {len(audio_data)} bytes"
    )

    print(
        f"Content type: {audio.content_type}"
    )


    # --------------------------------------------------------
    # UPLOAD AUDIO TO GEMINI
    # --------------------------------------------------------

    transcription_start = time.perf_counter()

    audio_file = await client.aio.files.upload(
    file=io.BytesIO(audio_data),
    config={
        "mime_type": "audio/webm"
    }
)

    print(
        f"Gemini audio upload complete: "
        f"{time.perf_counter() - transcription_start:.3f}s"
    )


    # --------------------------------------------------------
    # TRANSCRIPTION
    # --------------------------------------------------------

    transcription_response = (
        await client.aio.models.generate_content(
            model="gemini-3.5-transcribe",
            contents=[
                audio_file
            ],
        )
    )

    print("\n============================================================")
    print("TRANSCRIPTION RESPONSE")
    print("============================================================")
    print(transcription_response)


    transcription = ""

    if transcription_response.candidates:

        candidate = transcription_response.candidates[0]

        if candidate.content and candidate.content.parts:

            for part in candidate.content.parts:

                if part.audio_transcription:

                    transcription = (
                        part.audio_transcription.text
                    )

                    break

    transcription = transcription.strip()

    print(
        f"Transcription: {transcription}"
    )


    # --------------------------------------------------------
    # RUN LANGGRAPH
    # --------------------------------------------------------

    graph_start = time.perf_counter()

    result = await graph.ainvoke(
        {
            "message": transcription,
            "route": "",
            "response": "",
            "messages": []
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print(
        f"Graph execution: "
        f"{time.perf_counter() - graph_start:.3f}s"
    )

    answer = result["response"]

    print(
        f"Answer: {answer}"
    )


    # --------------------------------------------------------
    # TEXT TO SPEECH
    # --------------------------------------------------------

    tts_start = time.perf_counter()

    tts_response = (
        await client.aio.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents=answer,
            config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": "Kore"
                        }
                    }
                },
            },
        )
    )

    print(
        f"TTS generation: "
        f"{time.perf_counter() - tts_start:.3f}s"
    )


    # --------------------------------------------------------
    # GET PCM AUDIO
    # --------------------------------------------------------

    pcm_data = (
        tts_response
        .candidates[0]
        .content
        .parts[0]
        .inline_data
        .data
    )


    # --------------------------------------------------------
    # PCM → WAV
    # --------------------------------------------------------

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:

        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(24000)

        wav_file.writeframes(pcm_data)


    wav_buffer.seek(0)


    # --------------------------------------------------------
    # WAV → BASE64
    # --------------------------------------------------------

    audio_base64 = base64.b64encode(
        wav_buffer.read()
    ).decode("utf-8")


    # --------------------------------------------------------
    # TOTAL VOICE LATENCY
    # --------------------------------------------------------

    total_end = time.perf_counter()

    print("\n========================================")
    print(
        f"TOTAL /voice REQUEST: "
        f"{total_end - total_start:.3f}s"
    )
    print("========================================")


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "transcription": transcription,
        "answer": answer,
        "audio_base64": audio_base64,
    }



# ============================================================
# STREAMING SPEECH-TO-TEXT + LANGGRAPH + STREAMING TTS
# ============================================================

@app.websocket("/voice-stream")
async def voice_stream(websocket: WebSocket):

    await websocket.accept()

    thread_id = websocket.query_params.get("thread_id")

    if not thread_id:
        await websocket.close(code=1008)
        return

    send_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    final_event = asyncio.Event()

    final_transcripts = []
    tts_task = None
    tts_queue = None

    async def send_event(data: dict):
        """Safely send JSON events to React."""
        async with send_lock:
            try:
                await websocket.send_json(data)
            except (RuntimeError, WebSocketDisconnect):
                pass

    def add_transcription_segment(text: str):
        """Collect final transcription segments without duplication."""
        text = text.strip()

        if not text:
            return

        current = " ".join(final_transcripts).strip()

        if not current:
            final_transcripts.append(text)

        elif text == current:
            return

        elif text.startswith(current):
            final_transcripts.clear()
            final_transcripts.append(text)

        elif current.startswith(text):
            return

        elif text != final_transcripts[-1]:
            final_transcripts.append(text)

    try:
        print("\n========== STT STREAM STARTED ==========")

        async with client.aio.live.connect(
            model="gemini-3.5-transcribe-live",
            config=types.LiveConnectConfig(
                response_modalities=["TEXT"],
                input_audio_transcription=types.AudioTranscriptionConfig(
                    language_codes=[]
                ),
            ),
        ) as session:

            await send_event({"type": "ready"})
            print("Gemini Live connected.")

            # ------------------------------------------------
            # RECEIVE AUDIO FROM REACT
            # ------------------------------------------------

            async def forward_audio():
                try:
                    while True:
                        message = await websocket.receive()

                        if message["type"] == "websocket.disconnect":
                            break

                        if message.get("bytes") is not None:
                            audio_chunk = message["bytes"]

                            if audio_chunk:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=audio_chunk,
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )

                        elif message.get("text") is not None:
                            control = json.loads(message["text"])

                            if control.get("type") == "stop":
                                await session.send_realtime_input(
                                    audio_stream_end=True
                                )

                                print("Audio stream ended.")
                                return

                finally:
                    stop_event.set()

            # ------------------------------------------------
            # RECEIVE TRANSCRIPTION FROM GEMINI
            # ------------------------------------------------

            async def receive_transcription():
                async for response in session.receive():

                    content = response.server_content

                    if not content:
                        continue

                    interim = getattr(
                        content,
                        "interim_input_transcription",
                        None,
                    )

                    if interim and interim.text:
                        await send_event({
                            "type": "interim",
                            "text": interim.text,
                        })

                    final = getattr(
                        content,
                        "input_transcription",
                        None,
                    )

                    if final and final.text:
                        text = final.text.strip()

                        if text:
                            add_transcription_segment(text)

                            transcript = " ".join(final_transcripts)

                            print("Final transcription segment:", text)
                            print("Collected transcript:", transcript)

                            await send_event({
                                "type": "final",
                                "text": transcript,
                            })

                            final_event.set()

            sender_task = asyncio.create_task(forward_audio())
            receiver_task = asyncio.create_task(
                receive_transcription()
            )

            try:
                await stop_event.wait()

                await asyncio.wait_for(
                    final_event.wait(),
                    timeout=15,
                )

                # Allow late transcription segments to arrive.
                while True:
                    final_event.clear()

                    try:
                        await asyncio.wait_for(
                            final_event.wait(),
                            timeout=1.5,
                        )
                    except asyncio.TimeoutError:
                        break

            finally:
                for task in (sender_task, receiver_task):
                    if not task.done():
                        task.cancel()

                for task in (sender_task, receiver_task):
                    with suppress(asyncio.CancelledError, Exception):
                        await task

            # ------------------------------------------------
            # COMPLETE TRANSCRIPTION
            # ------------------------------------------------

            final_transcript = " ".join(final_transcripts).strip()

            print("\n" + "=" * 55)
            print("COMPLETE TRANSCRIPTION:")
            print(final_transcript)
            print("=" * 55)

            if not final_transcript:
                await send_event({
                    "type": "error",
                    "message": (
                        "No speech was recognized. Please try again."
                    ),
                })
                return

            
            # ------------------------------------------------
            # CARTESIA TTS QUEUE AND BACKGROUND WORKER
            # ------------------------------------------------

            tts_queue = asyncio.Queue()
            tts_buffer = ""
            tts_text_queued = False

            async def queue_tts_text(text: str, force=False):
                """
                Automatically collect streamed text into sentences.
                No manually written text segments are required.
                """

                nonlocal tts_buffer, tts_text_queued

                if text:
                    tts_buffer += text

                while tts_buffer:
                    # Find the first sentence-ending punctuation.
                    match = re.search(
                        r'[.!?।]["\']?(?:\s+|$)',
                        tts_buffer,
                    )

                    if match:
                        split_at = match.end()
                    elif len(tts_buffer) >= 180:
                        # Prevent very long sentences from waiting forever.
                        split_at = tts_buffer.rfind(" ", 0, 180)

                        if split_at <= 0:
                            split_at = min(180, len(tts_buffer))
                    elif force:
                        split_at = len(tts_buffer)
                    else:
                        break

                    segment = tts_buffer[:split_at].strip()
                    tts_buffer = tts_buffer[split_at:].lstrip()

                    if segment:
                        print(
                            f"[CARTESIA TEXT] {segment}",
                            flush=True,
                        )

                        await tts_queue.put(segment)
                        tts_text_queued = True

                if force:
                    tts_buffer = ""

            async def tts_worker():
                """
                One Cartesia WebSocket per complete response.
                Text is pushed progressively while audio is received.
                """

                started_at = time.perf_counter()
                first_audio = True
                audio_chunks = 0

                async with cartesia_client.tts.websocket_connect() as ws:

                    connected_at = time.perf_counter()

                    print(
                        "[CARTESIA] WebSocket connected in "
                        f"{connected_at - started_at:.3f}s",
                        flush=True,
                    )

                    ctx = ws.context(
                        model_id=CARTESIA_MODEL_ID,
                        voice=CARTESIA_VOICE_ID,
                        output_format={
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": CARTESIA_SAMPLE_RATE,
                        },
                        language="en",
                    )

                    await send_event({
                        "type": "tts_start",
                        "sample_rate": CARTESIA_SAMPLE_RATE,
                    })

                    async def receive_audio():
                        nonlocal first_audio, audio_chunks

                        async for event in ctx.receive():
                            if event.type != "chunk" or not event.audio:
                                continue

                            if first_audio:
                                first_audio = False

                                print(
                                    "[CARTESIA] First audio received in "
                                    f"{time.perf_counter() - connected_at:.3f}s "
                                    "after connection",
                                    flush=True,
                                )

                            audio_chunks += 1

                            async with send_lock:
                                await websocket.send_bytes(event.audio)

                    receiver = asyncio.create_task(receive_audio())

                    try:
                        while True:
                            segment = await tts_queue.get()

                            try:
                                if segment is None:
                                    break

                                # Send text over the existing connection.
                                await ctx.push(segment)

                            finally:
                                tts_queue.task_done()

                        # Tell Cartesia no more text is coming.
                        await ctx.no_more_inputs()

                        # Wait until the final audio chunks arrive.
                        await receiver

                    except Exception:
                        if not receiver.done():
                            receiver.cancel()

                        with suppress(asyncio.CancelledError, Exception):
                            await receiver

                        raise

                    # Now print the final audio chunk count.
                    print(
                        f"[CARTESIA] Audio chunks sent: {audio_chunks}",
                        flush=True,
                    )

                    await send_event({"type": "tts_end"})

                    print(
                        "[CARTESIA] Total TTS time: "
                        f"{time.perf_counter() - started_at:.3f}s",
                        flush=True,
                    )

            tts_task = asyncio.create_task(tts_worker())


            # ------------------------------------------------
            # LANGGRAPH STREAMING
            # ------------------------------------------------

            print("Starting LangGraph streaming...")

            graph_start = time.perf_counter()
            answer = ""
            streamed_mcp_text = False
            rag_response_queued = False

            async for mode, chunk in graph.astream(
                {
                    "message": final_transcript,
                    "route": "",
                    "response": "",
                    "messages": [],
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    }
                },
                stream_mode=["updates", "messages"],
            ):

                # --------------------------------------------
                # STREAM MCP LLM TEXT
                # --------------------------------------------

                if mode == "messages":
                    message_chunk, metadata = chunk
                    print(
                        "STREAMING NODE:",
                        metadata.get("langgraph_node"),
                        "TEXT:",
                        getattr(chunk[0], "content", ""),
                        flush=True,
                    )
                    if metadata.get("langgraph_node") != "mcp_agent":
                        continue

                    content = getattr(
                        message_chunk,
                        "content",
                        None,
                    )

                    text = ""

                    if isinstance(content, str):
                        text = content

                    elif isinstance(content, list):
                        text = "".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict)
                            and item.get("type") == "text"
                        )

                    if text:
                        streamed_mcp_text = True

                        # Send text to React immediately.
                        await send_event({
                            "type": "answer_chunk",
                            "text": text,
                        })

                        # Also feed text into the TTS queue.
                        await queue_tts_text(text)

                # --------------------------------------------
                # CAPTURE COMPLETED NODE RESPONSES
                # --------------------------------------------

                elif mode == "updates":

                    for node_name, node_update in chunk.items():

                        
                        # RAG-only response.
                        if (
                            node_name == "rag"
                            and node_update.get("route") == "rag"
                        ):
                            response_text = node_update.get("response", "")

                            if response_text and not rag_response_queued:
                                answer = response_text
                                rag_response_queued = True

                                print(
                                    "[RAG TTS] Queueing response once:",
                                    repr(response_text),
                                    flush=True,
                                )

                                await send_event({
                                    "type": "answer_chunk",
                                    "text": response_text,
                                })

                                await queue_tts_text(response_text)


                        # Blocked / unsupported question.
                        elif node_name == "blocked":
                            answer = node_update.get("response", "")

                            if answer:
                                await send_event({
                                    "type": "answer_chunk",
                                    "text": answer,
                                })

                                await queue_tts_text(answer)

                        # Save the authoritative final MCP response.
                        elif node_name == "mcp_response":
                            answer = node_update.get(
                                "response",
                                answer,
                            )

            graph_elapsed = time.perf_counter() - graph_start

            print(
                f"LangGraph streaming completed: "
                f"{graph_elapsed:.3f}s"
            )

            if not answer.strip():
                await send_event({
                    "type": "error",
                    "message": (
                        "Sorry, I could not generate a response. "
                        "Please try again."
                    ),
                })

                await tts_queue.put(None)
                await tts_task
                tts_task = None
                return

            # --------------------------------------------
            # FALLBACK IF NO TEXT WAS STREAMED
            # --------------------------------------------

            if not tts_text_queued and answer.strip():
                # Queue the final answer only if no route
                # already supplied text to the TTS queue.
                await queue_tts_text(answer)

            # Flush any incomplete sentence.
            await queue_tts_text("", force=True)

            # Tell React the complete answer is available.
            await send_event({
                "type": "answer",
                "transcription": final_transcript,
                "answer": answer,
            })

            # Finish queued TTS after all generated text
            # has been submitted.
            await tts_queue.put(None)
            await tts_task
            tts_task = None

            print("STT + LangGraph + TTS completed.")
            print("STT stream completed.")

    except asyncio.TimeoutError:
        await send_event({
            "type": "error",
            "message": (
                "Transcription timed out. Please try again."
            ),
        })

    except WebSocketDisconnect:
        print("Frontend disconnected.")

    except Exception as error:
        print("Voice streaming error:", repr(error))

        await send_event({
            "type": "error",
            "message": "Voice processing failed. Please try again.",
        })

    finally:
        stop_event.set()

        if tts_task is not None and not tts_task.done():
            tts_task.cancel()

            with suppress(asyncio.CancelledError, Exception):
                await tts_task

        with suppress(Exception):
            await websocket.close()




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

            print("\nDEBUG FINAL RESPONSE:", repr(final_response))
            print("DEBUG STREAMED TEXT:", repr(streamed_text))
            print("DEBUG DONE ANSWER:", repr(final_response or streamed_text))

            yield send_event(
                    "token",
                    {"text": final_response}
                )

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
