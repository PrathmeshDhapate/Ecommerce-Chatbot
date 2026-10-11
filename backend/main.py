import base64
import io
import time
import wave
import asyncio
import json

from contextlib import suppress
from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

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

from graph import (
    graph,
    initialize_mcp,
)

from dotenv import load_dotenv


load_dotenv()


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
# STREAMING SPEECH-TO-TEXT
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

    # Store the final transcription segments instead of
    # replacing the previous segment every time.
    final_transcripts = []

    async def send_event(data: dict):
        """Send an event safely to the React frontend."""
        async with send_lock:
            try:
                await websocket.send_json(data)
            except (RuntimeError, WebSocketDisconnect):
                pass

    def add_transcription_segment(text: str):
        """
        Collect final transcription text.
        Handle distinct segments and cumulative transcripts.
        """
        text = text.strip()

        if not text:
            return

        current = " ".join(final_transcripts).strip()

        if not current:
            final_transcripts.append(text)

        elif text == current:
            # Gemini repeated the complete transcript.
            return

        elif text.startswith(current):
            # Gemini sent the complete transcript again,
            # including the previously received text.
            final_transcripts.clear()
            final_transcripts.append(text)

        elif current.startswith(text):
            # The new text is already contained in the current
            # complete transcript.
            return

        elif text != final_transcripts[-1]:
            # Gemini sent another distinct final segment.
            final_transcripts.append(text)

    try:
        print("\n========== STT STREAM STARTED ==========")

        async with client.aio.live.connect(
            model="gemini-3.5-transcribe-live",
            config=types.LiveConnectConfig(
                response_modalities=["TEXT"],
                input_audio_transcription=(
                    types.AudioTranscriptionConfig(
                        language_codes=[]
                    )
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

                        # Raw 16 kHz PCM audio from React.
                        if message.get("bytes") is not None:
                            audio_chunk = message["bytes"]

                            if audio_chunk:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=audio_chunk,
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )

                        # Stop recording.
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

                    # Interim transcription while the user speaks.
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

                    # Final transcription segment.
                    final = getattr(
                        content,
                        "input_transcription",
                        None,
                    )

                    if final and final.text:
                        text = final.text.strip()

                        if text:
                            add_transcription_segment(text)

                            print(
                                "Final transcription segment:",
                                text,
                            )

                            print(
                                "Collected transcript:",
                                " ".join(final_transcripts),
                            )

                            # Notify the frontend, but do not
                            # treat the first segment as the end
                            # of the complete transcription.
                            await send_event({
                                "type": "final",
                                "text": " ".join(final_transcripts),
                            })

                            final_event.set()

            # Run both tasks concurrently.
            sender_task = asyncio.create_task(forward_audio())
            receiver_task = asyncio.create_task(
                receive_transcription()
            )

            try:
                # Wait until the user presses Stop.
                await stop_event.wait()

                # Wait for at least one final transcription.
                await asyncio.wait_for(
                    final_event.wait(),
                    timeout=15,
                )

                # Wait until no new final segment arrives for
                # 1.5 seconds. This allows late segments to arrive.
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
            # COMBINE ALL TRANSCRIPTION SEGMENTS
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
                        "No speech was recognized. "
                        "Please try again."
                    ),
                })
                return

            # ------------------------------------------------
            # SEND COMPLETE QUESTION TO LANGGRAPH
            # ------------------------------------------------

            print("Sending complete transcript to LangGraph...")

            result = await graph.ainvoke(
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
            )

            answer = result["response"]

            # ------------------------------------------------
            # SEND TEXT ANSWER AND STREAM TTS
            # ------------------------------------------------

            print("Sending text answer to React...")

            await send_event({
                "type": "answer",
                "transcription": final_transcript,
                "answer": answer,
            })

            print("Starting TTS streaming...")

            await stream_tts_audio(answer, websocket)

            print("STT + LangGraph + TTS completed.")
            print("STT stream completed.")

    except asyncio.TimeoutError:

        await send_event({
            "type": "error",
            "message": (
                "Transcription timed out. "
                "Please try again."
            ),
        })

    except WebSocketDisconnect:
        print("Frontend disconnected.")

    except Exception as error:
        print("STT streaming error:", repr(error))

        await send_event({
            "type": "error",
            "message": (
                "Voice transcription failed. "
                "Please try again."
            ),
        })

    finally:
        stop_event.set()

        with suppress(Exception):
            await websocket.close()



TTS_MODEL = "gemini-3.1-flash-tts-preview"


async def stream_tts_audio(text: str, websocket):
    """
    Generate speech from text and send audio chunks
    to the frontend through WebSocket.
    """

    if not text.strip():
        return

    start_time = time.perf_counter()
    first_chunk = True

    # Tell the frontend that audio streaming is starting.
    await websocket.send_json({
        "type": "tts_start",
        "sample_rate": 24000
    })

    try:
        # Run the synchronous streaming API outside the event loop.
        def generate_audio():
            return client.interactions.create(
                model=TTS_MODEL,
                input=text,
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [
                        {"voice": "Kore"}
                    ]
                },
                stream=True,
            )
        print("TTS: Starting generation request...", flush=True)
        events = await asyncio.to_thread(generate_audio)
        try:
            events = await asyncio.wait_for(
                asyncio.to_thread(generate_audio),
                timeout=30
            )
        except asyncio.TimeoutError:
            print(
                "TTS ERROR: Generation request timed out after 30 seconds.",
                flush=True
            )
            await websocket.send_json({
                "type": "tts_error",
                "message": "Speech generation timed out."
            })
            return
        print("TTS: Generation request returned.", flush=True)
        for event in events:
            print(
                "TTS event received:",
                getattr(event, "event_type", "unknown"),
                flush=True,
            )
            if getattr(event, "event_type", None) != "step.delta":
                continue

            delta = getattr(event, "delta", None)

            if getattr(delta, "type", None) != "audio":
                continue

            audio_data = getattr(delta, "data", None)

            if not audio_data:
                continue

            # Decode audio if the SDK provides base64 text.
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)

            if first_chunk:
                first_chunk = False
                print(
                    "TTS first chunk:",
                    round(time.perf_counter() - start_time, 3),
                    "seconds"
                )

            # Send raw audio bytes as a WebSocket binary message.
            await websocket.send_bytes(audio_data)

        await websocket.send_json({"type": "tts_end"})

        print(
            "TTS streaming completed:",
            round(time.perf_counter() - start_time, 3),
            "seconds"
        )

    except Exception as error:
        print("TTS streaming error:", repr(error))

        await websocket.send_json({
            "type": "tts_error",
            "message": "Could not generate speech."
        })