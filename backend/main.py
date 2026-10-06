import base64
import io
import time
import wave

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
)

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
            "messages": []
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