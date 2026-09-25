
import base64
import io
import wave
import time

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google import genai

from graph import graph
from dotenv import load_dotenv


load_dotenv()


client = genai.Client()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str


@app.get("/")
def home():
    return {
        "message": "Backend is running"
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    result = await graph.ainvoke(
        {
            "message": request.message,
            "route": "",
            "response": ""
        },
        config={
            "configurable": {
                "thread_id": request.thread_id
            }
        }
    )

    return {
        "answer": result["response"]
    }


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    thread_id: str = Form(...)
):

    # ==================================================
    # TOTAL VOICE REQUEST TIMER
    # ==================================================

    total_start = time.perf_counter()

    print("\n========================================")
    print("VOICE REQUEST STARTED")
    print("========================================")


    # ==================================================
    # 1. READ AUDIO
    # ==================================================

    start = time.perf_counter()

    audio_data = await audio.read()

    end = time.perf_counter()

    print(
        f"\n1. Audio read: "
        f"{end - start:.3f}s"
    )

    print(
        "Audio size:",
        len(audio_data),
        "bytes"
    )


    # ==================================================
    # 2. AUDIO → BASE64
    # ==================================================

    start = time.perf_counter()

    audio_base64 = base64.b64encode(
        audio_data
    ).decode("utf-8")

    end = time.perf_counter()

    print(
        f"2. Input Base64 encoding: "
        f"{end - start:.3f}s"
    )


    # ==================================================
    # 3. SPEECH TO TEXT
    # ==================================================

    start = time.perf_counter()

    interaction = client.interactions.create(
        model="gemini-3.5-transcribe",
        input=[
            {
                "type": "audio",
                "data": audio_base64,
                "mime_type": "audio/webm",
            }
        ],
    )

    end = time.perf_counter()

    print(
        f"3. Gemini STT: "
        f"{end - start:.3f}s"
    )


    transcription = interaction.output_text

    print("\nTRANSCRIPTION:")
    print(transcription)


    # ==================================================
    # 4. LANGGRAPH
    # ==================================================

    start = time.perf_counter()

    result = await graph.ainvoke(
        {
            "message": transcription,
            "route": "",
            "response": ""
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    end = time.perf_counter()

    print(
        f"\n4. LangGraph total: "
        f"{end - start:.3f}s"
    )


    answer = result["response"]

    print("\nLANGGRAPH RESPONSE:")
    print(answer)


    # ==================================================
    # 5. TEXT TO SPEECH
    # ==================================================

    start = time.perf_counter()

    tts_interaction = client.interactions.create(
        model="gemini-3.1-flash-tts-preview",
        input=answer,
        response_format={
            "type": "audio"
        },
        generation_config={
            "speech_config": [
                {
                    "voice": "Kore"
                }
            ]
        }
    )

    end = time.perf_counter()

    print(
        f"\n5. Gemini TTS: "
        f"{end - start:.3f}s"
    )


    # ==================================================
    # 6. DECODE PCM AUDIO
    # ==================================================

    start = time.perf_counter()

    pcm_audio = base64.b64decode(
        tts_interaction.output_audio.data
    )

    end = time.perf_counter()

    print(
        f"6. PCM Base64 decoding: "
        f"{end - start:.3f}s"
    )


    # ==================================================
    # 7. PCM → WAV
    # ==================================================

    start = time.perf_counter()

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(24000)

        wf.writeframes(pcm_audio)

    wav_bytes = wav_buffer.getvalue()

    end = time.perf_counter()

    print(
        f"7. PCM → WAV conversion: "
        f"{end - start:.3f}s"
    )


    # ==================================================
    # 8. WAV → BASE64
    # ==================================================

    start = time.perf_counter()

    wav_base64 = base64.b64encode(
        wav_bytes
    ).decode("utf-8")

    end = time.perf_counter()

    print(
        f"8. Output Base64 encoding: "
        f"{end - start:.3f}s"
    )


    print(
        "\nWAV SIZE:",
        len(wav_bytes),
        "bytes"
    )


    # ==================================================
    # 9. TOTAL VOICE LATENCY
    # ==================================================

    total_end = time.perf_counter()

    print("\n========================================")
    print(
        f"TOTAL VOICE REQUEST: "
        f"{total_end - total_start:.3f}s"
    )
    print("========================================")


    # ==================================================
    # 10. RETURN TO REACT
    # ==================================================

    return {
        "transcription": transcription,
        "answer": answer,
        "audio_base64": wav_base64
    }