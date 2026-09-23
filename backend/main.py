
import base64
import io
import wave

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

    print("Thread ID:", thread_id)
    print("Filename:", audio.filename)
    print("Content type:", audio.content_type)

    audio_data = await audio.read()

    print(
        "Audio size:",
        len(audio_data),
        "bytes"
    )


    # --------------------------------------------------
    # 1. SPEECH TO TEXT
    # --------------------------------------------------

    audio_base64 = base64.b64encode(
        audio_data
    ).decode("utf-8")


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


    transcription = interaction.output_text


    print("\nTRANSCRIPTION:")
    print(transcription)


    # --------------------------------------------------
    # 2. LANGGRAPH
    # --------------------------------------------------

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


    answer = result["response"]


    print("\nLANGGRAPH RESPONSE:")
    print(answer)


    # --------------------------------------------------
    # 3. TEXT TO SPEECH
    # --------------------------------------------------

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


    # Gemini returns raw audio data.
    # Convert it into actual WAV bytes.

    pcm_audio = base64.b64decode(
        tts_interaction.output_audio.data
    )


    wav_buffer = io.BytesIO()


    with wave.open(wav_buffer, "wb") as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(24000)

        wf.writeframes(pcm_audio)


    wav_bytes = wav_buffer.getvalue()


    # Convert WAV bytes to Base64
    # so React can receive them in JSON.

    wav_base64 = base64.b64encode(
        wav_bytes
    ).decode("utf-8")


    print(
        "\nTTS WAV SIZE:",
        len(wav_bytes),
        "bytes"
    )


    # --------------------------------------------------
    # 4. RETURN EVERYTHING TO REACT
    # --------------------------------------------------

    return {
        "transcription": transcription,
        "answer": answer,
        "audio_base64": wav_base64
    }

