import base64
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google import genai

import tempfile
import os
import time

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
async def voice(audio: UploadFile = File(...)):

    print("Filename:", audio.filename)
    print("Content type:", audio.content_type)

    audio_data = await audio.read()

    print(
        "Audio size:",
        len(audio_data),
        "bytes"
    )

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

    print("\nTRANSCRIPTION:")
    print(interaction.output_text)

    return {
        "transcription": interaction.output_text
    }

