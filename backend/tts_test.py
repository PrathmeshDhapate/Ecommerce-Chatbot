import os
import base64
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load .env
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")

print("API key loaded:", api_key is not None)


# Create Gemini client
client = genai.Client(
    api_key=api_key
)

print("Gemini client created.")


# Generate speech
interaction = client.interactions.create(
    model="gemini-3.1-flash-tts-preview",
    input="Hello! This is a test of Gemini text to speech.",
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

print("TTS response received.")


# Decode audio
audio_data = base64.b64decode(
    interaction.output_audio.data
)

print(
    "Audio bytes:",
    len(audio_data)
)


# Save as WAV
with wave.open("tts_output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(audio_data)


print("TTS audio generated successfully.")
print("Saved as: tts_output.wav")