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