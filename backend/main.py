from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI

from graph import graph


app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Backend is running"}


@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({
        "message": request.message,
        "response": ""
    })

    return {
        "answer": result["response"]
    }