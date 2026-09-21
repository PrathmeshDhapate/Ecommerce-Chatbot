from fastapi import FastAPI
from pydantic import BaseModel


from graph import graph

app = FastAPI()



class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Backend is running"}


@app.post("/chat")
async def chat(request: ChatRequest):

    result = await graph.ainvoke({
        "message": request.message,
        "route": "",
        "response": ""
    })

    return {
        "answer": result["response"]
    }