from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from graph import graph

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
    return {"message": "Backend is running"}


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