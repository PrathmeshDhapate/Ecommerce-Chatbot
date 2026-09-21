from typing import TypedDict
from rag import retrieve_faq

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieve_faq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


class State(TypedDict):
    message: str
    context: str
    response: str

def chatbot_node(state: State):

    context = retrieve_faq(state["message"])

    if not context:
        return {
            "context": "",
            "response": "I couldn't find relevant information in our FAQ."
        }

    return {
        "context": context,
        "response": context
    }


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()