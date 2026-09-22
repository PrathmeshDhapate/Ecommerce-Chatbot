from typing import TypedDict
from typing_extensions import Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


from dotenv import load_dotenv
from rag import retrieve_faq

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    message: str
    route: str
    response: str


def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return str(content)


async def ask_mcp_agent(messages):

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
            }
        }
    )

    tools = await client.get_tools()

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    agent = create_agent(
        model,
        tools
    )

    result = await agent.ainvoke(
        {
            "messages": messages
        }
    )

    return extract_text(result["messages"][-1].content)


def add_user_message(state: State):

    return {
        "messages": [
            HumanMessage(content=state["message"])
        ]
    }


def router_node(state: State):

    question = state["message"].lower()

    if any(word in question for word in [
        "cancel my order",
        "cancel order",
        "track my order",
        "track my package",
        "where is my order",
        "order status",
        "status of order",
        "return my order",
        "return this order",
        "refund my order"
    ]):
        return {
            "route": "mcp"
        }

    return {
        "route": "rag"
    }


async def mcp_node(state: State):

    answer = await ask_mcp_agent(state["messages"])

    return {
        "response": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }

def rag_node(state: State):

    answer = retrieve_faq(state["message"])

    return {
        "response": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }


graph_builder = StateGraph(State)

graph_builder.add_node("add_message", add_user_message)
graph_builder.add_node("router", router_node)
graph_builder.add_node("mcp", mcp_node)
graph_builder.add_node("rag", rag_node)

graph_builder.add_edge(START, "add_message")
graph_builder.add_edge("add_message", "router")


def route_question(state: State):

    return state["route"]


graph_builder.add_conditional_edges(
    "router",
    route_question,
    {
        "mcp": "mcp",
        "rag": "rag"
    }
)

graph_builder.add_edge("mcp", END)
graph_builder.add_edge("rag", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)