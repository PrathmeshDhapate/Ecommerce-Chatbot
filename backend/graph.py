from typing import TypedDict
from typing_extensions import Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
        model="gemini-3.5-flash-lite"
    )

    agent = create_agent(
        model,
        tools
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Answer the user's request using the available tools. "
                        "If the request cannot be answered using any available "
                        "tool, respond with exactly: UNSUPPORTED\n\n"
                        f"User request: {messages[-1].content}"
                    )
                )
            ]
        }
    )

    return extract_text(result["messages"][-1].content)


def add_user_message(state: State):

    return {
        "messages": [
            HumanMessage(content=state["message"])
        ]
    }


def rag_node(state: State):

    answer = retrieve_faq(state["message"])

    if answer is not None:
        return {
            "route": "faq_found",
            "response": answer,
            "messages": [
                AIMessage(content=answer)
            ]
        }

    return {
        "route": "mcp",
    }


async def mcp_node(state: State):

    answer = await ask_mcp_agent(state["messages"])

    if answer.strip() == "UNSUPPORTED":
        return {
            "route": "unsupported",
        }

    return {
        "route": "mcp_found",
        "response": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }


def unsupported_node(state: State):

    answer = (
        "Sorry, I can't help with that request. "
        "I can help with commerce-related questions such as "
        "orders, returns, shipping, payments, and related FAQs."
    )

    return {
        "response": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }


def route_after_rag(state: State):
    return state["route"]


def route_after_mcp(state: State):
    return state["route"]


graph_builder = StateGraph(State)

graph_builder.add_node("add_message", add_user_message)
graph_builder.add_node("rag", rag_node)
graph_builder.add_node("mcp", mcp_node)
graph_builder.add_node("unsupported", unsupported_node)

graph_builder.add_edge(START, "add_message")
graph_builder.add_edge("add_message", "rag")

graph_builder.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "faq_found": END,
        "mcp": "mcp"
    }
)

graph_builder.add_conditional_edges(
    "mcp",
    route_after_mcp,
    {
        "mcp_found": END,
        "unsupported": "unsupported"
    }
)

graph_builder.add_edge("unsupported", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)