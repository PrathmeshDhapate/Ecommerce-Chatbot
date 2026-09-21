from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from rag import retrieve_faq

from dotenv import load_dotenv

load_dotenv()


# -------------------------
# 1. State
# -------------------------

class State(TypedDict):
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

# -------------------------
# 2. MCP Agent
# -------------------------

async def ask_mcp_agent(question: str):

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
                ("user", question)
            ]
        }
    )

    return extract_text(result["messages"][-1].content)

# -------------------------
# Router Node for MCP or RAG
# -------------------------

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
# -------------------------
# 3. LangGraph node
# -------------------------

async def mcp_node(state: State):

    answer = await ask_mcp_agent(state["message"])

    return {
        "response": answer
    }

# -------------------------
#  RAG Node
# -------------------------

def rag_node(state: State):

    answer = retrieve_faq(state["message"])

    return {
        "response": answer
    }


# -------------------------
# 4. Build graph
# -------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("router", router_node)
graph_builder.add_node("mcp", mcp_node)
graph_builder.add_node("rag", rag_node)

graph_builder.add_edge(START, "router")

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
graph = graph_builder.compile()