from typing import TypedDict
from typing_extensions import Annotated
import operator
import time

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
from rag import retrieve_faq

load_dotenv()


# ============================================
# MCP CLIENT AND TOOL CACHE
# ============================================

MCP_CLIENT = MultiServerMCPClient(
    {
        "commerce": {
            "transport": "sse",
            "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
        }
    }
)

MCP_TOOLS = None


async def initialize_mcp():
    global MCP_TOOLS

    if MCP_TOOLS is None:
        print("\nLoading MCP tools...")

        start = time.perf_counter()

        MCP_TOOLS = await MCP_CLIENT.get_tools()

        end = time.perf_counter()

        print(f"MCP get_tools(): {end - start:.3f}s")

        print("\nAVAILABLE TOOLS:")
        for tool in MCP_TOOLS:
            print(tool.name)


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


# ============================================
# MCP AGENT
# ============================================

async def ask_mcp_agent(messages):

    # Initialize MCP only if tools haven't
    # already been loaded.
    await initialize_mcp()

    tools = MCP_TOOLS

    # ========================================
    # Create Gemini model
    # ========================================

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    # Give Gemini information about the available tools
    model_with_tools = model.bind_tools(tools)

    # ========================================
    # Create messages for Gemini
    # ========================================

    mcp_messages = [
        SystemMessage(
            content=(
                "You are a commerce assistant. "
                "Never invent, guess, or assume values for tool arguments. "
                "Only use argument values that were explicitly provided by "
                "the user or are available in the conversation history. "
                "If a required tool argument is missing, ask the user for it "
                "instead of calling the tool with an invented value.\n\n"

                "If the user provides a missing value in a follow-up message, "
                "use the conversation history to understand what that value "
                "refers to and continue the previous request.\n\n"

                "If the request cannot be answered using any available tool, "
                "respond with exactly: UNSUPPORTED"
            )
        ),
        *messages
    ]

    # ========================================
    # First call to Gemini
    # ========================================

    print("\nMCP MESSAGES:")

    for message in mcp_messages:
        print(
            type(message).__name__,
            ":",
            message.content
        )

    start = time.perf_counter()

    response = await model_with_tools.ainvoke(mcp_messages)

    end = time.perf_counter()

    print(f"Gemini first call: {end - start:.3f}s")

    print("\nFIRST MODEL RESPONSE:")
    print(response)

    print("\nTOOL CALLS:")
    print(response.tool_calls)

    # ========================================
    # Tool calling loop
    # ========================================

    while response.tool_calls:

        # Add Gemini's response containing tool calls
        mcp_messages.append(response)

        # Handle all requested tools
        for tool_call in response.tool_calls:

            print("\nTOOL REQUESTED:")
            print(tool_call)

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print("\nTOOL NAME:")
            print(tool_name)

            print("\nTOOL ARGUMENTS:")
            print(tool_args)

            # Find the requested tool
            tool = next(
                tool
                for tool in tools
                if tool.name == tool_name
            )

            # ====================================
            # Execute MCP tool
            # ====================================

            start = time.perf_counter()

            tool_result = await tool.ainvoke(tool_args)

            end = time.perf_counter()

            print(
                f"MCP tool execution: "
                f"{end - start:.3f}s"
            )

            print("\nTOOL RESULT:")
            print(tool_result)

            # Add tool result to conversation
            mcp_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # ====================================
        # Send tool result back to Gemini
        # ====================================

        start = time.perf_counter()

        response = await model_with_tools.ainvoke(
            mcp_messages
        )

        end = time.perf_counter()

        print(
            f"Gemini final call: "
            f"{end - start:.3f}s"
        )

        print("\nMODEL RESPONSE AFTER TOOL:")
        print(response)

        print("\nTOOL CALLS AFTER TOOL:")
        print(response.tool_calls)

    # ========================================
    # Final response
    # ========================================

    print("\nFINAL RESPONSE:")
    print(response.content)

    return extract_text(response.content)


# ============================================
# LANGGRAPH NODES
# ============================================

def add_user_message(state: State):

    return {
        "messages": [
            HumanMessage(
                content=state["message"]
            )
        ]
    }


def rag_node(state: State):

    question = state["message"]

    # Order-specific requests should go directly to MCP.
    order_specific_keywords = [
        "order",
        "order id",
        "order number",
        "customer id",
    ]

    if any(
        keyword in question.lower()
        for keyword in order_specific_keywords
    ):
        print("RAG: Order-specific query → MCP")
        return {
            "route": "mcp",
        }

    answer = retrieve_faq(question)

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

    answer = await ask_mcp_agent(
        state["messages"]
    )

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


# ============================================
# ROUTING
# ============================================

def route_after_rag(state: State):
    return state["route"]


def route_after_mcp(state: State):
    return state["route"]


# ============================================
# BUILD GRAPH
# ============================================

graph_builder = StateGraph(State)

graph_builder.add_node(
    "add_message",
    add_user_message
)

graph_builder.add_node(
    "rag",
    rag_node
)

graph_builder.add_node(
    "mcp",
    mcp_node
)

graph_builder.add_node(
    "unsupported",
    unsupported_node
)

graph_builder.add_edge(
    START,
    "add_message"
)

graph_builder.add_edge(
    "add_message",
    "rag"
)

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

graph_builder.add_edge(
    "unsupported",
    END
)


# ============================================
# CHECKPOINTER
# ============================================

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)