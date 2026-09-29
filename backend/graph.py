from typing import TypedDict
from typing_extensions import Annotated
import operator
import time
import asyncio

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

    print("MCP_TOOLS BEFORE:", MCP_TOOLS is None)

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
# MCP AGENT NODE
# ============================================

async def mcp_agent_node(state: State):

    await initialize_mcp()

    tools = MCP_TOOLS

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    model_with_tools = model.bind_tools(tools)

    mcp_messages = [
        SystemMessage(
            content=(
                """You are a commerce assistant that MUST use the available MCP tools
                for order-specific information.

                STRICT ORDER RULES:

                1. For ANY question about a specific order, you MUST use the
                appropriate MCP tool. Never answer an order-specific question
                from general knowledge or from assumptions.

                2. NEVER invent, guess, infer, or fabricate tool arguments.

                3. For get_order_status, use the customer_id and order_id exactly
                as provided by the user or as explicitly available in the conversation
                history.

                4. If get_order_status requires a customer_id and the customer_id
                is NOT available in the current user message or conversation history,
                DO NOT call the tool. Ask the user for the customer ID.

                5. An order ID by itself does NOT give you permission to invent
                or assume a customer ID.

                6. Do not assume that different orders belong to the same customer.
                Each order must be associated with its explicitly provided customer ID.

                7. If multiple orders are requested, determine the customer ID
                for EACH order separately. Never apply one customer ID to another
                order unless the user explicitly says they belong to the same customer.

                8. If the user refers to an order indirectly using words such as
                "it", "this", "that order", "my order", or "the order", resolve
                the reference only if a specific order is clearly identifiable
                from the conversation history.

                9. If an order reference cannot be resolved unambiguously,
                ask the user for clarification instead of guessing.

                10. After a tool returns a result, use the actual tool result.
                Never replace, modify, or contradict values returned by the tool.

                11. For shipping, payment, tracking, returns, cancellation,
                exchanges, warranty, size, account, loyalty, product care,
                support, and other commerce requests, use the appropriate MCP
                tool whenever the required information is available.

                12. For multiple independent tool calls, request all required
                tools and allow the tool executor to execute them concurrently.

                13. If a required argument is genuinely unavailable, ask the user
                for that argument.

                14. If the request cannot be answered using the available MCP tools,
                respond with exactly: UNSUPPORTED

                IMPORTANT:
                The conversation history is context, not permission to guess.
                Only use information that is explicitly present in the conversation.

                Never fabricate order status, customer IDs, tracking numbers,
                dates, prices, or any other order information."""
            )
        ),
        *state["messages"]
    ]

    start = time.perf_counter()

    response = await model_with_tools.ainvoke(
        mcp_messages
    )

    end = time.perf_counter()

    print(
        f"Gemini MCP call: {end - start:.3f}s"
    )

    if response.tool_calls:
        return {
            "messages": [response]
        }

    return {
        "messages": [response],
        "response": extract_text(response.content)
    }

# ============================================
# MCP TOOL EXECUTOR
# ============================================

async def tool_executor_node(state: State):

    await initialize_mcp()

    tools = MCP_TOOLS

    # Get the latest Gemini message
    last_message = state["messages"][-1]

    tool_tasks = []

    for tool_call in last_message.tool_calls:

        tool = next(
            tool
            for tool in tools
            if tool.name == tool_call["name"]
        )

        tool_tasks.append(
            tool.ainvoke(tool_call["args"])
        )

    print("\nExecuting MCP tools concurrently...")

    start = time.perf_counter()

    tool_results = await asyncio.gather(
        *tool_tasks
    )

    end = time.perf_counter()

    print(
        f"MCP tools total execution: {end - start:.3f}s"
    )

    tool_messages = []

    for tool_call, tool_result in zip(
        last_message.tool_calls,
        tool_results
    ):

        print("\nTOOL RESULT:")
        print(tool_result)

        tool_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": tool_messages
    }

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

async def router_node(state: State):

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    router_prompt = """
You are a routing classifier for a commerce customer-support chatbot.

Decide whether the user's latest message should be handled by:

RAG:
- General FAQ questions
- Return policy questions
- Shipping policy questions
- Payment information questions
- Warranty information questions
- Size guide questions
- General commerce information

MCP:
- Questions about a specific order
- Order status
- Tracking
- Cancelling an order
- Processing a return for a specific order
- Processing an exchange for a specific order
- Any follow-up question referring to a specific order from the conversation history
- Messages containing references such as "it", "this", "that order", "my order", "the order",
  but ONLY when the conversation history contains a specific order or order-related context

IMPORTANT:
Use the conversation history to understand references such as
"it", "this", "that", or "the order".

For example:

Previous message:
"What is the status of order ORD-1003-D?"

Current message:
"Can I return this?"

The correct route is MCP.

However:

"What is your return policy?"

should be routed to RAG because it is a general policy question.

Return ONLY one word:

rag

or

mcp

IMPORTANT:

Do NOT route a message to MCP merely because it contains
"it", "this", or "that".

If there is no specific order or order-related context
in the conversation history, treat the message as a general
commerce/return question and route it to RAG.

Example:

New conversation:
"Can I return this?"

Correct route: rag

Conversation:
User: "What is the status of order ORD-1003-D?"
Assistant: "Order ORD-1003-D was delivered."

User: "Can I return this?"

Correct route: mcp
"""

    router_messages = [
        SystemMessage(content=router_prompt),
        *state["messages"]
    ]

    response = await model.ainvoke(router_messages)

    route = extract_text(response.content).strip().lower()

    if route not in ["rag", "mcp"]:
        route = "rag"

    print(f"ROUTER: {route.upper()}")

    return {
        "route": route
    }

def rag_node(state: State):

    question = state["message"]

    answer = retrieve_faq(question)

    if answer is not None:
        print("RAG: FAQ FOUND")

        return {
            "route": "faq_found",
            "response": answer,
            "messages": [
                AIMessage(content=answer)
            ]
        }

    print("RAG: FAQ NOT FOUND")

    return {
        "route": "faq_not_found"
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

def route_after_router(state: State):
    return state["route"]

def route_after_rag(state: State):
    return state["route"]




def route_after_mcp_agent(state: State):

    last_message = state["messages"][-1]

    # Gemini requested one or more tools
    if last_message.tool_calls:
        return "tools"

    # Gemini did not request a tool
    answer = extract_text(last_message.content)

    if answer.strip() == "UNSUPPORTED":
        return "unsupported"

    return "final"


# ============================================
# BUILD GRAPH
# ============================================

graph_builder = StateGraph(State)

graph_builder.add_node(
    "add_message",
    add_user_message
)

graph_builder.add_node(
    "router",
    router_node
)

graph_builder.add_node(
    "rag",
    rag_node
)

graph_builder.add_node(
    "mcp_agent",
    mcp_agent_node
)

graph_builder.add_node(
    "tool_executor",
    tool_executor_node
)

graph_builder.add_node(
    "unsupported",
    unsupported_node
)

graph_builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "rag": "rag",
        "mcp": "mcp_agent"
    }
)

graph_builder.add_edge(
    "tool_executor",
    "mcp_agent"
)

graph_builder.add_conditional_edges(
    "mcp_agent",
    route_after_mcp_agent,
    {
        "tools": "tool_executor",
        "final": END,
        "unsupported": "unsupported"
    }
)

graph_builder.add_edge(
    START,
    "add_message"
)

graph_builder.add_edge(
    "add_message",
    "router"
)

graph_builder.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "faq_found": END,
        "faq_not_found": "mcp_agent"
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