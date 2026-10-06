from typing import TypedDict
from typing_extensions import Annotated

import operator
import time
from langchain_core.callbacks import BaseCallbackHandler

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from rag import retrieve_faq_for_router


load_dotenv()


# ============================================================
# MCP CLIENT
# ============================================================

MCP_CLIENT = MultiServerMCPClient(
    {
        "commerce": {
            "transport": "sse",
            "url": (
                "https://slavpilus-mpc-for-commerce-platforms.hf.space"
                "/gradio_api/mcp/sse"
            ),
        }
    }
)


# ============================================================
# CACHED MCP COMPONENTS
# ============================================================

MCP_TOOLS = None
MCP_AGENT = None


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        operator.add
    ]

    message: str
    route: str
    response: str


# ============================================================
# HELPER
# ============================================================

def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            )
        )

    return str(content)


# ============================================================
# LLM #1 — RAG JUDGE
# ============================================================

RAG_JUDGE_PROMPT = """
You are a routing judge for a commerce FAQ chatbot.

Your ONLY job is to decide whether the retrieved FAQ answer
is sufficient to answer the user's current question.

You must choose exactly ONE label:

RAG
NOT_RAG

Use RAG when:
- The user is asking for general information or a policy.
- The retrieved FAQ directly answers the user's question.
- The retrieved FAQ provides enough information to answer the question
  without needing a live system or an MCP tool.
- The question is about topics covered by the retrieved FAQ.

Examples of questions that should usually be RAG:
- "What is the return policy?"
- "How long does shipping take?"
- "What payment methods do you accept?"
- "What is your warranty policy?"
- "How do I track a package?" 
  (when the FAQ provides general tracking instructions)

Use NOT_RAG when:
- The user asks about a specific order, customer, package, or account.
- The user needs live or real-time information.
- The user wants to perform an action.
- The user asks to cancel an order.
- The user asks to process a return for a specific order.
- The user asks for the current status/location of a specific package.
- The retrieved FAQ does not actually answer the question.
- The retrieved FAQ answer is empty.
- The user asks something unrelated to the retrieved FAQ.

IMPORTANT:
Do NOT classify a question as NOT_RAG merely because it is about
returns, shipping, payments, tracking, or another commerce topic.

The key question is:

"Can the retrieved FAQ answer the user's question directly,
without calling an MCP tool?"

If YES → RAG
If NO → NOT_RAG

Examples:

User: "What is the return policy?"
FAQ: "Most products can be returned within 30 days of delivery."
→ RAG

User: "Can I return order ORD-1005-D?"
FAQ: "Most products can be returned within 30 days of delivery."
→ NOT_RAG

User: "How long does shipping take?"
FAQ: "Orders usually arrive within 5–7 business days."
→ RAG

User: "Where is my order ORD-1005-D?"
FAQ: "Orders usually arrive within 5–7 business days."
→ NOT_RAG

User: "Can you cancel order ORD-1005-D?"
FAQ: "Most products can be returned within 30 days of delivery."
→ NOT_RAG

Do not answer the user's question.
Do not explain your decision.
Do not return JSON.
Do not return a list.
Do not return any additional text.

Return exactly one of:

RAG

NOT_RAG
"""


# ============================================================
# LLM #2 — MCP AGENT
# ============================================================

MCP_SYSTEM_PROMPT = """
You are a commerce support agent.

You have access to MCP tools for handling commerce-related requests.

IMPORTANT RULES:

1. Use an MCP tool when the user's request can be handled by one
   of the available tools.

2. Never invent, guess, or fabricate customer IDs.

3. If the user explicitly provides a customer ID, use exactly that
   customer ID.

4. If a tool requires a customer ID but the user has not provided one,
   do not invent one such as:

   - "unknown"
   - "default_customer"
   - "CUST-001"
   - any other guessed value

   Ask the user for the customer ID instead.

5. For unsupported requests, do not call an unrelated MCP tool.

   Clearly explain that you can only help with commerce-related
   requests.

6. Always use the actual MCP tool result when generating the final
   answer.

7. Never fabricate information that was not returned by an MCP tool.

8. For order-specific questions, always use the appropriate MCP tool.

9. For cancellation requests, if the cancellation reason is required
   and the user has not provided one, ask the user for the reason
   instead of inventing one.

10. If a required argument is genuinely unavailable, ask the user
    for that argument.

11. If the request cannot be answered using the available MCP tools,
    clearly explain that the request is outside the supported
    commerce capabilities.

12. Never fabricate order status, customer IDs, tracking numbers,
    dates, prices, refund information, or any other commerce data.
"""

class MCPTimingCallback(BaseCallbackHandler):

    def __init__(self):
        self.starts = {}

    def on_chat_model_start(
        self,
        serialized,
        messages,
        run_id,
        **kwargs
    ):
        self.starts[run_id] = time.perf_counter()

        print("\n>>> MCP AGENT LLM START")

    def on_llm_end(
        self,
        response,
        run_id,
        **kwargs
    ):
        if run_id in self.starts:
            elapsed = time.perf_counter() - self.starts.pop(run_id)

            print(
                f">>> MCP AGENT LLM END: "
                f"{elapsed:.3f}s"
            )

    def on_tool_start(
        self,
        serialized,
        input_str,
        run_id,
        **kwargs
    ):
        self.starts[run_id] = time.perf_counter()

        tool_name = serialized.get(
            "name",
            "unknown"
        )

        print(
            f"\n>>> MCP TOOL START: "
            f"{tool_name}"
        )

    def on_tool_end(
        self,
        output,
        run_id,
        **kwargs
    ):
        if run_id in self.starts:
            elapsed = time.perf_counter() - self.starts.pop(run_id)

            print(
                f">>> MCP TOOL END: "
                f"{elapsed:.3f}s"
            )

# ============================================================
# MCP INITIALIZATION
# ============================================================

async def initialize_mcp():

    global MCP_TOOLS
    global MCP_AGENT

    # Already initialized
    if MCP_AGENT is not None:

        print("MCP agent already initialized. Reusing it.")

        return

    print("\n============================================================")
    print("INITIALIZING MCP")
    print("============================================================")

    # --------------------------------------------------------
    # Load MCP tools
    # --------------------------------------------------------

    print("\nLoading MCP tools...")

    start = time.perf_counter()

    MCP_TOOLS = await MCP_CLIENT.get_tools()

    end = time.perf_counter()

    print(
        f"MCP get_tools(): "
        f"{end - start:.3f}s"
    )

    print("\nAVAILABLE TOOLS:")

    for tool in MCP_TOOLS:
        print(tool.name)

    # --------------------------------------------------------
    # Create Gemini model
    # --------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    # --------------------------------------------------------
    # Create MCP agent ONCE
    # --------------------------------------------------------

    MCP_AGENT = create_agent(
        model=llm,
        tools=MCP_TOOLS,
        system_prompt=MCP_SYSTEM_PROMPT
    )

    print("\nMCP agent initialized.")


# ============================================================
# NODE 1 — ADD USER MESSAGE
# ============================================================

def add_user_message(state: State):

    return {
        "messages": [
            HumanMessage(
                content=state["message"]
            )
        ]
    }


# ============================================================
# NODE 2 — RAG + LLM #1 JUDGE
# ============================================================

async def rag_node(state: State):

    question = state["message"]

    print("\n")
    print("=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    # --------------------------------------------------------
    # Retrieve FAQ
    # --------------------------------------------------------

    start = time.perf_counter()

    faq = retrieve_faq_for_router(question)

    end = time.perf_counter()

    print(
        f"RAG retrieval time: "
        f"{end - start:.3f}s"
    )

    print("\nRAG SCORE:")
    print(faq["score"])

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    messages = state.get(
        "messages",
        []
    )

    history_messages = messages[:-1]

    history = []

    for message in history_messages:

        if isinstance(message, HumanMessage):

            history.append(
                f"User: {extract_text(message.content)}"
            )

        elif isinstance(message, AIMessage):

            history.append(
                f"Assistant: {extract_text(message.content)}"
            )

    history_text = "\n".join(history)

    if not history_text:

        history_text = "No previous conversation."

    # --------------------------------------------------------
    # LLM #1 prompt
    # --------------------------------------------------------

    judge_prompt = f"""
Conversation history:

{history_text}


Current user query:

{question}


Retrieved FAQ question:

{faq["faq_question"]}


Retrieved FAQ answer:

{faq["faq_answer"]}


Retrieval score:

{faq["score"]}


Decide whether the retrieved FAQ actually answers
the current user query.

Return only:

RAG

or

NOT_RAG
"""

    # --------------------------------------------------------
    # Create RAG judge model
    # --------------------------------------------------------

    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    # --------------------------------------------------------
    # LLM #1
    # --------------------------------------------------------

    start = time.perf_counter()

    judge_response = await judge_llm.ainvoke(
        [
            {
                "role": "system",
                "content": RAG_JUDGE_PROMPT,
            },
            {
                "role": "user",
                "content": judge_prompt,
            },
        ]
    )

    end = time.perf_counter()

    print(
        f"\nGemini RAG Judge call: "
        f"{end - start:.3f}s"
    )

    decision = extract_text(
        judge_response.content
    ).strip().upper()

    # --------------------------------------------------------
    # RAG decision
    # --------------------------------------------------------

    if decision == "RAG":

        print("\nLLM #1 DECISION: RAG")

        return {
            "route": "rag",
            "response": faq["faq_answer"],
            "messages": [
                AIMessage(
                    content=faq["faq_answer"]
                )
            ],
        }

    print("\nLLM #1 DECISION: NOT_RAG")

    return {
        "route": "mcp"
    }


# ============================================================
# ROUTING AFTER RAG
# ============================================================

def route_after_rag(state: State):

    return state["route"]


# ============================================================
# NODE 3 — MCP AGENT
# ============================================================

async def mcp_agent_node(state: State):

    print("\n============================================================")
    print("MCP AGENT")
    print("============================================================")

    # Agent was initialized during application startup.
    # This is only a safety check.
    await initialize_mcp()

    # --------------------------------------------------------
    # Reuse existing MCP agent
    # --------------------------------------------------------

    start = time.perf_counter()

    timing_callback = MCPTimingCallback()

    result = await MCP_AGENT.ainvoke(
        {
            "messages": state["messages"]
        },
        config={
            "callbacks": [timing_callback]
        }
    )

    end = time.perf_counter()

    print(f"\nAgent ainvoke: {end - start:.3f}s")

    

    # --------------------------------------------------------
    # Extract new messages
    # --------------------------------------------------------

    existing_messages = state["messages"]

    new_messages = result["messages"][
        len(existing_messages):
    ]

    for message in new_messages:

        print("\n---")

        print(
            "TYPE:",
            type(message).__name__
        )

        print(
            "NAME:",
            getattr(
                message,
                "name",
                None
            )
        )

        print(
            "TOOL CALLS:",
            getattr(
                message,
                "tool_calls",
                None
            )
        )

        print(
            "CONTENT:",
            message.content
        )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    final_message = result["messages"][-1]

    answer = extract_text(
        final_message.content
    )

    print("\nFINAL MCP ANSWER:")
    print(answer)

    return {
        "messages": new_messages,
        "response": answer,
        "route": "mcp"
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

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
    "mcp_agent",
    mcp_agent_node
)


# ============================================================
# GRAPH EDGES
# ============================================================

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
        "rag": END,
        "mcp": "mcp_agent",
    }
)

graph_builder.add_edge(
    "mcp_agent",
    END
)


# ============================================================
# CHECKPOINTER
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# COMPILE GRAPH
# ============================================================

graph = graph_builder.compile(
    checkpointer=checkpointer
)