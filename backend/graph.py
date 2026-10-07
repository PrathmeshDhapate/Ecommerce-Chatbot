
import operator
import time

from typing import Annotated, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

from rag import retrieve_faq


# ============================================================
# MCP CLIENT
# ============================================================

MCP_CLIENT = MultiServerMCPClient(
    {
        "commerce": {
            "transport": "sse",
            "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
        }
    }
)


MCP_TOOLS = None
MCP_LLM = None
MCP_TOOL_NODE = None


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    message: str
    route: str
    response: str


# ============================================================
# HELPERS
# ============================================================

def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)

        return "".join(parts)

    if isinstance(content, dict):
        return content.get("text", "")

    return str(content)


# ============================================================
# RAG JUDGE PROMPT
# ============================================================


RAG_JUDGE_PROMPT = """
You are a routing judge for an e-commerce chatbot.

Your job is to decide whether the user's question should be
answered using the retrieved FAQ or sent to MCP tools.

USER QUESTION:
{query}

RETRIEVED FAQ ANSWER:
{faq_answer}

Return ONLY one of:

RAG
NOT_RAG


========================
CHOOSE RAG
========================

Choose RAG when the user's question is a general/static
information question AND the retrieved FAQ provides the
information needed to answer it.

Examples:

User question:
"How long does shipping take?"

Retrieved FAQ:
"Shipping takes 5–7 business days."

Decision:
RAG


User question:
"What is your return policy?"

Retrieved FAQ:
"Most products can be returned within 30 days of delivery."

Decision:
RAG


User question:
"What payment methods do you accept?"

Retrieved FAQ:
"We accept cards, debit cards, and online payments."

Decision:
RAG


========================
CHOOSE NOT_RAG
========================

Choose NOT_RAG when the question requires specific,
live, account-related, order-related, package-related,
or action-related information.

Examples:

"What is the status of ORD-1005-D?"
→ NOT_RAG

"Where is my package?"
→ NOT_RAG

"Track order ORD-1005-D."
→ NOT_RAG

"Cancel order ORD-1005-D."
→ NOT_RAG

"Process a return for ORD-1005-D."
→ NOT_RAG


========================
IMPORTANT
========================

Do NOT choose NOT_RAG merely because an MCP tool exists
that could provide similar information.

If the user's question is general/static and the retrieved
FAQ contains the answer, choose RAG.

The retrieved FAQ does not need to have exactly the same
wording as the question. Semantic equivalence is sufficient.

However, if the question requires information about a
specific order, customer, package, account, or live status,
choose NOT_RAG even if the retrieved FAQ is semantically
similar.

Return ONLY:

RAG

or

NOT_RAG
"""



# ============================================================
# MCP SYSTEM PROMPT
# ============================================================


MCP_SYSTEM_PROMPT = """
You are an e-commerce support assistant.

You have access to MCP tools.

IMPORTANT TOOL SELECTION RULES:

1. ONLY call an MCP tool when the user's request requires
   information or an action that that specific tool provides.

2. DO NOT call unrelated tools.

3. For a simple question that requires one tool, call ONLY
   that one tool.

4. Do NOT call multiple tools merely because they are available
   or because their information might be useful in a general
   support response.

5. When the user asks about a specific order, use only the tool
   required to answer the specific question.

Examples:

User:
"What is the status of order ORD-1005-D?"

Call ONLY:
get_order_status

Do NOT call:
get_return_policy
get_shipping_info
get_payment_information


User:
"What is the return policy for order ORD-1005-D?"

If the user is asking only for the general return policy,
call ONLY:
get_return_policy

If the user is asking whether the specific order is eligible
for a return, first obtain the required order information and
then use the appropriate return-related tool if necessary.


User:
"What shipping information do you provide?"

Call ONLY:
get_shipping_info


User:
"What payment methods do you accept?"

Call ONLY:
get_payment_information


6. Multiple tools should be called in the SAME assistant turn
ONLY when the user explicitly asks for multiple independent
pieces of information.

Example:

"What is the status of ORD-1005-D and what is your return policy?"

Call:
get_order_status
get_return_policy

7. If one tool's result is required to determine the arguments
or decision for another tool, execute the dependent tool only
after receiving the first tool's result.

8. Never invent order IDs, customer IDs, tracking numbers,
statuses, dates, or other information.

IMPORTANT ID FORMATS:

- Order IDs look like: ORD-1005-D
- Customer IDs look like: CUST-1005
- An order ID and customer ID are different values.
- NEVER use an order ID as a customer ID.
- If the customer ID is required but the user did not provide it,
  ask the user for the customer ID.

9. If a required tool argument is missing, ask the user for
that information.

10. Use the actual results returned by the tools.

11. Do not claim that an action was completed unless the
corresponding MCP tool confirms it.

12. After obtaining the information required to answer the
user's question, provide a clear and concise final answer.

13. Do not expose internal tool execution details to the user.

IMPORTANT:
The user's request determines which tool(s) are necessary.
Do not call extra tools simply because they might provide
additional information.
"""



# ============================================================
# MCP TIMING CALLBACK
# ============================================================

class MCPTimingCallback(BaseCallbackHandler):

    def __init__(self):
        self.llm_start_time = None
        self.tool_start_times = {}

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_start_time = time.perf_counter()
        print("\n>>> MCP AGENT LLM START")

    def on_llm_end(self, response, **kwargs):
        if self.llm_start_time is not None:
            elapsed = time.perf_counter() - self.llm_start_time
            print(f">>> MCP AGENT LLM END: {elapsed:.3f}s")
            self.llm_start_time = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        self.tool_start_times[tool_name] = time.perf_counter()
        print(f"\n>>> MCP TOOL START: {tool_name}")

    def on_tool_end(self, output, **kwargs):
        tool_name = kwargs.get("name", "unknown_tool")

        start_time = self.tool_start_times.get(tool_name)

        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            print(f">>> MCP TOOL END: {elapsed:.3f}s")
            del self.tool_start_times[tool_name]


# ============================================================
# MCP INITIALIZATION
# ============================================================

async def initialize_mcp():

    global MCP_TOOLS, MCP_LLM, MCP_TOOL_NODE

    if MCP_TOOLS is not None:
        return

    print("\n" + "=" * 60)
    print("INITIALIZING MCP")
    print("=" * 60)

    print("\nLoading MCP tools...")

    start = time.perf_counter()

    MCP_TOOLS = await MCP_CLIENT.get_tools()

    elapsed = time.perf_counter() - start

    print(f"MCP get_tools(): {elapsed:.3f}s")

    MCP_TOOLS = await MCP_CLIENT.get_tools()

    for tool in MCP_TOOLS:
        if tool.name == "get_order_status":
            print("\n" + "=" * 70)
            print("FULL GET_ORDER_STATUS SCHEMA")
            print("=" * 70)

            print("\nARGS_SCHEMA:")
            print(tool.args_schema)

            print("\nTOOL INPUT SCHEMA:")
            print(tool.get_input_schema().model_json_schema())

    print("\nAVAILABLE TOOLS:")

    for tool in MCP_TOOLS:
        print(tool.name)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    MCP_LLM = llm.bind_tools(MCP_TOOLS)

    MCP_TOOL_NODE = ToolNode(MCP_TOOLS)

    print("\nMCP manual LangGraph agent initialized.")


# ============================================================
# ADD USER MESSAGE
# ============================================================

async def add_message_node(state: State):

    return {
        "messages": [
            HumanMessage(content=state["message"])
        ]
    }


# ============================================================
# RAG NODE
# ============================================================

async def rag_node(state: State):

    query = state["message"]

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    start = time.perf_counter()

    faq_answer = retrieve_faq(query)

    retrieval_time = time.perf_counter() - start

    print(f"RAG retrieval time: {retrieval_time:.3f}s")

    if faq_answer is None:
        faq_answer = ""

    judge_prompt = RAG_JUDGE_PROMPT.format(
        query=query,
        faq_answer=faq_answer
    )

    print("\nCalling Gemini RAG Judge...")

    start = time.perf_counter()

    judge_response = await MCP_LLM.ainvoke(judge_prompt)

    judge_time = time.perf_counter() - start

    print(f"Gemini RAG Judge: {judge_time:.3f}s")

    decision = extract_text(judge_response).strip().upper()

    if "RAG" in decision and "NOT_RAG" not in decision:

        print("LLM DECISION: RAG")

        return {
            "route": "rag",
            "response": faq_answer
        }

    print("LLM DECISION: NOT_RAG")

    return {
        "route": "mcp",
        "response": ""
    }


# ============================================================
# MCP AGENT NODE
# ============================================================

async def mcp_agent_node(state: State):

    if MCP_LLM is None:
        raise RuntimeError("MCP has not been initialized.")

    timing_callback = MCPTimingCallback()

    start = time.perf_counter()

    mcp_messages = [
    SystemMessage(content=MCP_SYSTEM_PROMPT),
    *state["messages"]
    ]

    print("\n" + "=" * 70)
    print("MESSAGES SENT TO MCP LLM")
    print("=" * 70)

    for message in mcp_messages:
        print(f"\n{type(message).__name__}:")
        print(message.content)

    result = await MCP_LLM.ainvoke(
        mcp_messages,
        config={
            "callbacks": [timing_callback]
        }
    )

    elapsed = time.perf_counter() - start

    print(f"\nMCP LLM node total: {elapsed:.3f}s")

    return {
        "messages": [result]
    }


# ============================================================
# MCP TOOL NODE
# ============================================================

async def mcp_tool_node(state: State):

    if MCP_TOOL_NODE is None:
        raise RuntimeError(
            "MCP ToolNode has not been initialized."
        )

    last_message = state["messages"][-1]

    print("\n" + "=" * 60)
    print("MCP TOOL EXECUTION")
    print("=" * 60)

    if isinstance(last_message, AIMessage):

        tool_calls = last_message.tool_calls

        print(f"Tool calls requested: {len(tool_calls)}")

        for call in tool_calls:
            print(f"  → {call['name']}")
            print(f"    Args: {call['args']}")

    start = time.perf_counter()

    result = await MCP_TOOL_NODE.ainvoke(state)

    elapsed = time.perf_counter() - start

    print(f"\nMCP ToolNode total: {elapsed:.3f}s")

    return result


# ============================================================
# MCP FINAL RESPONSE NODE
# ============================================================

async def mcp_response_node(state: State):

    if not state["messages"]:
        return {
            "response": ""
        }

    final_message = state["messages"][-1]

    response = extract_text(final_message.content)

    return {
        "response": response
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_rag(state: State):

    if state["route"] == "rag":
        return "rag_end"

    return "mcp_agent"


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(State)

builder.add_node(
    "add_message",
    add_message_node
)

builder.add_node(
    "rag",
    rag_node
)

builder.add_node(
    "mcp_agent",
    mcp_agent_node
)

builder.add_node(
    "mcp_tools",
    mcp_tool_node
)

builder.add_node(
    "mcp_response",
    mcp_response_node
)


builder.add_edge(
    START,
    "add_message"
)

builder.add_edge(
    "add_message",
    "rag"
)


builder.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "rag_end": END,
        "mcp_agent": "mcp_agent"
    }
)


builder.add_conditional_edges(
    "mcp_agent",
    tools_condition,
    {
        "tools": "mcp_tools",
        END: "mcp_response"
    }
)


builder.add_edge(
    "mcp_tools",
    "mcp_agent"
)

builder.add_edge(
    "mcp_response",
    END
)


# ============================================================
# CHECKPOINTING
# ============================================================

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)
