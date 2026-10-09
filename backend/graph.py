
import operator
import time
import json
import re

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
    rag_answers: list[str]
    mcp_query: str

    # Pending task context across user turns
    pending_mcp_query: str
    pending_user_request: str
    pending_rag_answers: list[str]
    pending_task: bool


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

Your job is to analyze the user's question and determine which
parts can be answered from the retrieved FAQ candidates and
which parts require MCP tools.

The user may ask one question or multiple independent questions.

USER QUESTION:
{query}

RETRIEVED FAQ CANDIDATES:

{faq_candidates}


============================================================
YOUR TASK
============================================================

For every part of the user's question:

1. If it is a general/static information question and one of
   the FAQ candidates provides the required information,
   assign that part to RAG.

2. If it requires specific, live, order-related, customer-related,
   package-related, account-related, or action-related information,
   assign that part to MCP.

3. A single user question may contain BOTH RAG and MCP parts.

4. Do not assign a question to MCP merely because an MCP tool
   exists for similar information. Use RAG when the FAQ candidate
   clearly provides the required general/static information.

5. Only select FAQ candidates that actually answer a part of
   the user's question.

6. If multiple FAQ candidates answer different parts of the
   question, select all relevant candidates.

7. If MCP is needed, create mcp_query containing ONLY the
   MCP-relevant part of the user's question.

   IMPORTANT:
   - Preserve the user's original wording as much as possible.
   - DO NOT remove, rewrite, summarize, or alter information
     that may be required by an MCP tool.
   - This includes order IDs, customer IDs, tracking numbers,
     product names, dates, account information, and any other
     identifiers or arguments mentioned by the user.
   - If a customer ID is mentioned anywhere in the user's
     question, it MUST be included in mcp_query when MCP is
     being used.
   - If an order ID is mentioned, preserve it exactly.
   - Do not invent any missing information.

8. mcp_query must be a faithful subset of the original user
   question, not a summary or rewritten version.

   Example:

   USER QUESTION:
   "What is the return policy and what is the status of
   order ORD-1005-D? My customer ID is cust005."

   CORRECT mcp_query:
   "What is the status of order ORD-1005-D? My customer ID is cust005."

   INCORRECT mcp_query:
   "What is the status of order ORD-1005-D?"

   The incorrect version is NOT allowed because it removes
   the customer ID.

9. If one or more FAQ candidates are selected for RAG, generate
   a concise answer to the user's question using ONLY the
   information contained in the selected FAQ candidates.

   The answer MUST be written in the same language as the
   user's question.

   Do not translate identifiers such as order IDs, customer IDs,
   tracking numbers, or product names.

   If the user's question is in English, answer in English.
   If it is in Hindi, answer in Hindi.
   If it is in Marathi, answer in Marathi.

   Do not add information that is not supported by the
   selected FAQ candidates.

10. Select the minimum number of FAQ candidates needed to fully
    answer the user's question.

    If multiple candidates provide substantially the same
    information, select only the most direct and relevant
    candidate.

    Do not select redundant FAQ candidates just because they
    are semantically related to the user's question.

11. If MCP is not needed, mcp_query must be an empty string.

12. If RAG is not needed, rag_indices must be an empty list.

13. If RAG is not needed, answer must be an empty string.


============================================================
EXAMPLES
============================================================

Example 1:

USER QUESTION:
"What is your return policy?"

FAQ CANDIDATES:
[0] Return policy: Most products can be returned within 30 days.
[1] Shipping takes 5–7 business days.

OUTPUT:
{{"rag_indices":[0],"mcp_query":"","answer":"You can return most products within 30 days."}}


Example 2:

USER QUESTION:
"What is your return policy and how long does shipping take?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] How long should I expect shipping to take?
[2] Most products can be returned within 30 days.

OUTPUT:
{{"rag_indices":[0,2],"mcp_query":"","answer":"Shipping usually takes 5–7 business days, and most products can be returned within 30 days."}}


Example 3:

USER QUESTION:
"What is your return policy and what is the status of order ORD-1005-D?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] How long should I expect shipping to take?
[2] Most products can be returned within 30 days.

OUTPUT:
{{"rag_indices":[2],"mcp_query":"What is the status of order ORD-1005-D?","answer":"You can return most products within 30 days."}}


Example 4:

USER QUESTION:
"How long does shipping take and what is the status of order ORD-1005-D?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] How long should I expect shipping to take?
[2] Most products can be returned within 30 days.

OUTPUT:
{{"rag_indices":[0,1],"mcp_query":"What is the status of order ORD-1005-D?","answer":"Shipping usually takes 5–7 business days."}}


Example 5:

USER QUESTION:
"What is the status of order ORD-1005-D?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] Return policy: Most products can be returned within 30 days.

OUTPUT:
{{"rag_indices":[],"mcp_query":"What is the status of order ORD-1005-D?","answer":""}}


Example 6:

USER QUESTION:
"रिटर्न पॉलिसी क्या है?"

FAQ CANDIDATES:
[0] Return policy: Most products can be returned within 30 days.
[1] What is the process for returning a product?

OUTPUT:
{{"rag_indices":[0],"mcp_query":"","answer":"आप अधिकांश उत्पादों को डिलीवरी के 30 दिनों के भीतर वापस कर सकते हैं।"}}


Example 7:

USER QUESTION:
"रिटर्न पॉलिसी काय आहे?"

FAQ CANDIDATES:
[0] Return policy: Most products can be returned within 30 days.
[1] What is the process for returning a product?

OUTPUT:
{{"rag_indices":[0],"mcp_query":"","answer":"तुम्ही बहुतेक उत्पादने डिलिव्हरीच्या ३० दिवसांच्या आत परत करू शकता."}}


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

The JSON must contain exactly these three fields:

{{
  "rag_indices": [integer indices],
  "mcp_query": "string",
  "answer": "string"
}}

Rules for the fields:

- rag_indices must contain only valid FAQ candidate indices.
- mcp_query must contain only the MCP-relevant part of the
  original user question.
- answer must contain the final RAG answer in the same language
  as the user's question.
- If RAG is not needed, rag_indices must be [] and answer must
  be "".
- If MCP is not needed, mcp_query must be "".

Do not include markdown.
Do not include explanations.
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

RAG CONTEXT AND RESPONSE FORMATTING:

14. The user request may already have been partially answered using
    FAQ information retrieved by the RAG component.

15. The application may provide a RAG ANSWER alongside the
    MCP-specific query. Treat that answer as existing information
    available to you.

16. If a RAG ANSWER is provided:

* Preserve its relevant information in the final response.
* Answer the remaining part using MCP tool results.
* Combine both parts into one coherent, concise response.
* Avoid repeating the same information.
* Do not contradict the RAG answer unless verified MCP results
  establish that it is outdated or incorrect.

17. If no RAG ANSWER is provided, answer normally using the
    user's request and the MCP results.

18. Never invent facts to fill gaps in either source.
    If required information is missing, ask the user for it.

19. If a required customer ID or another required argument is
    missing, clearly ask the user for it. Do not imply that an
    order lookup or action succeeded when it did not.


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

    
    
    current_message = state["message"].strip()
    pending_query = state.get("pending_mcp_query", "")
    pending_task = state.get("pending_task", False)

    print("DEBUG pending_mcp_query:", repr(pending_query))
    print("DEBUG pending_task:", repr(pending_task))

    if pending_task and pending_query:
        original_request = (
            state.get("pending_user_request")
            or pending_query
        )
        query = (
            f"Original request: {original_request}\n"
            f"Pending MCP task: {pending_query}\n"
            f"Follow-up from user: {current_message}"
        )
    else:
        query = current_message

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    start = time.perf_counter()

    faq_candidates = retrieve_faq(query)

    retrieval_time = time.perf_counter() - start

    print(f"RAG retrieval time: {retrieval_time:.3f}s")

    if faq_candidates is None:
        faq_candidates = []

    faq_candidates_text = "\n\n".join(
        f"[{i}] Question: {candidate['question']}\n"
        f"Answer: {candidate['answer']}\n"
        f"Score: {candidate['score']}"
        for i, candidate in enumerate(faq_candidates)
    )

    if not faq_candidates_text:
        faq_candidates_text = "No relevant FAQ candidates found."

    judge_prompt = RAG_JUDGE_PROMPT.format(
        query=query,
        faq_candidates=faq_candidates_text
    )

    print("\nCalling Gemini RAG Judge...")

    start = time.perf_counter()

    judge_response = await MCP_LLM.ainvoke(judge_prompt)

    judge_time = time.perf_counter() - start

    print(f"Gemini RAG Judge: {judge_time:.3f}s")

    decision_text = extract_text(judge_response.content).strip()

    print("\nRAG JUDGE DECISION:")
    print(decision_text)

    try:
        decision = json.loads(decision_text)
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON returned by RAG judge.")

        return {
            "route": "mcp",
            "response": "",
            "rag_answers": [],
            "mcp_query": query
        }

    rag_indices = decision.get("rag_indices", [])
    mcp_query = decision.get("mcp_query", "").strip()

    # Final RAG answer generated by the Gemini judge
    rag_answer = decision.get("answer", "").strip()

    
   
    previous_rag_answers = (
        state.get("pending_rag_answers", [])
        if state.get("pending_task", False)
        else []
    )

    rag_answers = list(dict.fromkeys(previous_rag_answers))

    if rag_answer and rag_answer not in rag_answers:
        rag_answers.append(rag_answer)

    print("\nSELECTED RAG ANSWERS:")

    for answer in rag_answers:
        print(f"→ {answer}")

    print(f"\nMCP QUERY: {mcp_query}")

    if mcp_query:
        route = "mcp"
    else:
        route = "rag"

    print(f"ROUTE: {route}")

    
    if route == "rag":
        return {
            "route": "rag",
            "response": "\n\n".join(rag_answers),
            "rag_answers": rag_answers,
            "mcp_query": "",
            "pending_mcp_query": "",
            "pending_user_request": "",
            "pending_rag_answers": [],
            "pending_task": False
        }

    
    return {
        "route": route,
        "response": "",
        "rag_answers": rag_answers,
        "mcp_query": mcp_query,
        "pending_mcp_query": pending_query or mcp_query,
        "pending_user_request": (
            state.get("pending_user_request")
            or state["message"]
        ),
        "pending_rag_answers": rag_answers,
        "pending_task": pending_task
    }

# ============================================================
# MCP AGENT NODE
# ============================================================

async def mcp_agent_node(state: State):

    if MCP_LLM is None:
        raise RuntimeError("MCP has not been initialized.")

    timing_callback = MCPTimingCallback()

    start = time.perf_counter()

    # Find the latest MCP AI message and ToolMessage.
    # We do not send the entire LangGraph conversation history.
    messages = state["messages"]

    latest_tool_index = None

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].__class__.__name__ == "ToolMessage":
            latest_tool_index = i
            break

    
    if latest_tool_index is None:
        # First MCP call: send the MCP query and any RAG answer.
        mcp_query = state["mcp_query"] or state["message"]

        print("STATE RAG ANSWERS:", state.get("rag_answers"))
        print("STATE MCP QUERY:", state.get("mcp_query"))

        rag_context = "\n".join(
            answer for answer in state.get("rag_answers", [])
            if answer
        )

        user_content = f"MCP QUERY:\n{mcp_query}"

        if rag_context:
            user_content = (
                f"RAG ANSWER ALREADY AVAILABLE:\n{rag_context}\n\n"
                f"Answer the MCP-specific part and preserve the "
                f"relevant RAG information in your final response.\n\n"
                f"{user_content}"
            )

        mcp_messages = [
            SystemMessage(content=MCP_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ]

    else:
        # MCP follow-up: send only the MCP query,
        # the AI tool request, and the latest tool result.
        ai_message = None

        for i in range(latest_tool_index - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                ai_message = messages[i]
                break

        rag_context = "\n".join(
            answer for answer in state.get("rag_answers", [])
            if answer
        )

        user_content = state["mcp_query"] or state["message"]

        if rag_context:
            user_content = (
                f"RAG ANSWER ALREADY AVAILABLE:\n{rag_context}\n\n"
                f"Answer the user's request using the RAG answer "
                f"and the MCP tool result.\n\n"
                f"{user_content}"
            )

        mcp_messages = [
            SystemMessage(content=MCP_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ]  

        if ai_message is not None:
            mcp_messages.append(ai_message)

        mcp_messages.append(messages[latest_tool_index])

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

def validate_tool_calls(state: State):
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return "mcp_tools"

    customer_id_tools = {
        "get_order_status",
        "track_package",
    }

    user_text = " ".join(
        msg.content
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
        and isinstance(msg.content, str)
    )

    supplied_customer_ids = {
        cid.casefold()
        for cid in re.findall(
            r"\bCUST-?[A-Za-z0-9-]+\b",
            user_text,
            flags=re.IGNORECASE,
        )
    }

    for call in last_message.tool_calls:
        if call["name"] in customer_id_tools:
            args = call.get("args", {})
            customer_id = args.get("customer_id", "")

            if (
                not isinstance(customer_id, str)
                or customer_id.casefold() not in supplied_customer_ids
            ):
                print(
                    f"\n❌ BLOCKED: customer_id was not "
                    f"provided by the user for {call['name']}"
                )
                return "blocked"

    return "mcp_tools"



async def blocked_node(state: State):
    pending_query = state.get("mcp_query") or state["message"]

    return {
        "response": "Please provide your customer ID so I can check the status of your order.",
        "pending_mcp_query": pending_query,
        "pending_user_request": state.get("pending_user_request") or state["message"],
        "pending_rag_answers": (
            state.get("pending_rag_answers", [])
            + state.get("rag_answers", [])
        ),
        "pending_task": True
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
            "response": "",
            "pending_mcp_query": "",
            "pending_user_request": "",
            "pending_rag_answers": [],
            "pending_task": False
        }

    final_message = state["messages"][-1]
    mcp_response = extract_text(final_message.content)

    # Include previous pending answers and current RAG answers.
    rag_answers = (
        state.get("pending_rag_answers", [])
        + state.get("rag_answers", [])
    )

    # Remove duplicate answers while preserving their order.
    rag_answers = list(dict.fromkeys(
        answer.strip()
        for answer in rag_answers
        if answer and answer.strip()
    ))

    # ADD THESE PRINTS HERE
    print("DEBUG pending_rag_answers:", state.get("pending_rag_answers", []))
    print("DEBUG rag_answers:", state.get("rag_answers", []))
    print("DEBUG mcp_response:", repr(mcp_response))
    
    
    if mcp_response and mcp_response.strip():
        combined_parts = [mcp_response.strip()]
    else:
        combined_parts = rag_answers.copy()

    return {
        "response": "\n\n".join(combined_parts),
        "pending_mcp_query": "",
        "pending_user_request": "",
        "pending_rag_answers": [],
        "pending_task": False
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
        "tools": "validate_tools",
        END: "mcp_response"
    }
)

builder.add_node("validate_tools", lambda state: state)

builder.add_conditional_edges(
    "validate_tools",
    validate_tool_calls,
    {
        "mcp_tools": "mcp_tools",
        "blocked": "blocked"
    }
)

builder.add_node("blocked", blocked_node)
builder.add_edge("blocked", END)


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
