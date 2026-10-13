
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

Analyze the user's question and determine which parts can be
answered using the retrieved FAQ candidates, which require
MCP commerce tools, and which require a general LLM response.

USER QUESTION:
{query}

RETRIEVED FAQ CANDIDATES:
{faq_candidates}


============================================================
YOUR TASK
============================================================

1. Assign a question to RAG when the FAQ candidates directly
   provide the information needed.

2. Assign a question to MCP when it requires specific, live,
   order-related, customer-related, package-related,
   account-related, or action-related commerce information.

3. Assign a question to GENERAL when it is unrelated to
   the e-commerce support services and cannot be answered
   using the FAQ candidates or MCP commerce tools.

   Examples include:
   - "What is the weather today?"
   - "Explain how neural networks work."
   - Other unrelated general knowledge questions.

4. Do not route unrelated questions to MCP just because an MCP
   tool exists.

5. A single user question may contain multiple parts that need
   different routes.


5a. A pending MCP task from an earlier turn may exist. The
    current USER QUESTION is always the user's latest message.

    - Classify the latest message according to its actual
      content, using the normal routing rules.
    - Do not route a new, independent FAQ or GENERAL question
      to MCP merely because an earlier MCP task is pending.
    - If the latest message contains an identifier and new
      question(s), preserve the identifier exactly in mcp_query
      when it is relevant to the MCP portion.
    - Do not assume that an identifier in the latest message
      automatically means the user wants to resume the pending
      task. Consider the meaning of the complete latest message.
    - Do not invent missing information or identifiers.


6. Select only FAQ candidates that directly answer the question.
   If multiple candidates answer different parts, select all
   relevant candidates while avoiding redundant information.

7. If MCP is needed, create mcp_query containing ONLY the
   MCP-relevant part of the original question.

   IMPORTANT:
   - Preserve the user's original wording as much as possible.
   - Do not remove, rewrite, summarize, or alter information
     required by an MCP tool.
   - Preserve order IDs, customer IDs, tracking numbers,
     product names, dates, account information, and other
     identifiers supplied by the user.
   - If a customer ID is mentioned anywhere in the user's
     question, include it in mcp_query when MCP is needed.
   - Never invent missing information.

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

   The incorrect version removes the customer ID.

9. If RAG is needed, generate a concise answer using ONLY
   information contained in the selected FAQ candidates.

   The answer must be in the same language as the user's
   question. Do not add unsupported information.

10. For GENERAL questions, do not generate an answer to
    the user's actual question. Set answer to exactly:

    "I'm an e-commerce support chatbot, so I can only help
    with questions about our products, orders, shipping,
    returns, and related services."

11. If the question requires both RAG and MCP, select the
    relevant FAQ candidates and preserve the MCP query.
    The answer field should contain only the answer to the
    RAG portion. The MCP agent will handle the MCP portion.

12. Choose the route using these rules:

    - RAG: The FAQ candidates answer the entire question, and
      no MCP or GENERAL response is required.

    - MCP: A commerce tool or action is required, and no
      separate GENERAL response is required.

    - GENERAL: Neither the FAQ candidates nor MCP tools can
      handle the question.

    - If the question requires multiple routes, choose the
      route with this priority: MCP first, then GENERAL,
      then RAG. Preserve the relevant information for the
      other parts using rag_indices and mcp_query.

13. If RAG is not needed, rag_indices must be [].

14. If MCP is not needed, mcp_query must be "".

15. If the route is GENERAL, rag_indices must be [],
    mcp_query must be "", and answer must contain the
    fixed chatbot-scope message specified in rule 10.

16. If the route is MCP and RAG is not needed, answer must
    be "".

17. Never invent customer IDs, order IDs, or other identifiers.


============================================================
EXAMPLES
============================================================

Example 1: RAG

USER QUESTION:
"What is your return policy?"

FAQ CANDIDATES:
[0] Return policy: Most products can be returned within 30 days.
[1] Shipping takes 5–7 business days.

OUTPUT:
{{
  "route": "RAG",
  "rag_indices": [0],
  "mcp_query": "",
  "answer": "You can return most products within 30 days."
}}


Example 2: Multiple FAQ answers

USER QUESTION:
"What is your return policy and how long does shipping take?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] How long should I expect shipping to take?
[2] Most products can be returned within 30 days.

OUTPUT:
{{
  "route": "RAG",
  "rag_indices": [0, 2],
  "mcp_query": "",
  "answer": "Shipping usually takes 5–7 business days, and most products can be returned within 30 days."
}}


Example 3: RAG + MCP

USER QUESTION:
"What is your return policy and what is the status of order ORD-1005-D? My customer ID is cust005."

FAQ CANDIDATES:
[0] Most products can be returned within 30 days.

OUTPUT:
{{
  "route": "MCP",
  "rag_indices": [0],
  "mcp_query": "What is the status of order ORD-1005-D? My customer ID is cust005.",
  "answer": "Most products can be returned within 30 days."
}}


Example 4: MCP only

USER QUESTION:
"What is the status of order ORD-1005-D?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] Most products can be returned within 30 days.

OUTPUT:
{{
  "route": "MCP",
  "rag_indices": [],
  "mcp_query": "What is the status of order ORD-1005-D?",
  "answer": ""
}}


Example 5: General question

USER QUESTION:
"Explain how neural networks work."

FAQ CANDIDATES:
[0] Most products can be returned within 30 days.
[1] Shipping takes 5–7 business days.

OUTPUT:

{{
  "route": "GENERAL",
  "rag_indices": [],
  "mcp_query": "",
  "answer": "I'm an e-commerce support chatbot, so I can only help with questions about our products, orders, shipping, returns, and related services."
}}



Example 6: Current information

USER QUESTION:
"What is the weather today?"

FAQ CANDIDATES:
[0] Shipping takes 5–7 business days.
[1] Most products can be returned within 30 days.

OUTPUT:

{{
  "route": "GENERAL",
  "rag_indices": [],
  "mcp_query": "",
  "answer": "I'm an e-commerce support chatbot, so I can only help with questions about our products, orders, shipping, returns, and related services."
}}

Example 7: Hindi RAG question

USER QUESTION:
"रिटर्न पॉलिसी क्या है?"

FAQ CANDIDATES:
[0] Return policy: Most products can be returned within 30 days.
[1] Shipping takes 5–7 business days.

OUTPUT:
{{
  "route": "RAG",
  "rag_indices": [0],
  "mcp_query": "",
  "answer": "आप अधिकांश उत्पादों को डिलीवरी के 30 दिनों के भीतर वापस कर सकते हैं।"
}}


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON with exactly these four fields:

{{
  "route": "RAG" | "MCP" | "GENERAL",
  "rag_indices": [integer indices],
  "mcp_query": "string",
  "answer": "string"
}}

Rules:
- route must be RAG, MCP, or GENERAL.
- rag_indices must contain only valid FAQ candidate indices.
- mcp_query must contain only the MCP-relevant part of the
  original user question.
- answer must contain the RAG answer or GENERAL answer
  according to the rules above.
- If the route is GENERAL, do not select irrelevant FAQ
  candidates or create an MCP query.
- If the route is MCP and no RAG answer is needed, answer
  must be "".
- Preserve all user-provided identifiers exactly.
- Return valid JSON only.
- Do not include markdown or explanations outside the JSON.
"""


# ============================================================
# MCP SYSTEM PROMPT
# ============================================================

MCP_SYSTEM_PROMPT = """
You are the MCP commerce assistant for an e-commerce platform.

Your job is to answer questions and perform commerce operations
using the available MCP tools.

============================================================
1. TOOL USAGE
============================================================

- Use MCP tools when the user requests specific commerce
  information or an action that requires the tools.
- Choose the tool that directly addresses the user's request.
- Do not call tools that are irrelevant to the request.
- Do not call the same tool repeatedly without a reason.
- If the user asks multiple independent commerce questions,
  call the necessary tools to answer all requested parts.
- Do not invent tool names, arguments, or tool results.
- Do not claim an action succeeded unless the tool result
  confirms that it succeeded.
- If a tool returns an error, report the limitation honestly.
- For a return request, use `process_return` directly unless the user explicitly asks for order status or the return workflow requires a separate status check.
- Do not call `get_order_status` merely to verify an order before using another tool.
- After a tool executes, inspect its result and stop if the user's request has been completed.
- Do not repeat a completed action or make unnecessary additional tool calls.

============================================================
2. ORDER AND CUSTOMER IDENTIFIERS
============================================================

- Never invent, guess, or infer an order ID or customer ID.
- Never use an order ID as a customer ID.
- Never substitute one customer's ID for another.
- Preserve identifiers supplied by the user exactly.
- Do not silently modify identifiers to make them fit a format.

Expected identifier formats include:
- Order ID: ORD-1005-D
- Customer ID: CUST-1005

For tools that require both an order ID and a customer ID,
you must have both identifiers explicitly supplied by the user.

If the user provides an order ID but has not provided the
required customer ID:
- Do not call the tool.
- Ask the user to provide their customer ID.
- Preserve the original request so it can be completed
  when the user supplies the missing information.

Do not treat an identifier appearing in an earlier tool result
as permission to substitute it for an identifier the user
must provide.

============================================================
3. TOOL ARGUMENTS
============================================================

- Follow the argument schema of the selected MCP tool.
- Supply the required arguments using information provided
  by the user.
- Never fabricate missing required arguments.
- Do not add unnecessary arguments.
- Do not expose internal tool schemas or implementation
  details to the user.

============================================================
4. ACCURACY
============================================================

- Use actual MCP tool results for order status, tracking,
  returns, cancellations, account information, and other
  commerce-specific facts.
- Never fabricate an order status, delivery date, tracking
  number, refund, cancellation, or transaction result.
- If the tool result does not contain the requested
  information, explain what is missing.
- Distinguish between an action being requested and an
  action being successfully completed.
- Do not claim to have performed an operation that was
  not confirmed by a tool result.

============================================================
5. COMBINING RAG AND MCP ANSWERS
============================================================

The user message may contain an answer already obtained
from the FAQ/RAG system.

- Treat the supplied RAG answer as existing context.
- Answer the MCP-specific part using actual tool results.
- When a RAG answer is supplied, preserve its relevant
  information in the final response.
- Do not contradict the RAG answer without reliable evidence.
- Do not repeat the same information unnecessarily.

If no RAG answer is supplied, answer using the MCP results
and relevant conversational context.

============================================================
6. RESPONSE STYLE
============================================================

- Be concise, clear, and helpful.
- Answer in the same language as the user.
- Ask a direct question when required information is missing.
- Do not expose chain-of-thought or internal reasoning.
- Do not mention internal routing decisions.
- Do not present unsupported assumptions as facts.
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
    has_pending_task = bool(pending_task and pending_query)

    print("DEBUG pending_mcp_query:", repr(pending_query))
    print("DEBUG pending_task:", repr(pending_task))

    # ========================================================
    # DETECT IDENTIFIERS
    # ========================================================

    identifier_pattern = (
        r"\b(?:ORD-[A-Z0-9]+-[A-Z0-9]+|CUST-?[A-Z0-9]+)\b"
    )

    supplied_identifier = bool(
        re.search(
            identifier_pattern,
            current_message,
            re.IGNORECASE
        )
    )

    
    remaining_message = re.sub(
        identifier_pattern,
        "",
        current_message,
        flags=re.IGNORECASE
    )

    remaining_message = re.sub(
        r"[\s.,!?;:()\[\]{}'\"]+",
        "",
        remaining_message
    )

    identifier_only_message = not remaining_message


    # ========================================================
    # RESUME A PENDING MCP TASK WHEN AN ID IS PROVIDED
    # ========================================================

    if has_pending_task and supplied_identifier:

        combined_mcp_query = (
            f"{pending_query}\n"
            f"Additional information from user: {current_message}"
        )

        pending_rag_answers = list(dict.fromkeys(
            state.get("pending_rag_answers", [])
        ))

        print("\n" + "=" * 60)
        print("RESUMING PENDING MCP TASK")
        print("=" * 60)
        print("Previous MCP query:", pending_query)
        print("New user information:", current_message)
        print("Combined MCP query:", combined_mcp_query)

        return {
            "route": "mcp",
            "response": "",
            "rag_answers": pending_rag_answers,
            "mcp_query": combined_mcp_query,
            "pending_mcp_query": combined_mcp_query,
            "pending_user_request": (
                state.get("pending_user_request")
                or pending_query
            ),
            "pending_rag_answers": pending_rag_answers,
            "pending_task": True
        }

    # ========================================================
    # NORMAL RAG RETRIEVAL
    # ========================================================

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

    # ========================================================
    # GEMINI RAG JUDGE
    # ========================================================

    if MCP_LLM is None:
        raise RuntimeError(
            "MCP_LLM is not initialized. "
            "Ensure initialize_mcp() runs before chat requests."
        )

    print("\nCalling Gemini RAG Judge...")

    start = time.perf_counter()
    judge_response = await MCP_LLM.ainvoke(judge_prompt)
    judge_time = time.perf_counter() - start

    print(f"Gemini RAG Judge: {judge_time:.3f}s")

    decision_text = extract_text(
        judge_response.content
    ).strip()

    print("\nRAG JUDGE RAW DECISION:")
    print(repr(decision_text))

    # ========================================================
    # CLEAN AND PARSE JSON
    # ========================================================

    # Remove an optional Markdown code fence.
    decision_text = re.sub(
        r"\A```(?:json)?[ \t]*\r?\n?|\r?\n?```[ \t]*\Z",
        "",
        decision_text,
        flags=re.IGNORECASE
    ).strip()

    try:
        decision = json.loads(decision_text)

        # Ensure the top-level JSON value is an object.
        if not isinstance(decision, dict):
            raise ValueError(
                "RAG judge output must be a JSON object."
            )

    except (json.JSONDecodeError, ValueError) as exc:

        print("ERROR: Invalid RAG judge JSON:", str(exc))

        if has_pending_task:

            # Preserve the original pending task and RAG context.
            pending_rag_answers = list(dict.fromkeys(
                state.get("pending_rag_answers", [])
            ))

            return {
                "route": "rag",
                "response": (
                    "I couldn't process that question right now. "
                    "Your pending order lookup is still waiting "
                    "for the required information."
                ),
                "rag_answers": [],
                "mcp_query": "",
                "pending_mcp_query": pending_query,
                "pending_user_request": (
                    state.get("pending_user_request")
                    or pending_query
                ),
                "pending_rag_answers": pending_rag_answers,
                "pending_task": True
            }

        # Normal-query fallback: do not invoke MCP.
        return {
            "route": "rag",
            "response": (
                "I couldn't process that question right now. "
                "Please try again."
            ),
            "rag_answers": [],
            "mcp_query": "",
            "pending_mcp_query": "",
            "pending_user_request": "",
            "pending_rag_answers": [],
            "pending_task": False
        }

    # ========================================================
    # EXTRACT AND VALIDATE DECISION FIELDS
    # ========================================================

    route = decision.get("route", "GENERAL")

    if not isinstance(route, str):
        route = "GENERAL"

    route = route.upper()

    rag_indices = decision.get("rag_indices", [])
    mcp_query = decision.get("mcp_query", "")
    rag_answer = decision.get("answer", "")

    # Invalid field types should not accidentally trigger MCP.
    if (
        not isinstance(rag_indices, list)
        or not isinstance(mcp_query, str)
        or not isinstance(rag_answer, str)
        or route not in ("RAG", "MCP", "GENERAL")
    ):
        print("ERROR: Invalid fields in RAG judge decision.")

        if has_pending_task:
            return {
                "route": "rag",
                "response": (
                    "I couldn't process that question right now. "
                    "Your pending order lookup is still waiting "
                    "for the required information."
                ),
                "rag_answers": [],
                "mcp_query": "",
                "pending_mcp_query": pending_query,
                "pending_user_request": (
                    state.get("pending_user_request")
                    or pending_query
                ),
                "pending_rag_answers": (
                    state.get("pending_rag_answers", [])
                ),
                "pending_task": True
            }

        return {
            "route": "rag",
            "response": (
                "I couldn't process that question right now. "
                "Please try again."
            ),
            "rag_answers": [],
            "mcp_query": "",
            "pending_mcp_query": "",
            "pending_user_request": "",
            "pending_rag_answers": [],
            "pending_task": False
        }

    mcp_query = mcp_query.strip()
    rag_answer = rag_answer.strip()

    # ========================================================
    # COLLECT SELECTED RAG ANSWERS
    # ========================================================

    rag_answers = []

    for index in rag_indices:

        # bool is a subclass of int in Python, so reject it.
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(faq_candidates)
        ):
            candidate_answer = faq_candidates[index]["answer"]

            if candidate_answer not in rag_answers:
                rag_answers.append(candidate_answer)

        else:
            print(
                "WARNING: Ignoring invalid FAQ index:",
                repr(index)
            )

    # Preserve the judge-generated answer for GENERAL questions
    # and when no indexed FAQ answer was selected.
    if rag_answer and not rag_answers:
        rag_answers.append(rag_answer)

    print("\nSELECTED RAG ANSWERS:")

    for answer in rag_answers:
        print(f"→ {answer}")

    print(f"\nMCP QUERY: {mcp_query}")

    # ========================================================
    # ROUTING
    # ========================================================

    # A non-empty MCP query means MCP work is required.
    if mcp_query:
        route = "MCP"

    print(f"ROUTE: {route}")

    # ========================================================
    # RAG-ONLY RESPONSE
    # ========================================================

    if route == "RAG":

        if not rag_answers:
            return {
                "route": "rag",
                "response": (
                    "I couldn't find a relevant FAQ answer. "
                    "Please try rephrasing your question."
                ),
                "rag_answers": [],
                "mcp_query": "",
                "pending_mcp_query": (
                    pending_query if has_pending_task else ""
                ),
                "pending_user_request": (
                    state.get("pending_user_request", pending_query)
                    if has_pending_task else ""
                ),
                "pending_rag_answers": (
                    state.get("pending_rag_answers", [])
                    if has_pending_task else []
                ),
                "pending_task": has_pending_task
            }

        return {
            "route": "rag",
            "response": "\n\n".join(rag_answers),
            "rag_answers": rag_answers,
            "mcp_query": "",
            "pending_mcp_query": (
                pending_query if has_pending_task else ""
            ),
            "pending_user_request": (
                state.get("pending_user_request", pending_query)
                if has_pending_task else ""
            ),
            "pending_rag_answers": (
                state.get("pending_rag_answers", [])
                if has_pending_task else []
            ),
            "pending_task": has_pending_task
        }

    # ========================================================
    # GENERAL RESPONSE
    # ========================================================

    if route == "GENERAL":

        return {
            "route": "rag",
            "response": rag_answer,
            "rag_answers": [],
            "mcp_query": "",
            "pending_mcp_query": (
                pending_query if has_pending_task else ""
            ),
            "pending_user_request": (
                state.get("pending_user_request", pending_query)
                if has_pending_task else ""
            ),
            "pending_rag_answers": (
                state.get("pending_rag_answers", [])
                if has_pending_task else []
            ),
            "pending_task": has_pending_task
        }

    # ========================================================
    # MCP ROUTE
    # ========================================================

    previous_rag_answers = (
        state.get("pending_rag_answers", [])
    )

    combined_rag_answers = list(dict.fromkeys(
        previous_rag_answers + rag_answers
    ))

    return {
        "route": "mcp",
        "response": "",
        "rag_answers": combined_rag_answers,
        "mcp_query": mcp_query,
        "pending_mcp_query": mcp_query,
        "pending_user_request": (
            state.get("pending_user_request")
            or state["message"]
        ),
        "pending_rag_answers": combined_rag_answers,
        "pending_task": False
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
    mcp_response = extract_text(final_message.content).strip()

    rag_answers = (
        state.get("pending_rag_answers", [])
        + state.get("rag_answers", [])
    )

    rag_answers = list(dict.fromkeys(
        answer.strip()
        for answer in rag_answers
        if answer and answer.strip()
    ))

    print("DEBUG pending_rag_answers:", state.get("pending_rag_answers", []))
    print("DEBUG rag_answers:", state.get("rag_answers", []))
    print("DEBUG mcp_response:", repr(mcp_response))

    # Check whether the assistant is asking for missing information.
    asks_for_customer_id = (
        "customer id" in mcp_response.lower()
        and any(
            phrase in mcp_response.lower()
            for phrase in (
                "provide",
                "need",
                "please share",
                "could you",
                "what is",
                "required"
            )
        )
    )

    asks_for_order_id = (
        "order id" in mcp_response.lower()
        and any(
            phrase in mcp_response.lower()
            for phrase in (
                "provide",
                "need",
                "please share",
                "could you",
                "what is",
                "required"
            )
        )
    )

    needs_follow_up = asks_for_customer_id or asks_for_order_id

    if needs_follow_up:
        pending_query = (
            state.get("pending_mcp_query")
            or state.get("mcp_query")
            or state.get("message", "")
        )

        pending_user_request = (
            state.get("pending_user_request")
            or state.get("message", "")
        )

        pending_rag_answers = rag_answers

        print("PRESERVING PENDING MCP TASK")
        print("Pending query:", pending_query)
        print("Pending user request:", pending_user_request)

    else:
        pending_query = ""
        pending_user_request = ""
        pending_rag_answers = []

    
    combined_parts = rag_answers.copy()

    if mcp_response and mcp_response not in combined_parts:
        combined_parts.append(mcp_response)

    else:
        combined_parts = rag_answers.copy()

    return {
        "response": "\n\n".join(combined_parts),
        "pending_mcp_query": pending_query,
        "pending_user_request": pending_user_request,
        "pending_rag_answers": pending_rag_answers,
        "pending_task": needs_follow_up
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
