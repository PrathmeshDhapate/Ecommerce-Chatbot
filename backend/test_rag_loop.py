import asyncio
import operator
import time

from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_chroma import Chroma

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()


# ------------------------------------------------------------
# Gemini model
# ------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# ------------------------------------------------------------
# Chroma
# ------------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = Chroma(
    collection_name="faq_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ------------------------------------------------------------
# MCP
# ------------------------------------------------------------

MCP_SERVER_URL = (
    "https://slavpilus-mpc-for-commerce-platforms.hf.space/"
    "gradio_api/mcp/sse"
)

MCP_CLIENT = MultiServerMCPClient(
    {
        "commerce": {
            "url": MCP_SERVER_URL,
            "transport": "sse",
        }
    }
)


# These will be initialized once before testing.

MCP_TOOLS = None
MCP_LLM = None
MCP_TOOL_NODE = None


# ============================================================
# 2. QUERY SPLITTER PROMPT
# ============================================================

QUERY_SPLIT_PROMPT = """
You are analyzing an e-commerce user's request.

Your job is to identify ONE part of the user's request that can
be answered using general/static FAQ knowledge, and identify
what remains unanswered.

Do NOT answer the questions.

Return exactly this format:

RAG_PART:
<one RAG-answerable question, or NONE>

REMAINING_PART:
<everything that still needs to be handled, or NONE>


Rules:

1. RAG_PART must contain only a general/static question that
   could potentially be answered by an FAQ.

2. A question involving a specific order, customer, package,
   account, tracking number, live status, or action is NOT_RAG
   and belongs in REMAINING_PART.

3. If there are multiple RAG questions, put only ONE of them
   in RAG_PART. Put the others in REMAINING_PART.

4. NEVER invent, infer, or create a new question that was not
   explicitly asked by the user.

5. Preserve the user's original meaning.

6. If there is no RAG question, use:

RAG_PART:
NONE

7. If nothing remains, use:

REMAINING_PART:
NONE

USER QUERY:
{query}
"""


# ============================================================
# 3. RAG JUDGE PROMPT
# ============================================================

RAG_JUDGE_PROMPT = """
You are the routing and RAG judge for an e-commerce chatbot.

The user's query may contain one or multiple requests.

Your job is to:

1. Identify ALL parts of the user's query that can be
   answered using the retrieved FAQ candidates.
2. For each RAG-answerable part, select the FAQ answer
   that actually answers that part.
3. Identify everything that still needs to be handled by MCP.

Do NOT answer the user's questions yourself.
Only use answers provided by the retrieved FAQ candidates.

Return EXACTLY this format:

RAG_PARTS:
<one or more RAG questions with their FAQ answers, or NONE>

MCP_PART:
<everything that still needs to be handled by MCP, or NONE>

DECISION:
<RAG or NOT_RAG>


USER QUERY:
{query}

RETRIEVED FAQ CANDIDATES:
{faq_candidates}


Rules:

1. A RAG part must be a general/static information request
   that can be answered by one of the retrieved FAQ candidates.

2. A request involving a specific order, customer, package,
   tracking number, account, live information, or action belongs
   in MCP_PART.

3. If the user's query contains multiple RAG questions,
   identify ALL of them.

4. For each RAG question, use the FAQ answer from the candidate
   that actually answers that question.

5. Do NOT use an FAQ candidate merely because it is
   semantically similar. The candidate must actually answer
   the user's request.

6. If a requested topic has no valid FAQ candidate,
   put that request in MCP_PART.

7. NEVER invent an FAQ answer.

8. NEVER invent a question that the user did not ask.

9. Preserve the user's original meaning and wording as much
   as possible.

10. If the entire query is RAG-answerable:
    - Put ALL RAG questions in RAG_PARTS.
    - Put MCP_PART as NONE.
    - Set DECISION to RAG.

11. If the entire query requires MCP:
    - Set RAG_PARTS to NONE.
    - Put the original request in MCP_PART.
    - Set DECISION to NOT_RAG.

12. If the query contains both RAG and MCP parts:
    - Put ALL valid RAG questions and their FAQ answers in RAG_PARTS.
    - Put the remaining request in MCP_PART.
    - Set DECISION to RAG.

13. DECISION means whether at least one valid RAG answer
    was found. It does NOT mean whether MCP is also needed.

Return ONLY the requested fields.
"""


# ============================================================
# 4. MCP SYSTEM PROMPT
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

4. Do NOT call multiple tools merely because they are available.

5. When the user asks about a specific order, use only the
   tool required to answer the specific question.

6. Multiple tools should be called in the SAME assistant turn
   ONLY when the user explicitly asks for multiple independent
   pieces of information.

7. If one tool's result is required to determine the arguments
   or decision for another tool, execute the dependent tool only
   after receiving the first tool's result.

8. Never invent order IDs, customer IDs, tracking numbers,
   statuses, dates, or other information.

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
# 5. EXTRACT GEMINI TEXT
# ============================================================

def extract_text(response):

    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):

        for item in response.content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    return item.get("text", "")

    return str(response.content)


# ============================================================
# 6. INITIALIZE MCP
# ============================================================

async def initialize_mcp():

    global MCP_TOOLS, MCP_LLM, MCP_TOOL_NODE

    print("\n" + "=" * 70)
    print("INITIALIZING MCP")
    print("=" * 70)

    start = time.perf_counter()

    MCP_TOOLS = await MCP_CLIENT.get_tools()

    elapsed = time.perf_counter() - start
    print(f"MCP get_tools(): {elapsed:.3f}s")

    # ------------------------------------------------------------
    # DEBUG: INSPECT get_order_status TOOL SCHEMA
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("GET_ORDER_STATUS TOOL SCHEMA")
    print("=" * 70)

    for tool in MCP_TOOLS:

        if tool.name == "get_order_status":

            print("TOOL:")
            print(tool)

            print("\nTOOL ARGS SCHEMA:")
            print(tool.args)

            break

    # ------------------------------------------------------------
    # AVAILABLE TOOLS
    # ------------------------------------------------------------

    print("\nAVAILABLE MCP TOOLS:")

    for tool in MCP_TOOLS:
        print(tool.name)

    # ------------------------------------------------------------
    # INITIALIZE MCP LLM
    # ------------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    MCP_LLM = llm.bind_tools(MCP_TOOLS)

    MCP_TOOL_NODE = ToolNode(MCP_TOOLS)


# ============================================================
# 7. SPLIT QUERY
# ============================================================

def split_query(query):

    print("\n" + "-" * 70)
    print("STEP 1: QUERY SPLITTING")
    print("-" * 70)

    print("INPUT:")
    print(query)

    prompt = QUERY_SPLIT_PROMPT.format(
        query=query
    )

    response = llm.invoke(
        prompt
    )

    text = extract_text(
        response
    )

    print("\nGEMINI SPLITTER OUTPUT:")
    print(text)

    rag_part = ""
    remaining_part = ""

    if "RAG_PART:" in text:

        rag_part = text.split(
            "RAG_PART:",
            1
        )[1]

        if "REMAINING_PART:" in rag_part:

            rag_part, remaining_part = rag_part.split(
                "REMAINING_PART:",
                1
            )

    rag_part = rag_part.strip()
    remaining_part = remaining_part.strip()

    if rag_part.upper() == "NONE":
        rag_part = ""

    if remaining_part.upper() == "NONE":
        remaining_part = ""

    print("\nPARSED RESULT:")

    print(
        "RAG PART:",
        rag_part or "NONE"
    )

    print(
        "REMAINING PART:",
        remaining_part or "NONE"
    )

    return rag_part, remaining_part


# ============================================================
# 8. RAG RETRIEVAL
# ============================================================

def retrieve_faq(question):
    print("\n" + "-" * 70)
    print("STEP 2: RAG RETRIEVAL")
    print("-" * 70)

    print("QUERY:")
    print(question)

    start = time.perf_counter()

    results = vector_store.similarity_search_with_score(
        question,
        k=4
    )

    elapsed = time.perf_counter() - start
    print(f"\nRetrieval time: {elapsed:.3f}s")

    if not results:
        print("RAG: No results")
        return []

    candidates = []

    print("\nRETRIEVED FAQ CANDIDATES:")

    for i, (document, score) in enumerate(results, start=1):

        answer = document.metadata.get("answer", "")

        print(f"\nCandidate {i}:")
        print("Question:", document.page_content)
        print("Score:", score)
        print("Answer:", answer)

        candidates.append({
            "question": document.page_content,
            "answer": answer,
            "score": score
        })

    return candidates

# ============================================================
# 9. RAG JUDGE
# ============================================================

def judge_rag(query, faq_candidates):
    print("\n" + "-" * 70)
    print("STEP 3: RAG JUDGE + ROUTER")
    print("-" * 70)

    print("USER QUESTION:")
    print(query)

    # --------------------------------------------------
    # BUILD FAQ CANDIDATES TEXT
    # --------------------------------------------------

    if not faq_candidates:
        faq_candidates_text = "NONE"

    else:
        faq_candidates_text = ""

        for i, candidate in enumerate(faq_candidates, start=1):
            faq_candidates_text += (
                f"\nCandidate {i}:\n"
                f"Question: {candidate['question']}\n"
                f"Answer: {candidate['answer']}\n"
                f"Score: {candidate['score']}\n"
            )

    print("\nRETRIEVED FAQ CANDIDATES:")
    print(faq_candidates_text)

    # --------------------------------------------------
    # BUILD JUDGE PROMPT
    # --------------------------------------------------

    prompt = RAG_JUDGE_PROMPT.format(
        query=query,
        faq_candidates=faq_candidates_text
    )

    # Temporary latency debugging
    print("\nPROMPT LENGTH:", len(prompt))

    # --------------------------------------------------
    # GEMINI RAG JUDGE
    # --------------------------------------------------

    start = time.perf_counter()

    response = llm.invoke(prompt)

    elapsed = time.perf_counter() - start

    text = extract_text(response)

    print(f"\nJudge + Router time: {elapsed:.3f}s")

    print("\nGEMINI ROUTER OUTPUT:")
    print(text)

    # --------------------------------------------------
    # PARSE RAG PARTS
    # --------------------------------------------------

    rag_parts = []

    if "RAG_PARTS:" in text:

        rag_section = text.split("RAG_PARTS:", 1)[1]

        if "MCP_PART:" in rag_section:
            rag_section = rag_section.split("MCP_PART:", 1)[0]

        current_query = None
        current_answer = None

        for line in rag_section.splitlines():

            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            # Remove optional bullet
            if line.startswith("- "):
                line = line[2:].strip()

            lower_line = line.lower()

            # ------------------------------------------
            # EXPLICIT QUERY LABELS
            # ------------------------------------------

            if (
                lower_line.startswith("question:")
                or lower_line.startswith("user question:")
                or lower_line.startswith("user query:")
            ):

                # Save previous RAG part
                if current_query and current_answer:
                    rag_parts.append({
                        "query": current_query,
                        "answer": current_answer
                    })

                # Extract query
                current_query = line.split(":", 1)[1].strip()

                current_answer = None

            # ------------------------------------------
            # ANSWER
            # ------------------------------------------

            elif (
                lower_line.startswith("answer:")
                or lower_line.startswith("faq answer:")
            ):

                # Extract answer
                current_answer = line.split(":", 1)[1].strip()

            # ------------------------------------------
            # BARE QUERY
            # ------------------------------------------

            else:

                # If we don't currently have a query,
                # treat this line as the RAG question.
                if current_query is None:
                    current_query = line
                    current_answer = None

        # Save final RAG part
        if current_query and current_answer:
            rag_parts.append({
                "query": current_query,
                "answer": current_answer
            })

    # --------------------------------------------------
    # PARSE MCP PART
    # --------------------------------------------------

    mcp_part = ""

    if "MCP_PART:" in text:

        mcp_part = text.split("MCP_PART:", 1)[1]

        if "DECISION:" in mcp_part:
            mcp_part = mcp_part.split("DECISION:", 1)[0]

    mcp_part = mcp_part.strip()

    if mcp_part.upper() == "NONE":
        mcp_part = ""

    # --------------------------------------------------
    # PARSE DECISION
    # --------------------------------------------------

    decision = "NOT_RAG"

    if "DECISION:" in text:

        decision = (
            text.split("DECISION:", 1)[1]
            .strip()
            .upper()
        )

    # --------------------------------------------------
    # PRINT PARSED RESULT
    # --------------------------------------------------

    print("\nPARSED RESULT:")

    for part in rag_parts:

        print("RAG PART:", part["query"])
        print("RAG ANSWER:", part["answer"])

    print("MCP PART:", mcp_part or "NONE")
    print("DECISION:", decision)

    # --------------------------------------------------
    # RETURN
    # --------------------------------------------------

    return {
        "rag_parts": rag_parts,
        "mcp_part": mcp_part,
        "decision": decision
    }

# ============================================================
# 10. PROCESS RAG PART
# ============================================================

def process_rag_part(query):

    print("\n" + "=" * 70)
    print("PROCESSING RAG PART")
    print("=" * 70)

    print(query)

    faq_answer = retrieve_faq(query)

    router_result = judge_rag(
        query,
        faq_answer
    )

    return router_result

# ============================================================
# 11. MCP AGENT
# ============================================================

async def run_mcp_agent(query):

    print("\n" + "=" * 70)
    print("STEP 4: MCP PROCESSING")
    print("=" * 70)

    print("MCP QUERY:")
    print(query)

    messages = [
        SystemMessage(
            content=MCP_SYSTEM_PROMPT
        ),
        HumanMessage(
            content=query
        )
    ]

    while True:

        # --------------------------------------------------------
        # Ask Gemini what to do
        # --------------------------------------------------------

        start = time.perf_counter()

        response = await MCP_LLM.ainvoke(
            messages
        )

        elapsed = time.perf_counter() - start

        print(
            f"\nMCP AGENT LLM: {elapsed:.3f}s"
        )

        messages.append(response)

        # --------------------------------------------------------
        # Check whether Gemini requested tools
        # --------------------------------------------------------

        tool_calls = getattr(
            response,
            "tool_calls",
            []
        )

        if not tool_calls:

            print(
                "MCP: No more tools required."
            )

            return extract_text(
                response
            )

        # --------------------------------------------------------
        # Show requested tools
        # --------------------------------------------------------

        print("\nMCP TOOL CALLS:")

        for call in tool_calls:
            print(f"  → {call['name']}")
            print(f"    Args: {call['args']}")

        # --------------------------------------------------------
        # Execute MCP tools directly
        # --------------------------------------------------------

        start = time.perf_counter()

        tool_messages = []

        for call in tool_calls:

            tool_name = call["name"]
            tool_args = call["args"]

            tool = next(
                tool
                for tool in MCP_TOOLS
                if tool.name == tool_name
            )

            result = await tool.ainvoke(
                tool_args
            )

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                    name=tool_name
                )
            )

        elapsed = time.perf_counter() - start

        print(
            f"MCP Tool execution: {elapsed:.3f}s"
        )

        messages.extend(
            tool_messages
        )


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

async def run_pipeline(user_query):

    print("\n")
    print("#" * 70)
    print("STARTING RAG + MCP PIPELINE")
    print("#" * 70)

    print("\nORIGINAL USER QUERY:")
    print(user_query)

    answers = []

    # ========================================================
    # RAG RETRIEVAL + JUDGE + ROUTING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 1: RAG RETRIEVAL + JUDGE + ROUTING")
    print("=" * 70)

    faq_answer = retrieve_faq(
        user_query
    )

    router_result = judge_rag(
        user_query,
        faq_answer
    )

    # ========================================================
    # RAG RESULT
    # ========================================================

    for rag_part in router_result["rag_parts"]:
        answers.append({
            "query": rag_part["query"],
            "route": "RAG",
            "answer": rag_part["answer"]
        })



    # ========================================================
    # MCP RESULT
    # ========================================================

    mcp_query = router_result["mcp_part"]

    if mcp_query:

        mcp_answer = await run_mcp_agent(
            mcp_query
        )

        answers.append(
            {
                "query": mcp_query,
                "route": "MCP",
                "answer": mcp_answer
            }
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("#" * 70)
    print("FINAL RESULTS")
    print("#" * 70)

    for answer in answers:

        print("\n----------------------------------------")

        print(
            "QUERY:",
            answer["query"]
        )

        print(
            "ROUTE:",
            answer["route"]
        )

        print(
            "ANSWER:",
            answer["answer"]
        )

    print("\n")
    print("#" * 70)
    print("PIPELINE FINISHED")
    print("#" * 70)


# ============================================================
# 13. TEST QUERY
# ============================================================

TEST_QUERY = (
    "What is the return policy and"
    "what is the status of order ORD-1005-D?" 
)


# ============================================================
# 14. RUN
# ============================================================

async def main():

    await initialize_mcp()

    await run_pipeline(
        TEST_QUERY
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )