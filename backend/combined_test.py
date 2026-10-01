
import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag import retrieve_faq_for_router


load_dotenv()


# =========================================================
# MCP CONFIGURATION
# =========================================================

MCP_URL = (
    "https://slavpilus-mpc-for-commerce-platforms.hf.space"
    "/gradio_api/mcp/sse"
)


# =========================================================
# LLM #1 — RAG JUDGE
# =========================================================

RAG_JUDGE_PROMPT = """
You are a judge for a commerce FAQ chatbot.

Your job is ONLY to decide whether the retrieved FAQ answer
actually answers the user's current question.

Consider:
- the user's current query
- the conversation history
- the retrieved FAQ question
- the retrieved FAQ answer

Return exactly ONE label:

RAG
NOT_RAG

Choose RAG when the FAQ directly answers the user's question.

Choose NOT_RAG when:
- the FAQ is only related but does not answer the question
- the FAQ answer is empty
- the user is asking for a specific/live/order-related result
- the user is asking to perform an action
- the user asks something outside the FAQ
- the user needs an MCP tool

Do not answer the user.
Do not explain your decision.
Return only RAG or NOT_RAG.
"""


# =========================================================
# LLM #2 — MCP AGENT
# =========================================================

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
   - any other guessed value.

5. For unsupported requests, do not call an unrelated MCP tool.
   Clearly explain that you can only help with commerce-related
   requests.

6. Always use the actual MCP tool result when generating the final
   answer.

7. Never fabricate information that was not returned by an MCP tool.
"""


# =========================================================
# HELPER
# =========================================================

def extract_text(content):
    """
    Gemini may return either a normal string or a list of
    content blocks. This extracts the actual text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

        return "".join(text_parts)

    return str(content)


# =========================================================
# LLM #1
# =========================================================

async def judge_rag(
    llm,
    query,
    history,
    faq
):

    prompt = f"""
Conversation history:
{history}

Current user query:
{query}

Retrieved FAQ question:
{faq["faq_question"]}

Retrieved FAQ answer:
{faq["faq_answer"]}

Retrieval score:
{faq["score"]}

Decide whether the retrieved FAQ actually answers the
current user query.

Return only:

RAG

or

NOT_RAG
"""

    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": RAG_JUDGE_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    decision = extract_text(response.content).strip()

    # Safety cleanup in case the model returns extra whitespace
    decision = decision.upper()

    if "RAG" in decision and "NOT_RAG" not in decision:
        return "RAG"

    return "NOT_RAG"


# =========================================================
# LLM #2
# =========================================================

async def run_mcp_agent(
    agent,
    query,
    history
):

    messages = []

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    result = await agent.ainvoke(
        {
            "messages": messages
        }
    )

    return result["messages"]


# =========================================================
# MAIN
# =========================================================

async def main():

    print("\n")
    print("=" * 70)
    print("COMBINED RAG + MCP TEST")
    print("=" * 70)


    # =====================================================
    # 1. Create LLM #1
    # =====================================================

    print("\nCreating RAG judge...")

    rag_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )


    # =====================================================
    # 2. Connect to MCP
    # =====================================================

    print("Connecting to MCP server...")

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": MCP_URL,
            }
        }
    )


    # =====================================================
    # 3. Load MCP tools
    # =====================================================

    print("Loading MCP tools...")

    tools = await client.get_tools()

    print(
        f"Loaded {len(tools)} MCP tools."
    )


    # =====================================================
    # 4. Create LLM #2 MCP agent
    # =====================================================

    mcp_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    mcp_agent = create_agent(
        model=mcp_llm,
        tools=tools,
        system_prompt=MCP_SYSTEM_PROMPT
    )


    # =====================================================
    # 5. Test cases
    # =====================================================

    test_cases = [

        {
            "name": "FAQ",
            "query": "What's the return window?",
            "history": [],
        },

        {
            "name": "Specific order",
            "query": (
                "What is the status of order "
                "ORD-123 for customer CUST-001?"
            ),
            "history": [],
        },

        {
            "name": "Payment-specific",
            "query": "Can I pay cash on delivery?",
            "history": [],
        },

        {
            "name": "Cancel order",
            "query": (
                "Cancel order ORD-123 "
                "for customer CUST-001"
            ),
            "history": [],
        },

        {
            "name": "Unsupported",
            "query": "What's the weather today?",
            "history": [],
        },

        {
            "name": "Context-dependent FAQ",
            "query": "Can I return it?",
            "history": [
                {
                    "role": "user",
                    "content": "I bought something last week.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "What would you like to know "
                        "about your purchase?"
                    ),
                },
            ],
        },
    ]


    # =====================================================
    # 6. Run combined pipeline
    # =====================================================

    for index, test in enumerate(
        test_cases,
        start=1
    ):

        query = test["query"]
        history = test["history"]

        print("\n")
        print("=" * 70)
        print(
            f"TEST {index}: {test['name']}"
        )
        print("=" * 70)

        print("\nUSER:")
        print(query)


        # -------------------------------------------------
        # RAG retrieval
        # -------------------------------------------------

        faq = retrieve_faq_for_router(query)

        print("\nRAG RETRIEVAL:")

        print(
            "FAQ QUESTION:",
            faq["faq_question"]
        )

        print(
            "FAQ ANSWER:",
            faq["faq_answer"]
        )

        print(
            "SCORE:",
            faq["score"]
        )


        # -------------------------------------------------
        # LLM #1 — RAG judge
        # -------------------------------------------------

        decision = await judge_rag(
            rag_llm,
            query,
            history,
            faq
        )

        print("\nLLM #1 DECISION:")
        print(decision)


        # -------------------------------------------------
        # RAG path
        # -------------------------------------------------

        if decision == "RAG":

            print("\nFINAL ROUTE:")
            print("RAG")

            print("\nFINAL ANSWER:")
            print(faq["faq_answer"])

            continue


        # -------------------------------------------------
        # MCP path
        # -------------------------------------------------

        print("\nFINAL ROUTE:")
        print("MCP AGENT")


        messages = await run_mcp_agent(
            mcp_agent,
            query,
            history
        )


        print("\nMCP AGENT TRACE:")

        for message in messages:

            print("\n---")
            print(
                "TYPE:",
                type(message).__name__
            )

            if hasattr(message, "name"):
                print(
                    "NAME:",
                    message.name
                )

            if hasattr(message, "tool_calls"):

                if message.tool_calls:

                    print(
                        "TOOL CALLS:",
                        message.tool_calls
                    )

            print(
                "CONTENT:",
                message.content
            )


        # -------------------------------------------------
        # Final MCP answer
        # -------------------------------------------------

        final_message = messages[-1]

        print("\nFINAL ANSWER:")
        print(
            extract_text(
                final_message.content
            )
        )


    print("\n")
    print("=" * 70)
    print("COMBINED TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
