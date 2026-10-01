import json

from langchain_google_genai import ChatGoogleGenerativeAI

from rag import retrieve_faq_for_router


ROUTER_SYSTEM_PROMPT = """
You are a router for a commerce support chatbot.

Classify the user query into exactly ONE:
RAG
MCP
UNSUPPORTED

Return ONLY the label.


RULES

1. RAG
Use RAG when the retrieved FAQ directly answers the user's
general/static question.

Do NOT use RAG if the FAQ answer is empty, incomplete, or only
semantically related.


2. MCP
Use MCP when the query requires a specific, live, transactional,
account-related, product-specific, or action-based response that
matches one of these capabilities:

- order status
- cancel order
- process return
- track package
- support/contact information
- return policy
- shipping information
- size guide
- warranty information
- payment information
- account help
- loyalty information
- product care

Match the user's actual intent to a capability.
Do not assume a capability exists just because the query is
commerce-related.


3. UNSUPPORTED
Use UNSUPPORTED when neither the retrieved FAQ nor the available
MCP capabilities can handle the request.

Examples include unrelated questions, creative writing,
programming, general knowledge, and unsupported product questions.


DECISION

First ask:
"Does the retrieved FAQ directly answer the user's question?"

YES → RAG

NO → ask:
"Can an available MCP capability handle the user's request?"

YES → MCP

NO → UNSUPPORTED


IMPORTANT

- Specific/live/order-related requests → MCP when supported.
- Action requests → MCP when supported.
- General FAQ questions → RAG when directly answered.
- Semantic similarity alone is not enough for RAG.
- Do not choose MCP just because a keyword matches.
- Do not invent MCP capabilities.
- Pay attention to negation and the user's actual intent.
"""


def route_query(
    user_query: str,
    rag_question: str,
    rag_answer: str,
    rag_score: float
):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    prompt = f"""
{ROUTER_SYSTEM_PROMPT}

USER QUERY:
{user_query}

RETRIEVED FAQ QUESTION:
{rag_question}

RETRIEVED FAQ ANSWER:
{rag_answer}

RAG SIMILARITY SCORE:
{rag_score}

Now classify the request.

Return ONLY:
RAG
MCP
UNSUPPORTED
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return content.strip()


with open("routing_evaluation.json", "r", encoding="utf-8") as file:
    evaluation = json.load(file)

test_cases = evaluation["queries"]


for test in test_cases:

    query = test["query"]

    rag_result = retrieve_faq_for_router(query)

    result = route_query(
        query,
        rag_result["faq_question"],
        rag_result["faq_answer"],
        rag_result["score"]
    )

    print("\n-----------------------------")
    print("QUERY:", query)
    print("RAG QUESTION:", rag_result["faq_question"])
    print("RAG ANSWER:", rag_result["faq_answer"])
    print("RAG SCORE:", rag_result["score"])
    print("ROUTER:", result)
    print("EXPECTED:", test["expected"])