
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import retrieve_faq_for_router


RAG_JUDGE_PROMPT = """
You are a judge for a commerce FAQ chatbot.

Your ONLY job is to decide whether the retrieved FAQ answer
actually answers the user's CURRENT question.

Use the conversation history to understand references such as:
- it
- that
- this
- my order
- my package
- the item

Return exactly ONE label:

RAG
NOT_RAG

Choose RAG only when the retrieved FAQ answer directly answers
the user's current question.

Choose NOT_RAG when:
- the FAQ is only related
- the FAQ does not answer the specific question
- the FAQ answer is empty
- the user is asking for a live or order-specific result
- the user is asking to perform an action
- the user asks something outside the FAQ

Do not answer the user.
Do not explain your decision.
Return only RAG or NOT_RAG.
"""


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


def judge_rag(
    user_query: str,
    conversation_history: str,
    rag_question: str,
    rag_answer: str,
    rag_score
):

    prompt = f"""
{RAG_JUDGE_PROMPT}

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER QUERY:
{user_query}

RETRIEVED FAQ QUESTION:
{rag_question}

RETRIEVED FAQ ANSWER:
{rag_answer}

RAG SCORE:
{rag_score}

Does the retrieved FAQ actually answer the user's current question?

Return only:
RAG
or
NOT_RAG
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return content.strip()


# ============================================================
# CONTEXT TESTS
# ============================================================

context_tests = [

    {
        "name": "Order context",
        "history": """
User: Where is my order ORD-123?
Assistant: Your order is currently being shipped.
""",
        "query": "When will it arrive?"
    },

    {
        "name": "Return context",
        "history": """
User: I want to return my shoes.
Assistant: Sure, I can help with that.
""",
        "query": "How do I do that?"
    },

    {
        "name": "Payment context",
        "history": """
User: What payment methods do you accept?
Assistant: We accept Visa, Mastercard, and other major cards.
""",
        "query": "What about UPI?"
    },

    {
        "name": "Item context",
        "history": """
User: I bought something last week.
Assistant: What would you like to know about your purchase?
""",
        "query": "Can I return it?"
    }
]


print("\n")
print("=" * 70)
print("CONVERSATION HISTORY TESTS")
print("=" * 70)


for test in context_tests:

    print("\n")
    print("-" * 70)
    print("TEST:", test["name"])
    print("-" * 70)

    query = test["query"]
    history = test["history"]

    print("CONVERSATION:")
    print(history.strip())

    print("\nCURRENT QUERY:")
    print(query)

    print("\nRetrieving FAQ...")

    rag_result = retrieve_faq_for_router(query)

    print("RAG QUESTION:")
    print(rag_result["faq_question"])

    print("\nRAG ANSWER:")
    print(rag_result["faq_answer"])

    print("\nRAG SCORE:")
    print(rag_result["score"])

    print("\nCalling RAG judge...")

    result = judge_rag(
        user_query=query,
        conversation_history=history,
        rag_question=rag_result["faq_question"],
        rag_answer=rag_result["faq_answer"],
        rag_score=rag_result["score"]
    )

    print("\nRAG JUDGE:")
    print(result)

    print("-" * 70)

