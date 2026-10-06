from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

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

4. If there is no RAG question, use:
RAG_PART:
NONE

5. If nothing remains after selecting the RAG part, use:
REMAINING_PART:
NONE

6. Preserve the meaning of the user's original request.

USER QUERY:
{query}
"""


test_queries = [
    "What is the status of ORD-1005-D and what is your return policy?",
    "What is the return policy and how long does shipping take?",
    "What is the status of ORD-1005-D and can I cancel it?",
    "How long does shipping take?",
    "What is the status of ORD-1005-D?",
]


for query in test_queries:
    print("\n" + "=" * 70)
    print("USER QUERY:")
    print(query)
    print("=" * 70)

    prompt = QUERY_SPLIT_PROMPT.format(query=query)

    response = llm.invoke(prompt)

    print("\nGEMINI OUTPUT:")
    print(response.content)