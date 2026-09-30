import json
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# -------------------------
# 1. Load FAQ data
# -------------------------

FAQ_FILE = Path(__file__).parent / "faq_data" / "faqs.json"

with open(FAQ_FILE, "r", encoding="utf-8") as file:
    faqs = json.load(file)


# -------------------------
# 2. Create embedding model
# -------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# -------------------------
# 3. Intent descriptions
# -------------------------

INFORMATION_INTENTS = [
    """
    The user's goal is to understand information rather than perform an operation.
    The user wants an explanation of general knowledge, rules, policies,
    requirements, eligibility, procedures, or guidelines.
    """,

    """
    The user wants to know how a service works or what conditions and rules
    apply to a situation. The desired result is an informational explanation,
    not access to or modification of current system state.
    """,

    """
    The user is seeking general guidance or clarification. The response can
    be provided by explaining established information without inspecting or
    changing a particular user's transaction or account.
    """
]

ACTION_INTENTS = [
    """
    The user's goal is for the system to access current information about
    a real user's situation. The response requires retrieving current state
    from a system rather than explaining a general rule or policy.
    """,

    """
    The user's goal is for the system to perform an operation on the user's
    behalf. The operation changes or acts upon real transactional or account
    state rather than merely providing information.
    """,

    """
    The user expects the system to interact with a live service or account
    in order to retrieve or modify current state. The desired result depends
    on accessing the system rather than giving a general explanation.
    """
]

information_embeddings = [
    embeddings.embed_query(intent)
    for intent in INFORMATION_INTENTS
]

action_embeddings = [
    embeddings.embed_query(intent)
    for intent in ACTION_INTENTS
]

def classify_intent(question: str):

    question_embedding = embeddings.embed_query(question)

    information_similarities = []
    action_similarities = []

    for information_embedding in information_embeddings:

        similarity = np.dot(
            question_embedding,
            information_embedding
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(information_embedding)
        )

        information_similarities.append(similarity)

    for action_embedding in action_embeddings:

        similarity = np.dot(
            question_embedding,
            action_embedding
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(action_embedding)
        )

        action_similarities.append(similarity)

    information_similarity = max(information_similarities)
    action_similarity = max(action_similarities)

    print("\nINTENT CLASSIFICATION:")
    print("Information similarity:", information_similarity)
    print("Action similarity:", action_similarity)

    if information_similarity >= action_similarity:
        return "information"

    return "action"
# -------------------------
# 3. Create Chroma vector store
# -------------------------

vector_store = Chroma(
    collection_name="faq_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# -------------------------
# 4. Store FAQs
# -------------------------

ids = [f"faq-{i}" for i in range(len(faqs))]

existing = vector_store.get()["ids"]

new_documents = []
new_metadatas = []
new_ids = []

for faq, faq_id in zip(faqs, ids):

    if faq_id not in existing:

        new_documents.append(faq["question"])

        new_metadatas.append({
            "answer": faq["answer"]
        })

        new_ids.append(faq_id)


if new_documents:
    vector_store.add_texts(
        texts=new_documents,
        metadatas=new_metadatas,
        ids=new_ids
    )


# -------------------------
# 5. Retrieve FAQ
# -------------------------

def retrieve_faq(question: str):
    results = vector_store.similarity_search_with_score(
        question,
        k=1
    )

    if not results:
        print("RAG: No results")
        return None

    document, score = results[0]

    print("\nRAG QUERY:")
    print(repr(question))

    print("RAG SCORE:")
    print(score)

    print("RAG MATCH:")
    print(document.page_content)

    if score > 0.6:
        print("RAG: Rejected")
        return None

    print("RAG: FAQ FOUND")
    return document.metadata["answer"]

if __name__ == "__main__":
    print(classify_intent("How long do I have to send something back?"))
    print(classify_intent("Can I cancel my purchase?"))
    print(classify_intent("What are the requirements for a return?"))
    print(classify_intent("Where is my package right now?"))
    print(classify_intent("Do you offer exchanges?"))
    print(classify_intent("I want to cancel order ORD-1001."))