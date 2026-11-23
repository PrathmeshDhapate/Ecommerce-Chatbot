
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ============================================================
# PATHS AND ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "faq_data" / "faqs.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")
load_dotenv()

COLLECTION_NAME = "faq_collection"


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# CHROMA
# ============================================================

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


# ============================================================
# RAG THRESHOLD
# ============================================================

RAG_THRESHOLD = 0.7


# ============================================================
# LOAD FAQ DATA
# ============================================================

def load_faq_documents():
    """Read FAQs from JSON and convert them to Documents."""

    if not FAQ_FILE.exists():
        raise FileNotFoundError(
            f"FAQ file not found: {FAQ_FILE}"
        )

    with FAQ_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    """
    Basically we convert the json from the faq.json to python objects stored in 
    variable data in the python memeory.Next if it is a dictionary then we are 
    extracting the lists from it and storing it in data.If no list was found
    then we are just giving an error.The items inside the list are dictionaries.
    """
    if isinstance(data, dict):
        for key in ("faqs", "FAQs", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            "Unexpected FAQ JSON structure. "
            "Check faqs.json before indexing."
        )

    documents = []

    for item in data:
        if not isinstance(item, dict):
            continue

        question = (
            item.get("question")
            or item.get("Question")
            or item.get("query")
        )

        answer = (
            item.get("answer")
            or item.get("Answer")
            or item.get("response")
        )

        if not isinstance(question, str) or not question.strip():
            continue

        if not isinstance(answer, str) or not answer.strip():
            continue

        documents.append(
            Document(
                page_content=question.strip(),
                metadata={
                    "answer": answer.strip()
                },
            )
        )

    if not documents:
        raise ValueError(
            "No valid question-answer pairs found in faqs.json. "
            "Check its actual JSON field names and structure."
        )

    return documents


# ============================================================
# INITIALIZE / REPAIR EMPTY COLLECTION
# ============================================================

def ensure_faq_indexed():
    """Populate Chroma only when the collection is empty."""

    count = vector_store._collection.count()

    print(f"Existing FAQ document count: {count}")

    if count == 0:
        documents = load_faq_documents()

        print(
            f"Indexing {len(documents)} FAQs into Chroma..."
        )

        vector_store.add_documents(documents)

        print("FAQ indexing complete.")

    print(
        "Current FAQ document count:",
        vector_store._collection.count(),
    )


# ============================================================
# RETRIEVE FAQ
# ============================================================

def retrieve_faq(query):
    """
    Retrieve relevant FAQ candidates.

    Returns a list of candidates, or None if no candidates
    meet the configured distance threshold.
    """

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)
    print(f"\nRAG QUERY:\n'{query}'")

    start = time.perf_counter()

    # Repair an empty collection before searching.
    ensure_faq_indexed()

    results = vector_store.similarity_search_with_score(
        query,
        k=5,
    )

    faq_candidates = []

    if not results:
        elapsed = time.perf_counter() - start
        print("RAG: No results found")
        print(f"RAG retrieval time: {elapsed:.3f}s")
        return None

    for document, score in results:
        print("\n----------------------------------------")
        print(f"RAG SCORE:\n{score}")
        print(f"RAG MATCH:\n{document.page_content}")

        if score <= RAG_THRESHOLD:
            answer = document.metadata.get("answer", "")

            faq_candidates.append({
                "question": document.page_content,
                "answer": answer,
                "score": score,
            })

            print("RAG: Candidate accepted")
        else:
            print("RAG: Candidate rejected")

    elapsed = time.perf_counter() - start

    print("\n----------------------------------------")
    print(f"RAG candidates found: {len(faq_candidates)}")
    print(f"RAG retrieval time: {elapsed:.3f}s")

    if not faq_candidates:
        return None

    return faq_candidates


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    candidates = retrieve_faq(
        "What is the return policy?"
    )

    print("\nFINAL TEST RESULT:")
    print(candidates)