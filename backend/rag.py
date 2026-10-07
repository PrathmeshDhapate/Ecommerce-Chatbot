
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


# ============================================================
# EMBEDDINGS
# ============================================================

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# CHROMA
# ============================================================

vector_store = Chroma(
    collection_name="faq_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ============================================================
# RAG THRESHOLD
# ============================================================

RAG_THRESHOLD = 0.6


# ============================================================
# RETRIEVE FAQ
# ============================================================

def retrieve_faq(query):
    """
    Retrieve relevant FAQ candidates for a query.

    Returns:
        list of FAQ candidates
        None if no sufficiently relevant FAQ is found
    """

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    print("\nRAG QUERY:")
    print(f"'{query}'")

    start = time.perf_counter()

    results = vector_store.similarity_search_with_score(
        query,
        k=5
    )

    elapsed = time.perf_counter() - start

    if not results:

        print("RAG: No results found")

        print(
            f"RAG retrieval time: "
            f"{elapsed:.3f}s"
        )

        return None

    faq_candidates = []

    for document, score in results:

        print("\n----------------------------------------")

        print(
            f"RAG SCORE:\n{score}"
        )

        print(
            f"RAG MATCH:\n"
            f"{document.page_content}"
        )

        if score <= RAG_THRESHOLD:

            answer = document.metadata.get(
                "answer",
                ""
            )

            faq_candidates.append({
                "question": document.page_content,
                "answer": answer,
                "score": score
            })

            print("RAG: Candidate accepted")

        else:

            print("RAG: Candidate rejected")

    print("\n----------------------------------------")

    print(
        f"RAG candidates found: "
        f"{len(faq_candidates)}"
    )

    print(
        f"RAG retrieval time: "
        f"{elapsed:.3f}s"
    )

    if not faq_candidates:
        return None

    return faq_candidates