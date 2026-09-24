import json
from pathlib import Path

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