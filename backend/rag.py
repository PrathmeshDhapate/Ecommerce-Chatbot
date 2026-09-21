import json
from pathlib import Path

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from dotenv import load_dotenv

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

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# -------------------------
# 3. Prepare FAQ documents
# -------------------------

documents = [
    f"Question: {faq['question']}\nAnswer: {faq['answer']}"
    for faq in faqs
]


# -------------------------
# 4. Create Chroma vector store
# -------------------------

vector_store = Chroma(
    collection_name="faq_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# -------------------------
# 5. Store FAQs
# -------------------------

ids = [f"faq-{i}" for i in range(len(documents))]

existing = vector_store.get()["ids"]

new_documents = []
new_ids = []

for document, faq_id in zip(documents, ids):
    if faq_id not in existing:
        new_documents.append(document)
        new_ids.append(faq_id)

if new_documents:
    vector_store.add_texts(
        texts=new_documents,
        ids=new_ids
    )


# -------------------------
# 6. Retrieval function
# -------------------------

def retrieve_faq(question: str):

    results = vector_store.similarity_search(
        question,
        k=1
    )

    return results[0].page_content

def generate_answer(question: str, context: str):

    prompt = f"""
Answer the user's question using only the information provided in the FAQ context.

FAQ context:
{context}

User question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content[0]["text"]


# -------------------------
# 7. Test retrieval
# -------------------------

question = "Can I return something?"

faq = retrieve_faq(question)

answer = generate_answer(question, faq)

print(answer)