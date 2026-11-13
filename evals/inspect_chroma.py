
from rag import vector_store

stored = vector_store._collection.get(
    include=["documents", "metadatas"]
)

documents = stored["documents"]
metadatas = stored["metadatas"]
ids = stored["ids"]

print("Total stored documents:", len(documents))

for doc_id, content, metadata in zip(
    ids, documents, metadatas
):
    print("\nID:", doc_id)
    print("Content:", repr(content))
    print("Metadata:", metadata)
