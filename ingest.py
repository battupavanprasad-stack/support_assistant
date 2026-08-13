
import os

import chromadb
from sentence_transformers import SentenceTransformer


DOCS_PATH = "/content/support_assistant/docs"
CHROMA_PATH = "/content/support_assistant/chroma_db"


def load_documents():
    documents = []
    document_ids = []

    for filename in sorted(os.listdir(DOCS_PATH)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_PATH, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())

            document_ids.append(filename)

    return documents, document_ids


def create_database():
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    documents, document_ids = load_documents()

    print(f"Loaded {len(documents)} documents.")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="zepto_policies",
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=document_ids,
        documents=documents,
        embeddings=embeddings
    )

    print("Documents added to ChromaDB.")
    print("Total documents:", collection.count())


if __name__ == "__main__":
    create_database()
