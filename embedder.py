import chromadb
from sentence_transformers import SentenceTransformer
from chunker import chunk_all_documents

COLLECTION_NAME = "azure_vmss_docs"
DB_PATH = "./chroma_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def embed_and_store(chunks: list[dict]) -> None:
    collection = get_collection()

    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} chunks. Clearing and re-embedding.")
        collection.delete(where={"source": {"$ne": ""}})

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")

if __name__ == "__main__":
    chunks = chunk_all_documents()
    embed_and_store(chunks)
    print("\nDone. Vector store is ready.")