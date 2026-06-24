import chromadb
from sentence_transformers import SentenceTransformer
from embedder import COLLECTION_NAME, DB_PATH

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    collection = get_collection()
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "distance": results["distances"][0][i],
            "similarity": round(1 - results["distances"][0][i], 4)
        })

    return chunks

def print_results(query: str, chunks: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print(f"Top {len(chunks)} results:\n")
    for i, chunk in enumerate(chunks):
        print(f"Result {i+1}")
        print(f"  Source    : {chunk['source']}")
        print(f"  Similarity: {chunk['similarity']}")
        print(f"  Preview   : {chunk['text'][:200]}")
        print()

if __name__ == "__main__":
    test_queries = [
        "How does automatic zone balance work in VMSS?",
        "What health probe protocols does Azure Load Balancer support?",
        "How do I migrate a regional VM to an availability zone?",
        "What happens when resilient delete fails?"
    ]

    for query in test_queries:
        chunks = retrieve(query)
        print_results(query, chunks)
        print("-" * 60)