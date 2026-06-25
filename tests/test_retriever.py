import sys
sys.path.insert(0, ".")

from chunker import chunk_all_documents
from embedder import embed_and_store, get_collection
from retriever import retrieve

chunks = chunk_all_documents()
embed_and_store(chunks)

collection = get_collection()
assert collection.count() == 58, f"Expected 58, got {collection.count()}"

results = retrieve("How do health probes work?", top_k=3)
assert len(results) == 3
assert results[0]["similarity"] > 0.4
print(f"Retriever OK: top similarity {results[0]['similarity']}")