import sys
sys.path.insert(0, ".")

from chunker import chunk_all_documents

chunks = chunk_all_documents()
assert len(chunks) == 58, f"Expected 58 chunks, got {len(chunks)}"
assert chunks[0]["source"] is not None
assert chunks[0]["text"] is not None
print(f"Chunker OK: {len(chunks)} chunks")