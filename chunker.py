import os

def load_document(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end
        })

        start += chunk_size - overlap

    return chunks

def chunk_document(filepath: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    filename = os.path.basename(filepath)
    text = load_document(filepath)
    chunks = chunk_text(text, chunk_size, overlap)

    for i, chunk in enumerate(chunks):
        chunk["source"] = filename
        chunk["chunk_id"] = f"{filename}_chunk_{i}"

    return chunks

def chunk_all_documents(documents_dir: str = "documents") -> list[dict]:
    all_chunks = []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(documents_dir, filename)
            chunks = chunk_document(filepath)
            all_chunks.extend(chunks)
            print(f"Chunked {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_documents()
    print("\nSample chunk:")
    print(f"Source: {chunks[0]['source']}")
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Word count: {chunks[0]['word_count']}")
    print(f"Text preview: {chunks[0]['text'][:200]}")