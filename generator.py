from groq import Groq
from retriever import retrieve
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join([
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in chunks
    ])

    return f"""You are a technical assistant answering questions about Azure infrastructure.
Use only the context provided below to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
Do not use any knowledge outside of the context.

Context:
{context}

Question: {query}

Answer:"""

def generate_answer(query: str, top_k: int = 3) -> dict:
    chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "sources": list(set(chunk["source"] for chunk in chunks)),
        "chunks_used": len(chunks),
        "top_similarity": chunks[0]["similarity"] if chunks else 0
    }

def print_answer(result: dict) -> None:
    print(f"\nQuestion: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")
    print(f"Top similarity: {result['top_similarity']}")
    print()

if __name__ == "__main__":
    test_questions = [
        "How does automatic zone balance work in VMSS?",
        "What health probe protocols does Azure Load Balancer support?",
        "How do I migrate a regional VM to an availability zone?",
        "What happens when resilient delete fails?"
    ]

    for question in test_questions:
        result = generate_answer(question)
        print_answer(result)
        print("-" * 60)