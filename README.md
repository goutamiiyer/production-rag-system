# Production RAG System

A retrieval-augmented generation pipeline built from scratch on Azure
Virtual Machine Scale Sets documentation. No LangChain. Every component
is hand-built so the architecture is fully transparent and explainable.

## What it does

Takes a question in plain English, retrieves the most semantically
relevant documentation chunks from a vector store, and generates a
grounded answer using only the retrieved context. Refuses to answer
when the context is insufficient.

## Architecture

```
documents/ (14 txt files)
    → chunker.py     (split into 500-word chunks with 50-word overlap)
    → embedder.py    (sentence-transformers all-MiniLM-L6-v2 → ChromaDB)
    → retriever.py   (cosine similarity search, top-k chunks)
    → generator.py   (Groq/Llama 3.1 with retrieved context as prompt)
    → rag_eval.py    (LLM-as-judge evaluation against ground truth)
```

## Why no LangChain

LangChain abstracts away every design decision. Building each layer
manually means every architectural choice is explicit and defensible
in a system design interview: why 500-word chunks, why cosine
similarity, why top-3 retrieval, why this prompt structure.

## Eval results

10 ground truth questions against the Azure VMSS corpus:

| Metric | Score |
|---|---|
| Pass rate | 9/10 (90%) |
| Average eval score | 0.85 |
| Average retrieval similarity | 0.64 |

Key finding: retrieval similarity above 0.65 correlates strongly with
correct answers. Both failures occurred with similarity below 0.60,
indicating the retrieval layer is the primary quality bottleneck,
not the generation layer.

## Key design decisions

**Chunk size: 500 words with 50-word overlap**
Large enough to preserve context within a single concept. Overlap
prevents answers from being split across chunk boundaries. Smaller
chunks improve retrieval precision but lose context; larger chunks
reduce precision but improve coherence. This is a tunable tradeoff.

**Top-k: 3 chunks**
Enough context for most factual questions without overwhelming the
prompt. Increasing to 5 improves recall on complex questions at the
cost of more noise in the context window.

**Embedding model: all-MiniLM-L6-v2**
90MB, runs locally, no API cost. Good enough for technical
documentation retrieval. A larger model like all-mpnet-base-v2
would improve similarity scores at the cost of slower inference.

**Known limitation: noisy source attribution**
Low-similarity queries sometimes retrieve chunks from irrelevant
documents as third results. The answer remains correct because the
relevant chunk ranks first, but source attribution includes noise.
Hybrid search (dense + BM25) would reduce this.

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/production-rag-system
cd production-rag-system
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env

python embedder.py      # ingest and embed documents
python generator.py     # test question answering
python rag_eval.py      # run evaluation suite
```

## Corpus

14 Azure documentation pages covering Virtual Machine Scale Sets,
availability zones, health probes, resilient operations, and
container instances. Chosen because the domain is specific enough
to test retrieval precision without ambiguity.

## Tech stack

- Python 3.11
- ChromaDB (local persistent vector store)
- sentence-transformers all-MiniLM-L6-v2 (embeddings)
- Groq API with Llama 3.1 8B (generation, free tier)
- LLM-as-judge evaluation pattern

## Author

Goutami, software engineer with background in distributed systems
and backend infrastructure at Microsoft Azure, building toward
applied AI engineering roles. This project is part of a public
portfolio documenting that transition.