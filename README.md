# Production RAG System

A retrieval-augmented generation pipeline built from scratch on Azure
Virtual Machine Scale Sets documentation. No LangChain. Every component
is explicit and explainable.

## Architecture

```
documents/ (14 txt files, ~58 chunks)
    ↓
chunker.py       chunk into 500-word segments with 50-word overlap
    ↓
embedder.py      sentence-transformers → ChromaDB vector store
    ↓
retriever.py     cosine similarity search, top-k chunks
    ↓
generator.py     Groq/Llama 3.1 with retrieved context
    ↓
rag_eval.py      LLM-as-judge evaluation against ground truth
```

## Why no LangChain

LangChain abstracts away every design decision. Building each layer
manually means every architectural choice is explicit and defensible:
why 500-word chunks, why cosine similarity, why top-3 retrieval,
why this prompt structure. These are the questions that come up in
applied AI engineering interviews.

## CLI

```bash
python rag_cli.py ingest                           # embed all documents
python rag_cli.py query "How do health probes work?"
python rag_cli.py query "What is resilient delete?" --top-k 5
python rag_cli.py eval                             # run full eval suite
python rag_cli.py eval --output results.json       # save results
python rag_cli.py stats                            # vector store info
```

## Eval results

10 ground truth questions against the Azure VMSS corpus.
Results vary slightly between runs due to LLM non-determinism.

| Metric | Score |
|---|---|
| Pass rate | 9/10 (90%) |
| Average eval score | 0.80-0.85 |
| Average retrieval similarity | 0.64 |

## Key findings

**Retrieval similarity predicts answer quality.**
Questions with similarity above 0.65 consistently pass. Both
persistent failures occur below 0.60. The retrieval layer is the
primary quality bottleneck, not generation.

**LLM-as-judge scores vary between runs.**
Same questions, same system, different judge scores on re-runs.
This is LLM non-determinism in the evaluation layer itself, not
the RAG system. Production systems use multiple judge passes and
average the scores.

**Noisy source attribution on low-similarity queries.**
When retrieval similarity is low, irrelevant documents appear as
third results. The answer remains correct because the relevant
chunk ranks first, but source attribution is polluted. Hybrid
search (dense + BM25) would reduce this.

**One persistent retrieval failure.**
"What is the maximum retry duration for resilient create?" (answer:
30 minutes) fails consistently at 0.59 similarity. The information
is in the documents but the query phrasing doesn't match the chunk
containing the answer. Query expansion or re-ranking would fix this.

## Design decisions

**Chunk size: 500 words, 50-word overlap**
Large enough to preserve context, small enough for precise
retrieval. Overlap prevents answers from splitting across boundaries.

**Embedding: all-MiniLM-L6-v2**
90MB, runs locally, no API cost. Sufficient for technical
documentation retrieval.

**Vector store: ChromaDB (local persistent)**
Zero infrastructure. Runs on disk. Production would use Pinecone
or Qdrant for scale and query performance.

**Generation: Groq free tier, Llama 3.1 8B**
Provider-agnostic design. Swap the client in generator.py to use
any LLM API without touching retrieval logic.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/production-rag-system
cd production-rag-system
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python rag_cli.py ingest
python rag_cli.py query "How do health probes work in Azure Load Balancer?"
```

## Project structure

```
production-rag-system/
├── documents/                  # 14 Azure VMSS documentation pages
├── chunker.py                  # text splitting with overlap
├── embedder.py                 # embedding and ChromaDB storage
├── retriever.py                # semantic similarity search
├── generator.py                # prompt construction and LLM call
├── rag_eval.py                 # evaluation with LLM-as-judge
├── rag_cli.py                  # unified CLI entry point
├── eval_questions.jsonl        # 10 ground truth QA pairs
├── requirements.txt
└── .devcontainer/
    └── devcontainer.json       # Codespaces auto-setup
```

## Corpus

14 Azure documentation pages covering Virtual Machine Scale Sets,
availability zones, health probes, resilient operations, automatic
instance repairs, zone balancing, and container instances.

## Tech stack

- Python 3.11
- ChromaDB (local persistent vector store)
- sentence-transformers all-MiniLM-L6-v2
- Groq API, Llama 3.1 8B Instant (free tier)
- GitHub Actions CI

## Relation to llm-eval-framework

The evaluation pattern in `rag_eval.py` (LLM-as-judge, score plus
reason, pass/fail threshold) is directly derived from the
[llm-eval-framework](https://github.com/YOUR_USERNAME/llm-eval-framework)
built in Phase 1. The RAG system uses the eval framework's patterns
to measure its own output quality. Infrastructure first, then the
system that gets evaluated.

## Author

Goutami, software engineer with background in distributed systems
and backend infrastructure at Microsoft Azure, building toward
applied AI engineering roles.
