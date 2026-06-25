import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

def cmd_ingest(args):
    from chunker import chunk_all_documents
    from embedder import embed_and_store

    print("Starting ingestion pipeline...")
    print(f"Documents directory: {args.docs_dir}\n")

    chunks = chunk_all_documents(args.docs_dir)
    embed_and_store(chunks)

    print("\nIngestion complete. Vector store is ready.")

def cmd_query(args):
    from generator import generate_answer

    if not args.question:
        print("Error: provide a question with --question or as a positional argument")
        return

    print(f"\nSearching documents for: {args.question}\n")
    result = generate_answer(args.question, top_k=args.top_k)

    print(f"Answer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")
    print(f"Top retrieval similarity: {result['top_similarity']}")

    if result['top_similarity'] < 0.5:
        print("\nWarning: low retrieval similarity. Answer may not be grounded.")

def cmd_eval(args):
    from rag_eval import run_rag_eval
    import json

    print("Running RAG evaluation suite...\n")
    results = run_rag_eval()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

def cmd_stats(args):
    from embedder import get_collection

    collection = get_collection()
    count = collection.count()

    print(f"\nVector store stats:")
    print(f"  Total chunks : {count}")

    if count > 0:
        sample = collection.get(limit=3, include=["metadatas"])
        sources = list(set(m["source"] for m in sample["metadatas"]))
        print(f"  Sample sources: {', '.join(sources)}")

    print(f"  DB path      : ./chroma_db")
    print(f"  Collection   : azure_vmss_docs")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Production RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_cli.py ingest
  python rag_cli.py ingest --docs-dir ./documents
  python rag_cli.py query "How do health probes work?"
  python rag_cli.py query "What is resilient delete?" --top-k 5
  python rag_cli.py eval
  python rag_cli.py eval --output results.json
  python rag_cli.py stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument(
        "--docs-dir",
        default="documents",
        help="Directory containing .txt documents (default: documents)"
    )

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)"
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation suite")
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (e.g. --output results.json)"
    )

    subparsers.add_parser("stats", help="Show vector store statistics")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "eval": cmd_eval,
        "stats": cmd_stats
    }

    commands[args.command](args)