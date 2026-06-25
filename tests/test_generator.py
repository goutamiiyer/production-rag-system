import sys
sys.path.insert(0, ".")

from generator import generate_answer

result = generate_answer("What health probe protocols does Azure Load Balancer support?")
assert result["answer"] is not None
assert len(result["answer"]) > 20
assert result["top_similarity"] > 0.5
print(f"Generator OK: answer length {len(result['answer'])} chars")
print(f"Sources: {result['sources']}")