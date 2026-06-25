import json
from generator import generate_answer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_eval_questions(path: str = "eval_questions.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def evaluate_answer(question: str, expected: str, actual: str) -> dict:
    judge_prompt = f"""You are evaluating whether an AI answer correctly addresses a question based on expected information.

Question: {question}
Expected information: {expected}
Actual answer: {actual}

Does the actual answer contain the expected information or convey the same meaning?
Respond in exactly this format:
SCORE: 1.0 if yes, 0.5 if partially, 0.0 if no
REASON: one sentence"""

    judgment = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    text = judgment.choices[0].message.content.strip()

    try:
        score_line = [l for l in text.split("\n") if l.startswith("SCORE:")][0]
        reason_line = [l for l in text.split("\n") if l.startswith("REASON:")][0]
        score = float(score_line.replace("SCORE:", "").strip())
        reason = reason_line.replace("REASON:", "").strip()
    except (IndexError, ValueError):
        score = 0.0
        reason = "Could not parse judge response"

    return {"score": score, "reason": reason}

def run_rag_eval():
    questions = load_eval_questions()
    results = []

    print("\n--- RAG Evaluation ---\n")

    for case in questions:
        rag_result = generate_answer(case["question"])
        eval_result = evaluate_answer(
            case["question"],
            case["expected"],
            rag_result["answer"]
        )

        result = {
            "question": case["question"],
            "expected": case["expected"],
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "retrieval_similarity": rag_result["top_similarity"],
            "eval_score": eval_result["score"],
            "eval_reason": eval_result["reason"]
        }
        results.append(result)

        status = "PASS" if eval_result["score"] >= 0.5 else "FAIL"
        print(f"Question: {case['question'][:70]}")
        print(f"  Status    : {status} (score: {eval_result['score']})")
        print(f"  Retrieval : {rag_result['top_similarity']}")
        print(f"  Reason    : {eval_result['reason']}")
        print()

    avg_eval = sum(r["eval_score"] for r in results) / len(results)
    avg_retrieval = sum(r["retrieval_similarity"] for r in results) / len(results)
    passed = sum(1 for r in results if r["eval_score"] >= 0.5)

    print(f"Results: {passed}/{len(results)} passed")
    print(f"Average eval score     : {avg_eval:.2f}")
    print(f"Average retrieval sim  : {avg_retrieval:.2f}")

    return results

if __name__ == "__main__":
    run_rag_eval()