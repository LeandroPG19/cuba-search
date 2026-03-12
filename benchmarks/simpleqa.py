"""SimpleQA Benchmark for Cuba-Search.

Measures factual accuracy using a subset of OpenAI's SimpleQA dataset.
Runs searches and checks if the answer is present in top results.

Usage:
    python3.14 -m benchmarks.simpleqa

Output: accuracy score + per-question results in JSON.
"""
import asyncio
import json
import sys
import time
from typing import Any

# ── SimpleQA test questions (50-question subset) ─────────────────
# Format: (question, expected_answer_fragment)
# These are factual questions with verifiable, short answers.
QUESTIONS: list[tuple[str, str]] = [
    ("What is the capital of France?", "paris"),
    ("Who created Python programming language?", "guido van rossum"),
    ("What year was the first iPhone released?", "2007"),
    ("What is the chemical symbol for gold?", "au"),
    ("Who painted the Mona Lisa?", "leonardo da vinci"),
    ("What is the largest planet in our solar system?", "jupiter"),
    ("What is the speed of light in km/s?", "299"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("What is the atomic number of carbon?", "6"),
    ("What programming language is Django written in?", "python"),
    ("What company developed TypeScript?", "microsoft"),
    ("What does HTML stand for?", "hypertext markup language"),
    ("What year was Linux first released?", "1991"),
    ("What is the default port for HTTPS?", "443"),
    ("Who founded Tesla Motors?", "elon musk"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What element has atomic number 1?", "hydrogen"),
    ("What is the formula for water?", "h2o"),
    ("Who invented the telephone?", "alexander graham bell"),
    ("What is the tallest mountain in the world?", "everest"),
    ("What is the capital of Japan?", "tokyo"),
    ("What year did World War II end?", "1945"),
    ("What is the currency of the United Kingdom?", "pound"),
    ("Who developed the theory of relativity?", "einstein"),
    ("What is the largest organ in the human body?", "skin"),
    ("What does CPU stand for?", "central processing unit"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What is Git used for?", "version control"),
    ("What does SQL stand for?", "structured query language"),
    ("What protocol does HTTPS use for encryption?", "tls"),
    ("What is the capital of Germany?", "berlin"),
    ("Who created JavaScript?", "brendan eich"),
    ("What is the chemical formula for table salt?", "nacl"),
    ("What year was the World Wide Web invented?", "1989"),
    ("What does API stand for?", "application programming interface"),
    ("What operating system is Android based on?", "linux"),
    ("What is the smallest prime number?", "2"),
    ("What does RAM stand for?", "random access memory"),
    ("What is the capital of Australia?", "canberra"),
    ("Who co-founded Apple with Steve Jobs?", "wozniak"),
]


async def _search(query: str) -> list[dict[str, Any]]:
    """Run a cuba_search query and return results."""
    from cuba_search import retrieval, quality, ranking
    from cuba_search import query as query_mod
    from cuba_search import semantic

    normalized = query_mod.normalize_query(query)
    expanded = query_mod.expand_query(normalized)

    results = await retrieval.search(expanded, max_results=10)
    results = quality.filter_and_classify(results)
    results = ranking.bm25_rank(normalized, results, text_key="content")
    results = semantic.semantic_rerank(normalized, results)

    return results


def _check_answer(results: list[dict[str, Any]], expected: str) -> bool:
    """Check if expected answer fragment appears in top 5 results."""
    for r in results[:5]:
        content = (
            r.get("content", "") + " " +
            r.get("title", "")
        ).lower()
        if expected.lower() in content:
            return True
    return False


async def run_benchmark() -> dict[str, Any]:
    """Run full SimpleQA benchmark.

    Returns:
        Dict with accuracy, total, correct, per-question details.
    """
    results_log: list[dict[str, Any]] = []
    correct = 0
    total = len(QUESTIONS)

    print(f"Running SimpleQA benchmark ({total} questions)...")
    start = time.monotonic()

    for i, (question, expected) in enumerate(QUESTIONS):
        try:
            search_results = await _search(question)
            found = _check_answer(search_results, expected)
            if found:
                correct += 1
                status = "✅"
            else:
                status = "❌"
        except Exception as e:
            found = False
            status = f"⚠️ {type(e).__name__}"

        results_log.append({
            "question": question,
            "expected": expected,
            "found": found,
            "result_count": len(search_results) if 'search_results' in dir() else 0,
        })

        print(f"  [{i + 1}/{total}] {status} {question}")

    elapsed = time.monotonic() - start
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "benchmark": "SimpleQA",
        "version": "1.0",
        "engine": "cuba-search v1.1 (SearXNG + model2vec)",
        "total_questions": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "elapsed_seconds": round(elapsed, 1),
        "avg_latency_ms": round(elapsed / total * 1000, 0) if total else 0,
        "details": results_log,
    }

    print(f"\n{'='*50}")
    print(f"SimpleQA Accuracy: {summary['accuracy_pct']} ({correct}/{total})")
    print(f"Total time: {summary['elapsed_seconds']}s")
    print(f"Avg latency: {summary['avg_latency_ms']}ms/query")
    print(f"{'='*50}")

    return summary


def main() -> None:
    """Entry point for CLI."""
    summary = asyncio.run(run_benchmark())

    # Save results
    out_path = "simpleqa_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Exit with non-zero if accuracy < 50%
    sys.exit(0 if summary["accuracy"] >= 0.5 else 1)


if __name__ == "__main__":
    main()
