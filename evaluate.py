from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.generator import answer_question


DEFAULT_TESTS = [
    {
        "question": "What is the employee leave policy?",
        "expected_keywords": ["leave"],
    },
    {
        "question": "What is the refund policy?",
        "expected_keywords": ["refund"],
    },
    {
        "question": "What is the company policy for something not in the documents?",
        "expected_keywords": ["could not find"],
    },
]


def load_tests(path: Path | None) -> list[dict]:
    if not path:
        return DEFAULT_TESTS
    return json.loads(path.read_text(encoding="utf-8"))


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    normalized = answer.lower()
    return all(keyword.lower() in normalized for keyword in keywords)


def run_evaluation(tests: list[dict]) -> dict:
    results = []
    hits = 0

    for test in tests:
        response = answer_question(test["question"])
        passed = keyword_hit(response["answer"], test.get("expected_keywords", []))
        hits += int(passed)
        results.append(
            {
                "question": test["question"],
                "answer": response["answer"],
                "sources": response["sources"],
                "confidence": response["confidence"],
                "passed_keyword_check": passed,
            }
        )

    return {
        "total": len(tests),
        "passed": hits,
        "keyword_accuracy": round(hits / max(len(tests), 1), 2),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple RAG quality checks.")
    parser.add_argument("--tests", type=Path, help="Path to a JSON test file.")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args()

    report = run_evaluation(load_tests(args.tests))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
