import json
from collections import Counter
from pathlib import Path
from typing import Iterable
from src.evaluation_harness import EvaluationResult


def write_jsonl(file_path: Path, results: Iterable[EvaluationResult]) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        for result in results:
            json.dump(result.to_dict(), handle, ensure_ascii=False)
            handle.write("\n")


def summarize_results(results: Iterable[EvaluationResult]) -> dict[str, object]:
    results = list(results)
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed
    error_counter = Counter(error for result in results for error in result.errors)
    failure_examples: dict[str, list[dict[str, object]]] = {}

    for result in results:
        for error in result.errors:
            if error not in failure_examples:
                failure_examples[error] = []
            if len(failure_examples[error]) < 3:
                failure_examples[error].append(
                    {
                        "sample_id": result.sample_id,
                        "prompt": result.prompt,
                        "story": result.story,
                        "metrics": result.metrics,
                    }
                )

    return {
        "total_samples": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "error_counts": dict(error_counter),
        "failure_examples": failure_examples,
    }


def write_summary(file_path: Path, results: Iterable[EvaluationResult]) -> None:
    summary = summarize_results(results)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
