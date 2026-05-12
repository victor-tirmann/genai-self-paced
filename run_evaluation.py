import argparse
import json
from pathlib import Path
from src.story_generation import generate_story
from src.evaluation_harness import Evaluator
from src.evaluation_output import write_jsonl, write_summary


def load_prompts(file_path: Path, limit: int = 10) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if "id" not in item or "prompt" not in item:
                raise ValueError(f"Each line must include 'id' and 'prompt': line {line_number}")

            prompts.append({"id": str(item["id"]), "prompt": str(item["prompt"])})
            if len(prompts) >= limit:
                break

    if not prompts:
        raise ValueError("No prompts found in the input JSONL file.")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSONL-based evaluation for story generation.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/test_prompts.jsonl"),
        help="Path to the JSONL file containing prompts.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/results.jsonl"),
        help="Path where per-sample evaluation results will be written.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/summary_report.json"),
        help="Path where the summary report will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of prompts to evaluate.",
    )

    args = parser.parse_args()
    prompts = load_prompts(args.prompts, limit=args.limit)

    evaluator = Evaluator()
    results = []

    for item in prompts:
        sample_id = item["id"]
        prompt = item["prompt"]
        print(f"Evaluating sample {sample_id}: {prompt}")
        response = generate_story(prompt)
        evaluation = evaluator.evaluate(sample_id, prompt, response)
        results.append(evaluation)

    write_jsonl(args.results, results)
    write_summary(args.summary, results)

    print(f"Wrote {len(results)} evaluation records to {args.results}")
    print(f"Wrote summary report to {args.summary}")

    with open(args.summary, 'r') as f:
        summary = json.load(f)
        print("\nSummary Report:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
