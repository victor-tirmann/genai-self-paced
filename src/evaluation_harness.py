from dataclasses import dataclass
from datetime import datetime
from typing import Any
from src.story_generation import StoryResponse
from src.evaluation_rules import validate_story_structure


@dataclass
class EvaluationResult:
    sample_id: str
    prompt: str
    story: dict[str, Any]
    passed: bool
    errors: list[str]
    metrics: dict[str, Any]
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "story": self.story,
            "passed": self.passed,
            "errors": self.errors,
            "metrics": self.metrics,
            "evaluated_at": self.evaluated_at,
        }


class Evaluator:
    def evaluate(self, sample_id: str, prompt: str, response: StoryResponse) -> EvaluationResult:
        validation = validate_story_structure(response)
        return EvaluationResult(
            sample_id=sample_id,
            prompt=prompt,
            story=response.model_dump(),
            passed=validation["passed"],
            errors=validation["errors"],
            metrics=validation["metrics"],
            evaluated_at=datetime.utcnow().isoformat() + "Z",
        )
