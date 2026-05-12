from typing import Any
from src.story_generation import StoryResponse


def count_words(text: str) -> int:
    return len(text.strip().split())


def validate_story_structure(response: StoryResponse) -> dict[str, Any]:
    errors: list[str] = []
    metrics: dict[str, Any] = {}

    story_text = response.story or ""
    title_text = response.title or ""
    mood_text = response.mood or ""
    keywords_list = response.keywords or []

    word_count = count_words(story_text)
    metrics["word_count"] = word_count

    if word_count > 100:
        errors.append("word_count_exceeded")

    if not title_text.strip():
        errors.append("missing_title")

    if not mood_text.strip():
        errors.append("missing_mood")

    keyword_values = [str(k).strip() for k in keywords_list if str(k).strip()]
    if not keyword_values:
        errors.append("missing_keywords")

    return {"passed": not errors, "errors": errors, "metrics": metrics}
