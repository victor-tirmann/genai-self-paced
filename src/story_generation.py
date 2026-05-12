from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI()

class StoryResponse(BaseModel):
    story: str
    title: str
    mood: str
    keywords: list[str]


def generate_story(prompt: str, max_words: int = 100) -> StoryResponse:
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": "Extract the story information."},
            {
                "role": "user",
                "content": f"Write a short story, no more than {max_words} words, about {prompt}.",
            },
        ],
        text_format=StoryResponse,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("Unable to parse the response into StoryResponse")
    return parsed
