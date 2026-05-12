from src.story_generation import generate_story


def main() -> None:
    story = generate_story("a unicorn")
    print(story.model_dump_json(indent=2))


if __name__ == "__main__":
    main()