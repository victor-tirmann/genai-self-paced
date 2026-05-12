# GenAI Self-Paced Chapter 1

This project generates structured story data with OpenAI and evaluates output quality using a JSONL-based harness.

## Project structure

- `main.py` — root entrypoint that runs a single story generation using `src.story_generation`
- `story_generation.py` — moved into `src/`; contains OpenAI client initialization and `StoryResponse` model
- `run_evaluation.py` — root entrypoint that loads prompts from `data/test_prompts.jsonl`, generates stories, evaluates them, and writes results
- `evaluation_rules.py` — moved into `src/`; validation rules for generated responses
- `evaluation_harness.py` — moved into `src/`; evaluator and result data model
- `evaluation_output.py` — moved into `src/`; writes `results.jsonl` and `summary_report.json`
- `data/test_prompts.jsonl` — 10 evaluation prompts
- `requirements.txt` — required Python packages
- `Dockerfile` — image for running `main.py`
- `Dockerfile.eval` — image for running evaluation

## Docker images

### Build the main runtime image

```bash
docker build -f Dockerfile -t genai-main .
```

### Run the main script

```bash
docker run --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" genai-main
```

### Build the evaluation image

```bash
docker build -f Dockerfile.eval -t genai-eval .
```

### Run the evaluation

```bash
docker run --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" genai-eval
```

## Environment variables

The container expects `OPENAI_API_KEY` to be provided at runtime. You can also mount a `.env` file if you prefer, but environment variables are recommended.

## Notes

- `main.py` is the default entrypoint for the main Docker image.
- `Dockerfile.eval` is dedicated to the evaluation harness and uses `run_evaluation.py` by default.
- Results are written to `data/results.jsonl` and `data/summary_report.json` inside the container by default unless you pass custom output paths. If you want them on the host, mount the repo directory as a volume.

## Optional host command with volume mount

```bash
docker run --rm -v "%cd%:/app" -e OPENAI_API_KEY="$OPENAI_API_KEY" genai-eval
```

This writes output files back into the current host directory.
