# Repository Guidelines

## Project Structure & Module Organization
- `test.py` is the primary script; it calls the OpenAI-compatible client against DeepSeek’s API.
- `env.ipynb` is an empty placeholder notebook.
- There are no dedicated `src/`, `tests/`, or `assets/` directories yet.

## Build, Test, and Development Commands
- `python test.py`: Runs the script and prints forward/backward average log-probabilities.
- `pip install openai`: Installs the only required dependency.

Environment variables:
- `DEEPSEEK_API_KEY`: Required API key.
- `DEEPSEEK_BASE_URL`: Optional base URL (defaults to `https://api.deepseek.com/beta`).

Example:
```bash
$env:DEEPSEEK_API_KEY="..."
python test.py
```

## Coding Style & Naming Conventions
- Python only; follow PEP 8 (4-space indentation).
- Prefer explicit, descriptive names like `average_logprob_for_continuation`.
- Keep helper functions private with a leading underscore (e.g., `_field`, `_continuation_indices`).

## Testing Guidelines
- No test framework or tests are present.
- If you add tests, place them under `tests/` and use `pytest` naming conventions (e.g., `test_scoring.py`).

## Commit & Pull Request Guidelines
- No Git history is available in this directory, so no commit message conventions are established.
- For PRs, include a short summary, usage instructions, and any new environment variables.

## Configuration Tips
- The script expects an OpenAI-compatible API and uses `MODEL = "deepseek-chat"`.
- If you change the model or base URL, keep them centralized near the top of `test.py`.
