# Agents

## Project Overview

Article reconstruction tool that parses OCR text fragments (ALTO XML) from Jawi Malay newspapers, sends them to an LLM to reconstruct complete articles, and evaluates results against ground truth using pairwise clustering F1, class accuracy, and coverage metrics.

## Commands

```bash
uv sync                  # install dependencies
uv run pytest           # run all 63 tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Architecture

```
ALTO XML → reconstruct.py (parse fragments)
         → llm.py (send to LLM, get JSON response)
         → reconstruct.py (parse JSON into items)
         → evaluate.py (compare against ground truth article XML)
         → main.py (CLI orchestration) / run_evals.sh (batch orchestration)
         → generate_dashboard.py (visualize eval logs as HTML)
```

### Module roles

| Module          | Responsibility                                          |
|-----------------|---------------------------------------------------------|
| `main.py`        | CLI entry point, argument parsing, mode dispatch        |
| `run_evals.sh`   | Bash script to orchestrate batched grid-search evaluations |
| `reconstruct.py`| ALTO XML parsing, fragment loading, article reconstruction, default prompts |
| `llm.py`        | LLM client wrapper (OpenAI-compatible API), client factory |
| `evaluate.py`   | Ground truth parsing, clustering F1, class accuracy, coverage, run logging |
| `generate_dashboard.py`| Generates interactive Alpine.js HTML dashboard from JSON eval logs |

## Code Conventions

- Python 3.12+
- `ruff` for lint and format (no separate formatter)
- TDD: tests in `tests/` mirror module names (`test_reconstruct.py`, `test_evaluate.py`, `test_e2e.py`)
- No comments unless explicitly requested
- Environment variables use `LLM_*` prefix (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), not `OPENAI_*`
- API key defaults to `"none"` (local servers ignore it); model is required
- Prompts are customizable: `reconstruct_articles` accepts `system_prompt` and `user_prompt_template` params (defaulting to `DEFAULT_SYSTEM_PROMPT` / `DEFAULT_USER_PROMPT_TEMPLATE`)
- CLI passes custom prompts via `--system-prompt` and `--user-prompt-template` flags, or `--prompt-file` to read the system prompt from a file
- Prompt name (derived from `--prompt-file` filename stem, default `"default"`) is included in eval log filenames

## Data Layout

```
data/
├── 0_external/   # Raw external data (git submodule)
│   ├── alto/         # 80 ALTO XML files (OCR text fragments)
│   ├── article_xml/  # 80 ground truth article XML files
│   ├── input.csv
│   └── metadata.yaml
├── 0_prompts/      # Prompt files (v01=baseline, v02+=improved variants)
├── 1_interim/      # Interim processed data
│   ├── fragments/          # Cached ALTO→JSON fragments
│   └── reconstructions/    # LLM output caches (by prompt/model)
└── 2_evaluations/      # Evaluation logs
```

- ALTO and article XML filenames must match (e.g., `UM-1956-01-09-6.xml` in both dirs)
- Fragment IDs in article XML `<Region ref="...">` must match `TextBlock ID` in ALTO XML
- `data/0_external/` is a git submodule pointing to `culturalheritagenus/ds-filteredUM1956alto`

## Testing Notes

- Unit tests use `tmp_path` fixtures with synthetic XML — no external data needed
- E2E tests mock `main.make_client` with canned LLM responses — no API key needed
- Real data smoke tests (`TestE2ERealData`) skip if `data/0_external/` is absent
- All 63 tests pass in <1s
