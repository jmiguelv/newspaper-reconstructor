# Agents

## Project Overview

Article reconstruction tool that parses OCR text fragments (ALTO XML) from Jawi Malay newspapers, sends them to an LLM to reconstruct complete articles, and evaluates results against ground truth using pairwise clustering F1, class accuracy, and coverage metrics.

## Commands

```bash
uv sync                  # install dependencies
uv run pytest           # run all tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Architecture

```
ALTO XML → main.py parse (extract JSON fragments)
         → main.py classify (enrich JSON with predicted classes via LLM)
         → main.py cluster (reconstruct JSON fragments into articles via LLM)
         → main.py evaluate (compare against ground truth article XML)
         → dashboard.html (visualize eval logs as HTML)
```

### Module roles

| Module                  | Responsibility                                          |
|-------------------------|---------------------------------------------------------|
| `main.py`               | Typer CLI entry point (parse, classify, cluster, evaluate, suggest) |
| `pipeline.sh`           | Bash script to run a single end-to-end evaluation pipeline |
| `run_experiments.sh`    | Bash script to orchestrate multiple batched grid-search evaluations |
| `reconstruct.py`        | Data transformation, dict parsing, and mapping to LLM inputs |
| `llm.py`                | LLM client wrapper (OpenAI-compatible API), client factory |
| `evaluate.py`           | Ground truth parsing, clustering F1, class accuracy, coverage |
| `dashboard.html`        | Interactive Alpine.js HTML dashboard to visualize JSON eval logs |
| `generate_network.py`   | Exports evaluation JSON to nodes/edges CSV for network visualizer |

## Code Conventions

- **CRITICAL RULE**: ALWAYS run `uv run ruff check . --fix && uv run ruff format .` after making any Python code changes.
- Python 3.12+
- `ruff` for lint and format (no separate formatter)
- TDD: tests in `tests/` mirror module names (`test_reconstruct.py`, `test_evaluate.py`, `test_e2e.py`)
- No comments unless explicitly requested
- Environment variables use `LLM_*` prefix (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), not `OPENAI_*`
- API key defaults to `"none"` (local servers ignore it); model is required
- Pipeline inputs/outputs explicitly pass directories (e.g. `-i input_folder -o output_folder`)

## Data Layout

```
data/
├── 0_external/           # Raw external datasets
│   └── <dataset_name>/   # E.g. ds-filteredUM1956alto (git submodule)
│       ├── alto/         # ALTO XML files (OCR text fragments)
│       └── article_xml/  # Ground truth article XML files
└── 1_interim/            # Interim pipeline I/O
    └── <dataset_name>/
        ├── fragments/        # Parsed JSON fragments
        ├── classified/       # JSON fragments enriched with 'predicted_class'
        └── reconstructions/  # LLM clustered articles (organized by experiment_id)
reports/
├── evaluations/          # Evaluation logs
├── networks/             # Exported nodes/edges CSV for the network visualizer
└── suggestions/          # Output from the LLM judge
```

- ALTO and article XML filenames must match (e.g., `UM-1956-01-09-6.xml` in both dirs)
- Fragment IDs in article XML `<Region ref="...">` must match `TextBlock ID` in ALTO XML
- `data/0_external/` is a git submodule pointing to `culturalheritagenus/ds-filteredUM1956alto`

## Testing Notes

- Unit tests use `tmp_path` fixtures with synthetic XML — no external data needed
- E2E tests mock `main.make_client` with canned LLM responses — no API key needed
- Real data smoke tests (`TestE2ERealData`) skip if `data/0_external/` is absent
