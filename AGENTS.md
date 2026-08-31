# Agents

## Project Overview

Article reconstruction tool that processes OCR text fragments from Jawi Malay newspapers, sends them to an LLM to reconstruct complete articles, and evaluates results against ground truth using pairwise clustering F1, B-Cubed F1, Adjusted Rand Index (ARI), class accuracy, and coverage metrics.

## Commands

```bash
uv sync                  # install dependencies
uv run pytest           # run all tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Architecture

```
Article JSON → main.py etl (convert {id: text} JSON to fragment lists)
ALTO XML     → main.py parse (extract JSON fragments from ALTO XML — legacy)
             → main.py classify (enrich JSON with predicted classes via LLM)
             → main.py cluster (reconstruct JSON fragments into articles via LLM)
             → main.py evaluate (compare against ground truth article XML)
             → dashboard.html (visualize eval logs as HTML)
```

### Module roles

| Module                                        | Responsibility                                                    |
|-----------------------------------------------|-------------------------------------------------------------------|
| `main.py`                                     | Typer CLI entry point (etl, parse, classify, cluster, evaluate, suggest) |
| `pipeline.sh`                                 | Bash script to run a single end-to-end evaluation pipeline        |
| `experiments/*.sh`                            | Bash scripts to orchestrate multiple batched grid-search evaluations |
| `src/newspaper_reconstructor/ingest.py`       | Load pre-extracted JSON articles into fragment lists              |
| `src/newspaper_reconstructor/reconstruct.py`  | Data transformation, dict parsing, and mapping to LLM inputs     |
| `src/newspaper_reconstructor/llm.py`          | LLM client wrapper (OpenAI-compatible API), client factory        |
| `src/newspaper_reconstructor/evaluate.py`     | Ground truth parsing, clustering F1, ARI, B³ F1, class accuracy, coverage |
| `src/newspaper_reconstructor/suggest.py`      | LLM judge for offline analysis and improvement suggestions        |
| `dashboard.html`                              | Interactive Alpine.js HTML dashboard to visualize JSON eval logs  |
| `generate_network.py`                         | Exports evaluation JSON to nodes/edges CSV for network visualizer |

## Code Conventions

- **CRITICAL RULE**: ALWAYS run `uv run ruff check . --fix && uv run ruff format .` after making any Python code changes.
- Python 3.12+
- `ruff` for lint and format (no separate formatter)
- TDD: tests in `tests/` mirror module names (`test_reconstruct.py`, `test_evaluate.py`, `test_e2e.py`, `test_ingest.py`)
- No comments unless explicitly requested
- Environment variables use `LLM_*` prefix (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), not `OPENAI_*`
- API key defaults to `"none"` (local servers ignore it); model is required
- Pipeline inputs/outputs explicitly pass directories (e.g. `-i input_folder -o output_folder`)

## Data Layout

```
data/
├── 0_external/           # Raw external datasets
│   ├── <dataset_name>/   # New format (e.g. ds-articlereconstruction-20260821)
│   │   ├── articles/     # Pre-extracted JSON: {region_id: ocr_text}
│   │   └── regions/      # Ground truth article XML files
│   └── <dataset_name>/   # Legacy format (e.g. ds-filteredUM1956alto)
│       ├── alto/         # ALTO XML files (OCR text fragments)
│       └── article_xml/  # Ground truth article XML files
└── 1_interim/            # Interim pipeline I/O
    └── <dataset_name>/
        ├── fragments/        # Parsed JSON fragments [{id, text}]
        ├── classified/       # JSON fragments enriched with 'predicted_class'
        └── reconstructions/  # LLM clustered articles (organized by experiment_id)

reports/
├── evaluations/          # Evaluation logs
├── networks/             # Exported nodes/edges CSV for the network visualizer
└── suggestions/          # Output from the LLM judge
```

- pipeline.sh auto-detects dataset format: `articles/` → new JSON format, `alto/` → legacy ALTO XML
- Fragment IDs in ground truth XML `<Region ref="...">` must match keys in article JSON (or `TextBlock ID` in ALTO XML)
- `data/0_external/` datasets may be git submodules

## Testing Notes

- Unit tests use `tmp_path` fixtures with synthetic data — no external data needed
- E2E tests mock `main.make_client` with canned LLM responses — no API key needed
- Real data smoke tests (`TestE2ERealData`, `TestE2ERealDataNewFormat`) skip if datasets are absent
