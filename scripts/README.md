# scripts/

Runnable utilities and orchestration for the reconstruction project. All are invoked
from the repo root (paths inside assume the current working directory is the project
root). Scripts run with `uv run python scripts/<name>.py …`; shell scripts with
`bash scripts/<name>.sh …`.

## Orchestration

- `pipeline.sh` — Single end-to-end run: ETL/Parse → Classify → Cluster → Evaluate.
  Auto-detects dataset format and resumes finished steps.
- `agree.sh` — Wrapper around the `agree` command with shared flag names and
  input-directory pre-validation.

## Utility scripts

- `generate_network.py` — Export an evaluation log (`reports/evaluations/`) to
  nodes/edges CSV for the network visualizer.
  `uv run python scripts/generate_network.py --eval-log reports/evaluations/<file>.json`
- `dump_prompt.py` — Render the exact prompt (and, with `--payload`, an
  OpenAI-compatible request body) for one page, for offline debugging.
  `uv run python scripts/dump_prompt.py --dataset <ds> --page-id <id> --prompt prompts/v01.01.02.md --payload`
- `fragment_stats_report.py` — CLI wrapper over
  `newspaper_reconstructor.fragment_stats`: corpus stats and script mix for a
  fragments dir/file set (`--csv`, `--group`, `--ascii-threshold`).
  `uv run python scripts/fragment_stats_report.py data/1_interim/<ds>/fragments`
- `token_lookup.py` — Look up the string form of a tokenizer token id. Requires the
  local transformers dependency group: `uv run --group local python scripts/token_lookup.py <model> <token_id>`
- `migrate_experiment_ids.py` — Backfill dataset name into `experiment_id` fields of
  existing evaluation logs.
