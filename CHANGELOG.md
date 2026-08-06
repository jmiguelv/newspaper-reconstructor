## 0.5.0 (2026-08-06)

### Feat

- **evaluation**: add LLM judge to suggest prompt and heuristic improvements
- **dashboard**: add sortable date/time column to runs table
- **dashboard**: add model, prompt, and sample size filter dropdowns

### Refactor

- restructure directories and module to hybrid CCDS

## v0.4.0 (2026-08-05)

## v0.3.0 (2026-08-05)

## v0.2.0 (2026-08-05)

### Feat

- add page-level evaluation metrics as columns on edges CSV
- add generate_network.py to export eval logs as nodes/edges CSV for the network visualizer
- dynamically load prompts in dashboard generation
- **ui**: add interactive completion ratio filter slider to runs table

### Fix

- update tests for give-up-immediately on 504/timeout; add data/3_networks to .gitignore

### Refactor

- rename edge column evaluation_weight to edge_weight
- use model name in segment column instead of llm_segment

## v0.1.0 (2026-08-05)

### Feat

- **ui**: group reference info and refine execution time formatting
- **metrics**: record and display execution time in runs
- **ui**: add highlights text for best performers
- **ui**: add models reference section to dashboard
- **ui**: add column sorting to per-page metrics table
- **ui**: restore dashboard styling and implement alignment
- note skipped pages in output, add pages_processed to eval config
- add evaluation dashboard generator
- add timeout param and per-model values in run_evals.sh
- add .md prompt file parsing in main.py, change default to v00.md

### Fix

- **eval**: bypass retries on 504 and set strict 60s timeout in script
- **llm**: skip retries on api timeouts to speed up failing runs
- **ui**: wrap fragment ids to prevent overflow

### Refactor

- distinguish APITimeoutError from other API errors in reconstruct.py
