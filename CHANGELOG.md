## 0.7.0 (2026-08-13)

### Feat

- add CLI options for LLM providers and saving raw prompts
- **llm**: add provider configuration and fix malformed response handling
- **dashboard**: independent sort, pipeline links, richer summary

### Fix

- **evaluate**: create parent directories before writing evaluation logs

### Refactor

- support provider prefixes in experiment IDs

## 0.6.0 (2026-08-11)

### Feat

- **suggest**: add --focus flag to analyze classification and clustering errors
- **dashboard**: layout restructure and experiment linking
- **eval**: add adjusted rand index (ARI) metric
- **ui**: redesign dashboard with split classification and reconstruction views and dynamic layout
- split classification and reconstruction evaluation and record metadata

### Fix

- **dashboard**: allow text to wrap in page details
- **dashboard**: resolve overflow in gt card and invisible chip text
- ignore metadata JSON files during processing and correct classify prompt

### Refactor

- rename runs to experiments

## v1.0.0 (2026-08-07)

### Feat

- **dashboard**: refactor to single-page dashboard.html
- decouple classify and cluster prompts in orchestrator scripts and enable classification caching
- introduce classify_v00.md prompt and integrate classification step into pipeline
- add support for parameterized datasets in pipeline orchestrator scripts
- add support for processing a single page in pipeline and CLI
- implement prompt pipelining using typer
- replace rigid sort with externalized proximity sort
- add --sort-fragments option to sort before reconstruction
- **dashboard**: show run ID and format dates in expanded view

### Fix

- initialize timestamp before early return to prevent UnboundLocalError
- construct results as list of dicts in evaluate command to match log_evaluation_run signature
- add missing run_id argument to log_evaluation_run to fix TypeError
- use named kwargs when calling make_client to avoid positional argument mismatch
- change provider prefix in run_id from openai_ to create_
- use column-aware RTL heuristic for --sort-fragments

### Refactor

- remove redundant inline prompt CLI args

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
