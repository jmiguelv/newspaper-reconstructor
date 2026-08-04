"""Generate a self-contained evaluation dashboard HTML from eval JSON files.

Usage:
    uv run python generate_dashboard.py
    uv run python generate_dashboard.py --eval-dir data/2_evaluations --output dashboard.html
"""

import argparse
import glob
import json
import os

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="light dark" />
    <title>Evaluation Dashboard — Jawi Newspaper Reconstruction</title>
    <link rel="icon" href="https://fav.farm/📰" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.amber.min.css" />
    <style>
        :root {
            --pico-font-family: "Inter", system-ui, sans-serif;
        }

        main.container { max-width: 100%; }

        table { width: 100%; font-size: 0.85rem; }
        table th { cursor: pointer; white-space: nowrap; user-select: none; }
        table th:hover { color: var(--pico-primary); }
        table tbody tr { cursor: pointer; }
        table tbody tr.active { background-color: var(--pico-secondary-background); }

        .metric-good { color: #2e7d32; font-weight: 600; }
        .metric-ok { color: #f57f17; font-weight: 600; }
        .metric-bad { color: #c62828; font-weight: 600; }
        .metric-na { color: var(--pico-muted-color); }

        .badge {
            border-radius: 0.25rem;
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.1rem 0.5rem;
        }
        .badge-article { background: #e3f2fd; color: #1565c0; }
        .badge-advertisement { background: #fff3e0; color: #e65100; }
        .badge-obituary { background: #f3e5f5; color: #6a1b9a; }
        .badge-miscellaneous { background: #f5f5f5; color: #616161; }

        .chip {
            border-radius: 0.25rem;
            display: inline-block;
            font-family: monospace;
            font-size: 0.7rem;
            margin: 0.1rem;
            padding: 0.1rem 0.35rem;
        }
        .chip-0 { background: #e3f2fd; color: #1565c0; }
        .chip-1 { background: #e8f5e9; color: #2e7d32; }
        .chip-2 { background: #fff3e0; color: #e65100; }
        .chip-3 { background: #fce4ec; color: #c62828; }
        .chip-4 { background: #f3e5f5; color: #6a1b9a; }
        .chip-5 { background: #e0f7fa; color: #00838f; }
        .chip-6 { background: #fff8e1; color: #f57f17; }
        .chip-7 { background: #e8eaf6; color: #283593; }
        .chip-8 { background: #efebe9; color: #4e342e; }
        .chip-9 { background: #e0f2f1; color: #00695c; }
        .chip-10 { background: #f1f8e9; color: #558b2f; }
        .chip-11 { background: #fce4ec; color: #880e4f; }
        .chip-unmatched { background: #f5f5f5; color: #9e9e9e; }

        .comparison-grid {
            grid-template-columns: 1fr 1fr;
            gap: var(--pico-spacing);
        }

        .comparison-grid article {
            margin-bottom: var(--pico-spacing);
        }
        .comparison-grid article.match { border-left: 3px solid #4caf50; }
        .comparison-grid article.mismatch { border-left: 3px solid #f44336; }

        .match-icon { color: #4caf50; font-weight: bold; }
        .mismatch-icon { color: #f44336; font-weight: bold; }

        .topic-tag {
            background: #e0e0e0;
            border-radius: 1rem;
            display: inline-block;
            font-size: 0.7rem;
            margin: 0.1rem;
            padding: 0.1rem 0.5rem;
        }

        .chips { line-height: 1.8; }

        dl.grid { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
        dt { color: var(--pico-muted-color); font-size: 0.8rem; font-weight: bold; }
        dd { margin: 0 0 0.5rem 0; word-break: break-word; }

        .bar-container {
            background: var(--pico-card-background-color);
            border-radius: 0.25rem;
            height: 1.2rem;
            min-width: 60px;
            overflow: hidden;
            position: relative;
        }
        .bar-fill {
            border-radius: 0.25rem;
            height: 100%;
            transition: width 0.3s ease;
        }
        .bar-fill.good { background: #4caf50; }
        .bar-fill.ok { background: #ff9800; }
        .bar-fill.bad { background: #f44336; }
        .bar-label {
            color: var(--pico-color);
            font-size: 0.75rem;
            font-weight: 600;
            left: 0.35rem;
            line-height: 1.2rem;
            position: absolute;
            top: 0;
        }

        .muted { color: var(--pico-muted-color); }

        .error-list {
            max-height: 30vh;
            overflow-y: auto;
        }
        .error-list ul { margin: 0.5rem 0; }
        .error-list li { font-size: 0.8rem; margin: 0.2rem 0; }
    </style>
</head>
<body>
    <header class="container">
        <hgroup>
            <h1>Evaluation Dashboard</h1>
            <p>Jawi Newspaper Article Reconstruction — <span x-data="{count: 0}" x-text="EVAL_DATA.length" x-init="count = EVAL_DATA.length"></span> runs</p>
        </hgroup>
    </header>

    <main class="container" x-data="dashboard()">
        <section>
            <h2>Runs</h2>
            <div class="overflow-auto">
            <table>
                <thead>
                    <tr>
                        <th @click="sort('config.model')">Model <span x-show="sortKey==='config.model'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('config.prompt_name')">Prompt <span x-show="sortKey==='config.prompt_name'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('config.sample_size')">Sample <span x-show="sortKey==='config.sample_size'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.total_pages')">Pages <span x-show="sortKey==='aggregate.total_pages'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.mean_clustering_f1')">F1 <span x-show="sortKey==='aggregate.mean_clustering_f1'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.mean_bcubed_f1')">BC F1 <span x-show="sortKey==='aggregate.mean_bcubed_f1'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.mean_clustering_precision')">Precision <span x-show="sortKey==='aggregate.mean_clustering_precision'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.mean_clustering_recall')">Recall <span x-show="sortKey==='aggregate.mean_clustering_recall'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.mean_coverage')">Coverage <span x-show="sortKey==='aggregate.mean_coverage'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                        <th @click="sort('aggregate.mean_class_accuracy')">Class Acc <span x-show="sortKey==='aggregate.mean_class_accuracy'" x-text="sortDir==='asc'?'\u25B2':'\u25BC'"></span></th>
                    </tr>
                </thead>
                <tbody>
                    <template x-for="run in sortedRuns" :key="run.run_id">
                        <tr @click="toggleRun(run.run_id)" :class="expandedRun === run.run_id ? 'active' : ''">
                            <td x-text="run.config.model"></td>
                            <td x-text="run.config.prompt_name"></td>
                            <td x-text="run.config.sample_size || 'all'"></td>
                            <td x-text="run.aggregate.total_pages"></td>
                            <td :class="metricClass(run.aggregate.mean_clustering_f1)" x-text="fmt(run.aggregate.mean_clustering_f1)"></td>
                            <td :class="metricClass(run.aggregate.mean_bcubed_f1)" x-text="fmt(run.aggregate.mean_bcubed_f1)"></td>
                            <td :class="metricClass(run.aggregate.mean_clustering_precision)" x-text="fmt(run.aggregate.mean_clustering_precision)"></td>
                            <td :class="metricClass(run.aggregate.mean_clustering_recall)" x-text="fmt(run.aggregate.mean_clustering_recall)"></td>
                            <td :class="metricClass(run.aggregate.mean_coverage)" x-text="fmt(run.aggregate.mean_coverage)"></td>
                            <td :class="metricClass(run.aggregate.mean_class_accuracy)" x-text="fmt(run.aggregate.mean_class_accuracy)"></td>
                        </tr>
                    </template>
                </tbody>
            </table>
            </div>
        </section>

        <template x-if="activeRun">
            <section>
                <hgroup>
                    <h2 x-text="activeRun.config.model + ' / ' + activeRun.config.prompt_name + ' / sample ' + (activeRun.config.sample_size || 'all')"></h2>
                    <p x-text="activeRun.timestamp"></p>
                </hgroup>

                <details>
                    <summary>Configuration</summary>
                    <dl class="grid">
                        <div><dt>Provider</dt><dd x-text="activeRun.config.provider"></dd></div>
                        <div><dt>Model</dt><dd x-text="activeRun.config.model"></dd></div>
                        <div><dt>Prompt</dt><dd x-text="activeRun.config.prompt_name"></dd></div>
                        <div><dt>Sample size</dt><dd x-text="activeRun.config.sample_size || 'all'"></dd></div>
                        <div><dt>Seed</dt><dd x-text="activeRun.config.seed || '-'"></dd></div>
                        <div><dt>Base URL</dt><dd x-text="activeRun.config.base_url || '-'"></dd></div>
                        <div><dt>System prompt</dt><dd x-text="truncate(activeRun.config.system_prompt, 500)"></dd></div>
                        <div><dt>User prompt template</dt><dd x-text="truncate(activeRun.config.user_prompt_template, 500)"></dd></div>
                    </dl>
                </details>

                <h3>Per-page metrics</h3>
                <div class="overflow-auto">
                <table>
                    <thead>
                        <tr>
                            <th>Page ID</th>
                            <th>F1</th>
                            <th>BC F1</th>
                            <th>Coverage</th>
                            <th>Class Acc</th>
                            <th>Pred</th>
                            <th>Truth</th>
                            <th>TP</th>
                            <th>FP</th>
                            <th>FN</th>
                        </tr>
                    </thead>
                    <tbody>
                        <template x-for="page in activeRun.pages" :key="page.page_id">
                            <tr @click="togglePage(page.page_id)" :class="expandedPage === page.page_id ? 'active' : ''">
                                <td x-text="page.page_id"></td>
                                <td :class="metricClass(page.metrics.clustering_f1)" x-text="fmt(page.metrics.clustering_f1)"></td>
                                <td :class="metricClass(page.metrics.bcubed_f1)" x-text="fmt(page.metrics.bcubed_f1)"></td>
                                <td :class="metricClass(page.metrics.coverage)" x-text="fmt(page.metrics.coverage)"></td>
                                <td :class="metricClass(page.metrics.class_accuracy)" x-text="fmt(page.metrics.class_accuracy)"></td>
                                <td x-text="page.metrics.num_predicted_items"></td>
                                <td x-text="page.metrics.num_ground_truth_items"></td>
                                <td x-text="page.metrics.tp"></td>
                                <td x-text="page.metrics.fp"></td>
                                <td x-text="page.metrics.fn"></td>
                            </tr>
                        </template>
                    </tbody>
                </table>
                </div>
            </section>
        </template>

        <template x-if="activePage">
            <section>
                <hgroup>
                    <h2 x-text="activePage.page_id"></h2>
                    <p>
                        F1: <code x-text="fmt(activePage.metrics.clustering_f1)"></code>
                        / BC F1: <code x-text="fmt(activePage.metrics.bcubed_f1)"></code>
                        / Coverage: <code x-text="fmt(activePage.metrics.coverage)"></code>
                        / Class Acc: <code x-text="fmt(activePage.metrics.class_accuracy)"></code>
                    </p>
                </hgroup>

                <div class="grid comparison-grid">
                    <div>
                        <h3>Predicted Items (<span x-text="activePage.predicted_items.length"></span>)</h3>
                        <template x-for="(item, idx) in activePage.predicted_items" :key="idx">
                            <article :class="isPredMatch(activePage, item) ? 'match' : 'mismatch'">
                                <header>
                                    <span class="badge" :class="'badge-' + item.class" x-text="item.class"></span>
                                    <span :class="isPredMatch(activePage, item) ? 'match-icon' : 'mismatch-icon'" x-text="isPredMatch(activePage, item) ? '\u2713' : '\u2717'"></span>
                                </header>
                                <h4 x-text="item.title"></h4>
                                <div class="chips">
                                    <template x-for="fid in item.fragment_ids" :key="fid">
                                        <span class="chip" :class="getChipClass(activePage, fid)" x-text="fid"></span>
                                    </template>
                                </div>
                            </article>
                        </template>
                    </div>

                    <div>
                        <h3>Ground Truth Items (<span x-text="(activePage.ground_truth_items || []).length"></span>)</h3>
                        <template x-for="(item, idx) in (activePage.ground_truth_items || [])" :key="idx">
                            <article :class="isTruthMatch(activePage, item) ? 'match' : 'mismatch'">
                                <header>
                                    <span class="badge" :class="'badge-' + item.class" x-text="item.class"></span>
                                    <span :class="isTruthMatch(activePage, item) ? 'match-icon' : 'mismatch-icon'" x-text="isTruthMatch(activePage, item) ? '\u2713' : '\u2717'"></span>
                                </header>
                                <code x-text="item.uuid"></code>
                                <div class="chips">
                                    <template x-for="fid in item.fragment_ids" :key="fid">
                                        <span class="chip" :class="getChipClass(activePage, fid)" x-text="fid"></span>
                                    </template>
                                </div>
                                <template x-if="item.topics && item.topics.length">
                                    <div>
                                        <template x-for="topic in item.topics" :key="topic">
                                            <span class="topic-tag" x-text="topic"></span>
                                        </template>
                                    </div>
                                </template>
                            </article>
                        </template>
                    </div>
                </div>

                <template x-if="activePage.metrics.false_positives.length || activePage.metrics.false_negatives.length">
                    <details>
                        <summary>Errors (<span x-text="activePage.metrics.false_positives.length + activePage.metrics.false_negatives.length"></span>)</summary>
                        <div class="grid">
                            <template x-if="activePage.metrics.false_positives.length">
                                <div class="error-list">
                                    <h4>False Positives (<span x-text="activePage.metrics.false_positives.length"></span>)</h4>
                                    <p class="muted">Fragment pairs incorrectly grouped together</p>
                                    <ul>
                                        <template x-for="(pair, i) in activePage.metrics.false_positives" :key="'fp'+i">
                                            <li><code x-text="pair[0]"></code> &harr; <code x-text="pair[1]"></code></li>
                                        </template>
                                    </ul>
                                </div>
                            </template>
                            <template x-if="activePage.metrics.false_negatives.length">
                                <div class="error-list">
                                    <h4>False Negatives (<span x-text="activePage.metrics.false_negatives.length"></span>)</h4>
                                    <p class="muted">Fragment pairs that should have been grouped together</p>
                                    <ul>
                                        <template x-for="(pair, i) in activePage.metrics.false_negatives" :key="'fn'+i">
                                            <li><code x-text="pair[0]"></code> &harr; <code x-text="pair[1]"></code></li>
                                        </template>
                                    </ul>
                                </div>
                            </template>
                        </div>
                    </details>
                </template>
            </section>
        </template>

        <p x-show="runs.length === 0" class="muted">No evaluation files found. Run evaluations first.</p>
    </main>

    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script>
        const EVAL_DATA = __DATA_PLACEHOLDER__;

        function dashboard() {
            return {
                runs: EVAL_DATA,
                sortKey: 'aggregate.mean_clustering_f1',
                sortDir: 'desc',
                expandedRun: null,
                expandedPage: null,

                get sortedRuns() {
                    return [...this.runs].sort((a, b) => {
                        const aVal = this.getNested(a, this.sortKey);
                        const bVal = this.getNested(b, this.sortKey);
                        if (typeof aVal === 'string') {
                            return this.sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                        }
                        const aNum = aVal ?? 0;
                        const bNum = bVal ?? 0;
                        return this.sortDir === 'asc' ? aNum - bNum : bNum - aNum;
                    });
                },

                get activeRun() {
                    return this.runs.find(r => r.run_id === this.expandedRun) || null;
                },

                get activePage() {
                    if (!this.activeRun) return null;
                    return this.activeRun.pages.find(p => p.page_id === this.expandedPage) || null;
                },

                getNested(obj, path) {
                    return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
                },

                sort(key) {
                    if (this.sortKey === key) {
                        this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
                    } else {
                        this.sortKey = key;
                        this.sortDir = 'desc';
                    }
                },

                toggleRun(runId) {
                    this.expandedRun = this.expandedRun === runId ? null : runId;
                    this.expandedPage = null;
                },

                togglePage(pageId) {
                    this.expandedPage = this.expandedPage === pageId ? null : pageId;
                },

                metricClass(val) {
                    if (val == null) return 'metric-na';
                    if (val >= 0.8) return 'metric-good';
                    if (val >= 0.6) return 'metric-ok';
                    return 'metric-bad';
                },

                fmt(val, decimals) {
                    if (val == null) return '-';
                    return val.toFixed(decimals || 4);
                },

                getChipClass(page, fragmentId) {
                    const idx = page.predicted_items.findIndex(item =>
                        item.fragment_ids.includes(fragmentId)
                    );
                    return idx !== -1 ? 'chip chip-' + (idx % 12) : 'chip chip-unmatched';
                },

                isPredMatch(page, item) {
                    const truth = page.ground_truth_items || [];
                    return truth.some(t =>
                        item.fragment_ids.every(fid => t.fragment_ids.includes(fid))
                    );
                },

                isTruthMatch(page, item) {
                    const pred = page.predicted_items || [];
                    return pred.some(p =>
                        item.fragment_ids.every(fid => p.fragment_ids.includes(fid))
                    );
                },

                truncate(str, len) {
                    if (!str) return '';
                    return str.length > len ? str.slice(0, len) + '\u2026' : str;
                },
            };
        }
    </script>
</body>
</html>"""


def load_evaluations(eval_dir: str = "data/2_evaluations") -> list[dict]:
    """Load all evaluation JSON files from the given directory."""
    runs = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def generate_html(runs: list[dict]) -> str:
    """Generate a self-contained HTML dashboard with embedded eval data."""
    embedded = json.dumps(runs, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", embedded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an evaluation dashboard HTML from eval JSON files."
    )
    parser.add_argument(
        "--eval-dir",
        default="data/2_evaluations",
        help="Directory containing evaluation JSON files (default: data/2_evaluations)",
    )
    parser.add_argument(
        "--output",
        default="data/2_evaluations/dashboard.html",
        help="Output HTML file path (default: data/2_evaluations/dashboard.html)",
    )
    args = parser.parse_args()

    runs = load_evaluations(args.eval_dir)
    if not runs:
        print(f"No evaluation files found in {args.eval_dir}")

    html = generate_html(runs)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written to {args.output} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
