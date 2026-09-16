import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DASHBOARD_PATH = ROOT / "dashboard.html"
BASELINE_REPORT = """# Baseline comparison — ds-filteredUM1956alto

Metrics are macro-averaged over 80 pages.

| Variant | pwF1 | pwP | pwR | B³ F1 | ARI |
| --- | --- | --- | --- | --- | --- |
| `baseline-singletons` | 0.000 | N/A¹ | 0.000 | 0.716 | 0.000 |
| `baseline-geometry` | 0.216 | 0.447 | 0.167 | 0.725 | 0.178 |
"""


def test_dashboard_parses_baseline_comparison_report():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the dashboard JavaScript")

    html = DASHBOARD_PATH.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.DOTALL)
    dashboard_script = next(
        script for script in scripts if "function dashboard()" in script
    )
    command = (
        dashboard_script
        + "\nprocess.stdout.write(JSON.stringify(parseBaselineComparison(process.argv[1])));"
    )

    result = subprocess.run(
        [node, "-e", command, BASELINE_REPORT],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(result.stdout)

    assert parsed["dataset"] == "ds-filteredUM1956alto"
    assert parsed["pages"] == 80
    assert len(parsed["rows"]) == 2
    assert parsed["rows"][0] == {
        "variant": "baseline-singletons",
        "f1": 0,
        "precision": None,
        "recall": 0,
        "bcubedF1": 0.716,
        "ari": 0,
    }


def test_dashboard_loads_baseline_report_from_server_and_folder_picker():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'fetch("reports/baselines/comparison.md")' in html
    assert "const baselineFile = files.find(" in html
    assert '"reports/baselines",' in html
    assert 'x-for="baseline in baselineReport.rows"' in html


def test_dashboard_places_baselines_after_llm_reconstructions():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    reconstruction_index = html.index('<section id="reconstruction-section">')
    llm_index = html.index("LLM reconstructions", reconstruction_index)
    llm_detail_index = html.index(
        "<!-- Reconstruction detail (inline) -->", reconstruction_index
    )
    baseline_index = html.index("Baseline methods", reconstruction_index)

    assert reconstruction_index < llm_index < llm_detail_index < baseline_index
    assert "Latest baseline" not in html


def test_dashboard_excludes_baselines_from_llm_reconstructions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the dashboard JavaScript")

    html = DASHBOARD_PATH.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.DOTALL)
    dashboard_script = next(
        script for script in scripts if "function dashboard()" in script
    )
    command = (
        dashboard_script
        + """
const state = dashboard();
state.experiments = [
    {experiment_id: "llm", config: {task: "reconstruction", provider: "openai"}, aggregate: {total_pages: 1}},
    {experiment_id: "baseline", config: {task: "reconstruction", provider: "baseline"}, aggregate: {total_pages: 1}},
    {experiment_id: "classification", config: {task: "classification", provider: "openai"}, aggregate: {total_pages: 1}},
];
process.stdout.write(JSON.stringify(state.filteredReconstructionExperiments.map((run) => run.experiment_id)));
"""
    )

    result = subprocess.run(
        [node, "-e", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["llm"]
