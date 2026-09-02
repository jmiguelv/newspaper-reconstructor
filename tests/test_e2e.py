"""End-to-end tests exercising the full Typer pipeline with a mocked LLM.

Mock is injected at main.make_client so everything below runs through real code.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from main import app

runner = CliRunner()

DEFAULT_PROMPT_MD = """# System Prompt

You are given text fragments from a newspaper.

# User Prompt Template

Fragments:

{fragments}

Return ONLY a JSON array.
"""


def _make_prompt_file(tmp_path) -> str:
    """Create a minimal prompt file in tmp_path and return its path."""
    p = tmp_path / "prompt.md"
    p.write_text(DEFAULT_PROMPT_MD, encoding="utf-8")
    return str(p)


ALTO_XML = """<?xml version='1.0' encoding='utf-8'?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="test_page" WIDTH="1000" HEIGHT="2000">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="2000">
        <TextBlock ID="r_1" HPOS="20" VPOS="30" WIDTH="200" HEIGHT="400" TYPE="text">
          <TextLine ID="l1" HPOS="20" VPOS="30" WIDTH="200" HEIGHT="50">
            <String ID="s1" CONTENT="Article text part 1" HPOS="20" VPOS="30" WIDTH="100" HEIGHT="50" />
          </TextLine>
        </TextBlock>
        <TextBlock ID="r_2" HPOS="300" VPOS="30" WIDTH="200" HEIGHT="400" TYPE="text">
          <TextLine ID="l2" HPOS="300" VPOS="30" WIDTH="200" HEIGHT="50">
            <String ID="s2" CONTENT="Article text part 2" HPOS="300" VPOS="30" WIDTH="100" HEIGHT="50" />
          </TextLine>
        </TextBlock>
        <TextBlock ID="r_3" HPOS="20" VPOS="500" WIDTH="200" HEIGHT="100" TYPE="text">
          <TextLine ID="l3" HPOS="20" VPOS="500" WIDTH="200" HEIGHT="50">
            <String ID="s3" CONTENT="Advertisement text" HPOS="20" VPOS="500" WIDTH="100" HEIGHT="50" />
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""

ARTICLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Articles pageId="test_page" modified="2026-01-01T00:00:00">
  <Article uuid="aaa-111" class="article">
    <Topics><Topic>news</Topic></Topics>
    <Notes></Notes>
    <Regions>
      <Region ref="r_1" seq="1"/>
      <Region ref="r_2" seq="2"/>
    </Regions>
  </Article>
  <Article uuid="bbb-222" class="advertisement">
    <Topics/>
    <Notes></Notes>
    <Regions>
      <Region ref="r_3" seq="1"/>
    </Regions>
  </Article>
</Articles>"""

MOCK_LLM_RESPONSE = json.dumps(
    [
        {"fragment_ids": ["r_1", "r_2"], "title": "News article", "class": "article"},
        {"fragment_ids": ["r_3"], "title": "Ad", "class": "advertisement"},
    ]
)


def _mock_client():
    client = MagicMock()
    client.complete.return_value = MOCK_LLM_RESPONSE
    return client


# ─── Parse ────────────────────────────────────────────────────────────


class TestE2EParse:
    def test_parse_directory(self, tmp_path):
        d = tmp_path / "alto"
        d.mkdir()
        (d / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        out_dir = tmp_path / "fragments"

        result = runner.invoke(
            app,
            [
                "parse",
                "-i",
                str(d),
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0

        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 1
        assert files[0].name == "page1.json"
        data = json.loads(files[0].read_text())
        assert len(data) == 3


# ─── ETL (JSON articles) ──────────────────────────────────────────────


ARTICLE_JSON = {
    "r_1": "Article text part 1",
    "r_2": "Article text part 2",
    "r_3": "Advertisement text",
}


class TestE2EEtl:
    def test_etl_directory(self, tmp_path):
        d = tmp_path / "articles"
        d.mkdir()
        (d / "page1.json").write_text(json.dumps(ARTICLE_JSON), encoding="utf-8")
        out_dir = tmp_path / "fragments"

        result = runner.invoke(
            app,
            [
                "etl",
                "-i",
                str(d),
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0

        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 1
        assert files[0].name == "page1.json"
        data = json.loads(files[0].read_text())
        assert len(data) == 3
        assert all("id" in f and "text" in f for f in data)

    def test_etl_page_filter(self, tmp_path):
        d = tmp_path / "articles"
        d.mkdir()
        (d / "page1.json").write_text(json.dumps(ARTICLE_JSON), encoding="utf-8")
        (d / "page2.json").write_text(json.dumps({"r_4": "Other"}), encoding="utf-8")
        out_dir = tmp_path / "fragments"

        result = runner.invoke(
            app,
            ["etl", "-i", str(d), "-o", str(out_dir), "--page-id", "page1"],
        )
        assert result.exit_code == 0

        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 1
        assert files[0].name == "page1.json"

    def test_etl_then_cluster(self, tmp_path):
        d = tmp_path / "articles"
        d.mkdir()
        (d / "page1.json").write_text(json.dumps(ARTICLE_JSON), encoding="utf-8")
        frag_dir = tmp_path / "fragments"
        out_dir = tmp_path / "reconstructions"
        prompt_file = _make_prompt_file(tmp_path)

        result = runner.invoke(app, ["etl", "-i", str(d), "-o", str(frag_dir)])
        assert result.exit_code == 0

        with patch("main.make_client", return_value=_mock_client()):
            result = runner.invoke(
                app,
                [
                    "cluster",
                    "-i",
                    str(frag_dir),
                    "-o",
                    str(out_dir),
                    "--model",
                    "test-model",
                    "-p",
                    prompt_file,
                ],
            )
        assert result.exit_code == 0
        files = [
            f for f in sorted(out_dir.glob("*.json")) if f.name != "_metadata.json"
        ]
        assert len(files) == 1


# ─── Cluster (mocked LLM) ─────────────────────────────────────────────


class TestE2ECluster:
    def test_cluster_dir(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        data = [{"id": "r_1", "text": "hello"}]
        (d / "page1.json").write_text(json.dumps(data), encoding="utf-8")
        (d / "page2.json").write_text(json.dumps(data), encoding="utf-8")
        out_dir = tmp_path / "reconstructions"
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            result = runner.invoke(
                app,
                [
                    "cluster",
                    "-i",
                    str(d),
                    "-o",
                    str(out_dir),
                    "--model",
                    "test-model",
                    "-p",
                    prompt_file,
                ],
            )
        assert result.exit_code == 0

        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 3  # 2 json files + _metadata.json
        with open(out_dir / "page1.json") as f:
            data1 = json.load(f)
            assert len(data1) == 1

    def test_cluster_parallel(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        data = [{"id": "r_1", "text": "hello"}]
        for i in range(4):
            (d / f"page{i}.json").write_text(json.dumps(data), encoding="utf-8")
        out_dir = tmp_path / "reconstructions"
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            result = runner.invoke(
                app,
                [
                    "cluster",
                    "-i",
                    str(d),
                    "-o",
                    str(out_dir),
                    "--model",
                    "test-model",
                    "-p",
                    prompt_file,
                    "--max-workers",
                    "4",
                ],
            )
        assert result.exit_code == 0

        output_files = [
            f for f in sorted(out_dir.glob("*.json")) if f.name != "_metadata.json"
        ]
        assert len(output_files) == 4
        for f in output_files:
            assert len(json.loads(f.read_text())) == 1

    def test_classify_parallel(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        data = [{"id": "r_1", "text": "hello"}]
        for i in range(4):
            (d / f"page{i}.json").write_text(json.dumps(data), encoding="utf-8")
        out_dir = tmp_path / "classified"
        prompt_file = _make_prompt_file(tmp_path)

        mock_classify_response = json.dumps({"r_1": "article"})
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_classify_response

        with patch("main.make_client", return_value=mock_client):
            result = runner.invoke(
                app,
                [
                    "classify",
                    "-i",
                    str(d),
                    "-o",
                    str(out_dir),
                    "--model",
                    "test-model",
                    "-p",
                    prompt_file,
                    "--max-workers",
                    "4",
                ],
            )
        assert result.exit_code == 0

        output_files = [
            f for f in sorted(out_dir.glob("*.json")) if f.name != "_metadata.json"
        ]
        assert len(output_files) == 4
        for f in output_files:
            frags = json.loads(f.read_text())
            assert frags[0]["predicted_class"] == "article"

    def test_classify_unknown_provider_clean_error(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        (d / "page1.json").write_text(
            json.dumps([{"id": "r_1", "text": "hello"}]), encoding="utf-8"
        )
        prompt_file = _make_prompt_file(tmp_path)

        result = runner.invoke(
            app,
            [
                "classify",
                "-i",
                str(d),
                "-o",
                str(tmp_path / "classified"),
                "--model",
                "test-model",
                "-p",
                prompt_file,
                "--provider",
                "no-such-provider",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown provider" in result.output
        assert not isinstance(result.exception, ValueError)


class TestE2EClusterUnknownProvider:
    def test_cluster_unknown_provider_clean_error(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        (d / "page1.json").write_text(
            json.dumps([{"id": "r_1", "text": "hello"}]), encoding="utf-8"
        )
        prompt_file = _make_prompt_file(tmp_path)

        result = runner.invoke(
            app,
            [
                "cluster",
                "-i",
                str(d),
                "-o",
                str(tmp_path / "reconstructions"),
                "--model",
                "test-model",
                "-p",
                prompt_file,
                "--provider",
                "no-such-provider",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown provider" in result.output
        assert not isinstance(result.exception, ValueError)


# ─── Evaluate single page ────────────────────────────────────────────────────


class TestE2EEvaluate:
    def test_evaluate_perfect_match(self, tmp_path):
        recon_dir = tmp_path / "reconstructions"
        recon_dir.mkdir()
        article_dir = tmp_path / "article_xml"
        article_dir.mkdir()

        recon_data = json.loads(MOCK_LLM_RESPONSE)
        (recon_dir / "test_page.json").write_text(
            json.dumps(recon_data), encoding="utf-8"
        )
        (article_dir / "test_page.xml").write_text(ARTICLE_XML, encoding="utf-8")

        eval_dir = str(tmp_path / "evaluations")

        result = runner.invoke(
            app,
            [
                "evaluate",
                "-i",
                str(recon_dir),
                "-g",
                str(article_dir),
                "--eval-dir",
                eval_dir,
                "--task",
                "reconstruction",
            ],
        )
        assert result.exit_code == 0

        eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
        assert len(eval_files) == 1
        with open(os.path.join(eval_dir, eval_files[0])) as f:
            log = json.load(f)
            assert log["pages"][0]["page_id"] == "test_page"
            assert log["aggregate"]["mean_clustering_f1"] == 1.0


# ─── Plan ────────────────────────────────────────────────────────────────────


class TestE2EPlan:
    def test_plan_reports_stats(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        (d / "page1.json").write_text(
            json.dumps(
                [
                    {"id": "r_1", "text": "hello world"},
                    {"id": "r_2", "text": "foo"},
                ]
            ),
            encoding="utf-8",
        )

        result = runner.invoke(app, ["plan", "-i", str(d)])
        assert result.exit_code == 0
        assert "Analyzed 1 pages" in result.output
        assert "Average fragments per page: 2.0" in result.output

    def test_plan_skips_invalid_json(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        (d / "good.json").write_text(
            json.dumps([{"id": "r_1", "text": "hello"}]), encoding="utf-8"
        )
        (d / "bad.json").write_text("{not json", encoding="utf-8")

        result = runner.invoke(app, ["plan", "-i", str(d)])
        assert result.exit_code == 0
        assert "Analyzed 1 pages" in result.output
        assert "Warning: Failed to process bad.json" in result.output

    def test_plan_missing_folder(self, tmp_path):
        result = runner.invoke(app, ["plan", "-i", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "does not exist or is not a directory" in result.output

    def test_plan_empty_folder(self, tmp_path):
        d = tmp_path / "fragments"
        d.mkdir()
        result = runner.invoke(app, ["plan", "-i", str(d)])
        assert result.exit_code == 1


# ─── Suggest (mocked LLM judge) ──────────────────────────────────────────────


class TestE2ESuggest:
    def _write_eval_log(self, tmp_path):
        """Create an eval log plus the fragments dir it points at."""
        eval_dir = tmp_path / "reports" / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        fragments_dir = tmp_path / "data" / "1_interim" / "ds" / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)
        (fragments_dir / "page1.json").write_text(
            json.dumps([{"id": "r_1", "text": "hello"}]), encoding="utf-8"
        )

        log = {
            "config": {
                "input_folder": "data/1_interim/ds/reconstructions/exp1",
                "system_prompt": "sys prompt",
                "user_prompt_template": "user prompt {fragments}",
            },
            "pages": [
                {
                    "page_id": "page1",
                    "metrics": {"clustering_f1": 0.5},
                    "predicted_items": [{"fragment_ids": ["r_1"], "class": "article"}],
                    "ground_truth_items": [
                        {
                            "fragment_ids": ["r_1"],
                            "class": "article",
                            "uuid": "u1",
                        }
                    ],
                }
            ],
        }
        (eval_dir / "exp1.json").write_text(json.dumps(log), encoding="utf-8")

    def test_suggest_writes_prompt_and_suggestions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_eval_log(tmp_path)

        mock_client = MagicMock()
        mock_client.complete.return_value = "## Suggestions\nUse better prompts."

        with patch("main.make_client", return_value=mock_client):
            result = runner.invoke(
                app,
                ["suggest", "--experiment-id", "exp1", "--model", "test-model"],
            )

        assert result.exit_code == 0
        out_dir = tmp_path / "reports" / "suggestions"
        suggestions = (out_dir / "exp1_suggestions.md").read_text()
        assert "## Suggestions" in suggestions
        assert "Use better prompts." in suggestions

        judge_prompt = (out_dir / "exp1_prompt.md").read_text()
        assert "sys prompt" in judge_prompt
        assert "hello" in judge_prompt  # fragment text included

    def test_suggest_missing_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "reports" / "evaluations").mkdir(parents=True)

        result = runner.invoke(
            app,
            ["suggest", "--experiment-id", "missing", "--model", "test-model"],
        )
        assert result.exit_code == 1
        assert "Evaluation log not found" in result.output

    def test_suggest_unknown_provider_clean_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "suggest",
                "--experiment-id",
                "exp1",
                "--model",
                "test-model",
                "--provider",
                "no-such-provider",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown provider" in result.output
        assert not isinstance(result.exception, ValueError)


# ─── Real data smoke tests ───────────────────────────────────────────────────

REAL_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/0_external",
)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(REAL_DATA, "alto")),
    reason="data/0_external/alto not available",
)
class TestE2ERealData:
    def test_parse_real_alto(self, tmp_path):
        alto = os.path.join(REAL_DATA, "alto")
        out_dir = tmp_path / "fragments"

        result = runner.invoke(
            app,
            ["parse", "-i", alto, "-o", str(out_dir), "--page-id", "UM-1956-01-09-6"],
        )
        assert result.exit_code == 0
        assert (out_dir / "UM-1956-01-09-6.json").exists()


NEW_DATASET = os.path.join(REAL_DATA, "ds-articlereconstruction-20260821")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(NEW_DATASET, "articles")),
    reason="ds-articlereconstruction-20260821 dataset not available",
)
class TestE2ERealDataNewFormat:
    def test_etl_real_json(self, tmp_path):
        articles = os.path.join(NEW_DATASET, "articles")
        out_dir = tmp_path / "fragments"

        result = runner.invoke(
            app,
            ["etl", "-i", articles, "-o", str(out_dir), "--page-id", "UM-1956-01-09-6"],
        )
        assert result.exit_code == 0
        assert (out_dir / "UM-1956-01-09-6.json").exists()

        data = json.loads((out_dir / "UM-1956-01-09-6.json").read_text())
        assert len(data) > 0
        assert all("id" in f and "text" in f for f in data)
