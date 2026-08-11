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


# ─── Evaluate single page ────────────────────────────────────────────────────


class TestE2EEvaluate:
    def test_evaluate_perfect_match(self, tmp_path):
        recon_dir = tmp_path / "reconstructions"
        recon_dir.mkdir()
        article_dir = tmp_path / "article_xml"
        article_dir.mkdir()

        recon_data = json.loads(MOCK_LLM_RESPONSE)
        # evaluation expects page dict in reconstructions
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
