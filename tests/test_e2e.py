"""End-to-end tests exercising the full main.py pipeline with a mocked LLM.

Mock is injected at main.make_client so everything below (ALTO parsing,
prompt building, JSON parsing, evaluation, logging) runs through real code.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from main import main

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


# ─── --json-only ────────────────────────────────────────────────────────────


class TestE2EJsonOnly:
    def test_json_only_single_file(self, tmp_path, capsys):
        p = tmp_path / "test_page.xml"
        p.write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        rc = main(
            [
                "--alto",
                str(p),
                "--json-only",
                "--interim-dir",
                str(tmp_path / "interim"),
                "--model",
                "test-model",
                "--prompt-file",
                prompt_file,
            ]
        )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert len(out) == 3
        ids = {f["id"] for f in out}
        assert ids == {"r_1", "r_2", "r_3"}

    def test_json_only_directory(self, tmp_path, capsys):
        d = tmp_path / "pages"
        d.mkdir()
        (d / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        rc = main(
            [
                "--input-dir",
                str(d),
                "--json-only",
                "--interim-dir",
                str(tmp_path / "interim"),
                "--model",
                "test-model",
                "--prompt-file",
                prompt_file,
            ]
        )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert "page1" in out
        assert len(out["page1"]) == 3

    def test_json_only_directory_output_dir(self, tmp_path):
        d = tmp_path / "pages"
        d.mkdir()
        (d / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        out_dir = tmp_path / "out"
        prompt_file = _make_prompt_file(tmp_path)

        rc = main(
            [
                "--input-dir",
                str(d),
                "--json-only",
                "--output-dir",
                str(out_dir),
                "--interim-dir",
                str(tmp_path / "interim"),
                "--model",
                "test-model",
                "--prompt-file",
                prompt_file,
            ]
        )
        assert rc == 0

        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 1
        assert files[0].name == "page1.json"
        data = json.loads(files[0].read_text())
        assert len(data) == 3


# ─── Reconstruction (mocked LLM) ─────────────────────────────────────────────


class TestE2EReconstruct:
    def test_reconstruct_single_page(self, tmp_path, capsys):
        p = tmp_path / "test_page.xml"
        p.write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--alto",
                    str(p),
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert out["page_id"] == "test_page"
        assert len(out["items"]) == 2
        assert out["items"][0]["fragment_ids"] == ["r_1", "r_2"]
        assert out["items"][1]["class"] == "advertisement"

    def test_reconstruct_to_file(self, tmp_path):
        p = tmp_path / "test_page.xml"
        p.write_text(ALTO_XML, encoding="utf-8")
        out_file = tmp_path / "result.json"
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--alto",
                    str(p),
                    "--output",
                    str(out_file),
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        with open(out_file) as f:
            data = json.load(f)
        assert data["page_id"] == "test_page"
        assert len(data["items"]) == 2

    def test_reconstruct_dir_to_output_dir(self, tmp_path):
        d = tmp_path / "pages"
        d.mkdir()
        (d / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        (d / "page2.xml").write_text(ALTO_XML, encoding="utf-8")
        out_dir = tmp_path / "out"
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--input-dir",
                    str(d),
                    "--output-dir",
                    str(out_dir),
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 2
        assert files[0].name == "page1.json"
        assert files[1].name == "page2.json"
        for f in files:
            data = json.loads(f.read_text())
            assert len(data) == 2


# ─── Evaluate single page ────────────────────────────────────────────────────


class TestE2EEvaluateSingle:
    def test_evaluate_perfect_match(self, tmp_path, capsys):
        alto = tmp_path / "test_page.xml"
        article = tmp_path / "articles.xml"
        alto.write_text(ALTO_XML, encoding="utf-8")
        article.write_text(ARTICLE_XML, encoding="utf-8")
        eval_dir = str(tmp_path / "evaluations")
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--alto",
                    str(alto),
                    "--article-xml",
                    str(article),
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert out["page_id"] == "test_page"
        assert out["metrics"]["clustering_f1"] == 1.0
        assert out["metrics"]["class_accuracy"] == 1.0
        assert out["metrics"]["coverage"] == 1.0

        eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
        assert len(eval_files) == 1

    def test_evaluate_wrong_clustering(self, tmp_path, capsys):
        alto = tmp_path / "test_page.xml"
        article = tmp_path / "articles.xml"
        alto.write_text(ALTO_XML, encoding="utf-8")
        article.write_text(ARTICLE_XML, encoding="utf-8")
        eval_dir = str(tmp_path / "evaluations")
        prompt_file = _make_prompt_file(tmp_path)

        wrong_response = json.dumps(
            [
                {"fragment_ids": ["r_1"], "title": "A", "class": "article"},
                {"fragment_ids": ["r_2", "r_3"], "title": "B", "class": "article"},
            ]
        )
        client = MagicMock()
        client.complete.return_value = wrong_response

        with patch("main.make_client", return_value=client):
            rc = main(
                [
                    "--alto",
                    str(alto),
                    "--article-xml",
                    str(article),
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        # Truth pairs: (r_1, r_2). Pred pairs: (r_2, r_3).
        # TP=0, FP=1, FN=1 -> F1=0
        assert out["metrics"]["clustering_f1"] == 0.0

    def test_evaluate_with_prompt_file(self, tmp_path, capsys):
        alto = tmp_path / "test_page.xml"
        article = tmp_path / "articles.xml"
        alto.write_text(ALTO_XML, encoding="utf-8")
        article.write_text(ARTICLE_XML, encoding="utf-8")
        eval_dir = str(tmp_path / "evaluations")

        prompt_file = tmp_path / "jawi_v2.txt"
        prompt_file.write_text("Custom system prompt", encoding="utf-8")

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--alto",
                    str(alto),
                    "--article-xml",
                    str(article),
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--prompt-file",
                    str(prompt_file),
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                ]
            )
        assert rc == 0

        eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
        assert len(eval_files) == 1
        assert "jawi_v2" in eval_files[0]

    def test_evaluate_with_json_prompt_file(self, tmp_path, capsys):
        alto = tmp_path / "test_page.xml"
        article = tmp_path / "articles.xml"
        alto.write_text(ALTO_XML, encoding="utf-8")
        article.write_text(ARTICLE_XML, encoding="utf-8")
        eval_dir = str(tmp_path / "evaluations")

        prompt_data = {
            "system_prompt": "You are an expert in historical Malay.",
            "user_prompt_template": "Fragments:\n\n{fragments}\n\nReturn ONLY a JSON array.",
        }
        prompt_file = tmp_path / "system_v01.json"
        prompt_file.write_text(json.dumps(prompt_data), encoding="utf-8")

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--alto",
                    str(alto),
                    "--article-xml",
                    str(article),
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--prompt-file",
                    str(prompt_file),
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                ]
            )
        assert rc == 0

        eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
        assert len(eval_files) == 1
        assert "system_v01" in eval_files[0]

        with open(os.path.join(eval_dir, eval_files[0])) as f:
            log = json.load(f)
        assert (
            log["config"]["system_prompt"] == "You are an expert in historical Malay."
        )
        assert (
            log["config"]["user_prompt_template"]
            == "Fragments:\n\n{fragments}\n\nReturn ONLY a JSON array."
        )


class TestE2EEvaluateInputDir:
    def test_evaluate_input_dir(self, tmp_path, capsys):
        alto_dir = tmp_path / "alto"
        article_dir = tmp_path / "article_xml"
        alto_dir.mkdir()
        article_dir.mkdir()

        (alto_dir / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        (article_dir / "page1.xml").write_text(ARTICLE_XML, encoding="utf-8")
        eval_dir = str(tmp_path / "evaluations")
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--ground-truth-dir",
                    str(article_dir),
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert len(out) == 1
        assert out[0]["page_id"] == "page1"
        assert out[0]["metrics"]["clustering_f1"] == 1.0
        assert out[0]["metrics"]["coverage"] == 1.0

        eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
        assert len(eval_files) == 1
        with open(os.path.join(eval_dir, eval_files[0])) as f:
            log = json.load(f)
        assert log["aggregate"]["mean_clustering_f1"] == 1.0
        assert log["aggregate"]["total_pages"] == 1

    def test_sample_size_processes_subset(self, tmp_path, capsys):
        alto_dir = tmp_path / "alto"
        alto_dir.mkdir()
        for i in range(4):
            (alto_dir / f"page{i}.xml").write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            rc = main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--sample-size",
                    "2",
                    "--seed",
                    "42",
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert len(out) == 2

    def test_sample_size_with_seed_reproducible(self, tmp_path, capsys):
        alto_dir = tmp_path / "alto"
        alto_dir.mkdir()
        for i in range(4):
            (alto_dir / f"page{i}.xml").write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--sample-size",
                    "2",
                    "--seed",
                    "42",
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
            out1 = json.loads(capsys.readouterr().out)

        with patch("main.make_client", return_value=_mock_client()):
            main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--sample-size",
                    "2",
                    "--seed",
                    "42",
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
            out2 = json.loads(capsys.readouterr().out)

        assert sorted(out1.keys()) == sorted(out2.keys())

    def test_default_seed_reproducible_without_flag(self, tmp_path, capsys):
        alto_dir = tmp_path / "alto"
        alto_dir.mkdir()
        for i in range(4):
            (alto_dir / f"page{i}.xml").write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        with patch("main.make_client", return_value=_mock_client()):
            main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--sample-size",
                    "2",
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
            out1 = json.loads(capsys.readouterr().out)

        with patch("main.make_client", return_value=_mock_client()):
            main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--sample-size",
                    "2",
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
            out2 = json.loads(capsys.readouterr().out)

        assert sorted(out1.keys()) == sorted(out2.keys())

    def test_failed_page_skipped_in_batch(self, tmp_path, capsys):
        from openai import APIError

        alto_dir = tmp_path / "alto"
        alto_dir.mkdir()
        (alto_dir / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        (alto_dir / "page2.xml").write_text(ALTO_XML, encoding="utf-8")
        prompt_file = _make_prompt_file(tmp_path)

        client = MagicMock()
        client.complete.side_effect = [
            MOCK_LLM_RESPONSE,
            APIError(message="504", request=None, body=None),
            APIError(message="504", request=None, body=None),
            APIError(message="504", request=None, body=None),
        ]

        with patch("main.make_client", return_value=client):
            rc = main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert "page1" in out
        assert "page2" not in out

        assert "[page2] SKIPPED (reconstruction failed)" in captured.err

    def test_eval_config_includes_pages_processed(self, tmp_path, capsys):
        from openai import APIError

        alto_dir = tmp_path / "alto"
        alto_dir.mkdir()
        (alto_dir / "page1.xml").write_text(ALTO_XML, encoding="utf-8")
        (alto_dir / "page2.xml").write_text(ALTO_XML, encoding="utf-8")
        article_dir = tmp_path / "articles"
        article_dir.mkdir()
        ground_truth_xml = ARTICLE_XML.replace("page1", "page1")
        (article_dir / "page1.xml").write_text(ground_truth_xml, encoding="utf-8")
        eval_dir = str(tmp_path / "evaluations")
        prompt_file = _make_prompt_file(tmp_path)

        client = MagicMock()
        client.complete.side_effect = [
            MOCK_LLM_RESPONSE,
            APIError(message="504", request=None, body=None),
            APIError(message="504", request=None, body=None),
            APIError(message="504", request=None, body=None),
        ]

        with patch("main.make_client", return_value=client):
            rc = main(
                [
                    "--input-dir",
                    str(alto_dir),
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--prompt-file",
                    prompt_file,
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--ground-truth-dir",
                    str(article_dir),
                ]
            )
        assert rc == 0

        captured = capsys.readouterr()
        eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
        assert len(eval_files) == 1
        with open(os.path.join(eval_dir, eval_files[0])) as f:
            log = json.load(f)
        assert log["config"]["pages_processed"] == 1
        assert log["config"]["pages_failed"] == 1
        assert log["aggregate"]["total_pages"] == 1
        assert "SKIPPED" in captured.err


# ─── Real data smoke tests ───────────────────────────────────────────────────

REAL_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/0_external",
)


@pytest.mark.skipif(
    not os.path.isdir(REAL_DATA), reason="data/0_external not available"
)
class TestE2ERealData:
    def test_json_only_real_alto(self, capsys, tmp_path):
        alto = os.path.join(REAL_DATA, "alto", "UM-1956-01-09-6.xml")
        prompt_file = _make_prompt_file(tmp_path)
        rc = main(
            [
                "--alto",
                alto,
                "--json-only",
                "--interim-dir",
                str(tmp_path / "interim"),
                "--model",
                "test-model",
                "--prompt-file",
                prompt_file,
            ]
        )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert len(out) > 0
        assert all("id" in f and "text" in f for f in out)

    def test_evaluate_real_page_mocked_llm(self, tmp_path, capsys):
        from src.newspaper_reconstructor.evaluate import parse_article_xml

        alto = os.path.join(REAL_DATA, "alto", "UM-1956-01-09-6.xml")
        article = os.path.join(REAL_DATA, "article_xml", "UM-1956-01-09-6.xml")
        eval_dir = str(tmp_path / "evaluations")
        prompt_file = _make_prompt_file(tmp_path)

        truth = parse_article_xml(article)
        mock_response = json.dumps(
            [
                {
                    "fragment_ids": t["fragment_ids"],
                    "title": "mock",
                    "class": t["class"],
                }
                for t in truth
            ]
        )
        client = MagicMock()
        client.complete.return_value = mock_response

        with patch("main.make_client", return_value=client):
            rc = main(
                [
                    "--alto",
                    alto,
                    "--article-xml",
                    article,
                    "--evaluate",
                    "--eval-dir",
                    eval_dir,
                    "--interim-dir",
                    str(tmp_path / "interim"),
                    "--model",
                    "test-model",
                    "--prompt-file",
                    prompt_file,
                ]
            )
        assert rc == 0

        out = json.loads(capsys.readouterr().out)
        assert out["page_id"] == "UM-1956-01-09-6"
        assert out["metrics"]["clustering_f1"] == 1.0
        assert out["metrics"]["coverage"] == pytest.approx(1.0, abs=0.05)
