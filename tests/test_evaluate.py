"""Tests for evaluate.py — ground truth parsing, F1 metrics, evaluation logging."""

import json
import os

import pytest

from src.newspaper_reconstructor.evaluate import (
    clustering_f1,
    evaluate_classification_page,
    evaluate_reconstruction_page,
    load_ground_truth_dir,
    log_evaluation_experiment,
    parse_article_xml,
)

# ─── parse_article_xml ───────────────────────────────────────────────────────

ARTICLE_XML_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<Articles pageId="test_page" modified="2026-01-01T00:00:00">
  <Article uuid="aaa-111" class="article">
    <Topics><Topic>politics</Topic></Topics>
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
  <Article uuid="ccc-333" class="letter">
    <Topics/>
    <Notes></Notes>
    <Regions>
      <Region ref="r_4" seq="1"/>
    </Regions>
  </Article>
  <Article uuid="ddd-444" class="caption">
    <Topics/>
    <Notes></Notes>
    <Regions>
      <Region ref="r_5" seq="1"/>
    </Regions>
  </Article>
</Articles>"""


class TestParseArticleXml:
    def test_parses_articles(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ARTICLE_XML_SAMPLE, encoding="utf-8")
        result = parse_article_xml(str(p))
        assert len(result) == 4

    def test_extracts_uuid_and_class(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ARTICLE_XML_SAMPLE, encoding="utf-8")
        result = parse_article_xml(str(p))
        first = next(r for r in result if r["uuid"] == "aaa-111")
        assert first["class"] == "article"
        assert first["fragment_ids"] == ["r_1", "r_2"]

    def test_extracts_topics(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ARTICLE_XML_SAMPLE, encoding="utf-8")
        result = parse_article_xml(str(p))
        first = next(r for r in result if r["uuid"] == "aaa-111")
        assert first["topics"] == ["politics"]

    def test_folds_letter_to_miscellaneous(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ARTICLE_XML_SAMPLE, encoding="utf-8")
        result = parse_article_xml(str(p))
        letter = next(r for r in result if r["uuid"] == "ccc-333")
        assert letter["class"] == "miscellaneous"

    def test_folds_caption_to_miscellaneous(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ARTICLE_XML_SAMPLE, encoding="utf-8")
        result = parse_article_xml(str(p))
        caption = next(r for r in result if r["uuid"] == "ddd-444")
        assert caption["class"] == "miscellaneous"


# ─── clustering_f1 ────────────────────────────────────────────────────────────


class TestClusteringF1:
    def test_perfect_match(self):
        pred = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c"]}]
        truth = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c"]}]
        result = clustering_f1(pred, truth)
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_completely_wrong(self):
        pred = [{"fragment_ids": ["a", "b", "c"]}]
        truth = [
            {"fragment_ids": ["a"]},
            {"fragment_ids": ["b"]},
            {"fragment_ids": ["c"]},
        ]
        result = clustering_f1(pred, truth)
        assert result["f1"] == 0.0

    def test_partial_match(self):
        # Truth: (a,b) (c,d)   Pred: (a,b) (c) (d)
        # True pairs: (a,b), (c,d) = 2
        # Pred pairs: (a,b) = 1
        # TP = 1 (pair a,b), FP = 0, FN = 1 (pair c,d)
        # Precision = 1/1 = 1.0, Recall = 1/2 = 0.5
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.6667
        pred = [
            {"fragment_ids": ["a", "b"]},
            {"fragment_ids": ["c"]},
            {"fragment_ids": ["d"]},
        ]
        truth = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c", "d"]}]
        result = clustering_f1(pred, truth)
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(2 / 3, abs=0.01)

    def test_single_fragment_items_dont_affect_score(self):
        pred = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c"]}]
        truth = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c"]}]
        result = clustering_f1(pred, truth)
        # Only one pair (a,b) in both → perfect
        assert result["f1"] == 1.0

    def test_all_single_fragments(self):
        pred = [{"fragment_ids": ["a"]}, {"fragment_ids": ["b"]}]
        truth = [{"fragment_ids": ["a"]}, {"fragment_ids": ["b"]}]
        result = clustering_f1(pred, truth)
        # No pairs to evaluate → defined as 1.0 (trivially correct)
        assert result["f1"] == 1.0

    def test_extra_cluster_in_pred(self):
        # Truth: (a,b,c)   Pred: (a,b) (c)
        # True pairs: (a,b), (a,c), (b,c) = 3
        # Pred pairs: (a,b) = 1
        # TP = 1, FP = 0, FN = 2
        # Precision = 1/1 = 1.0, Recall = 1/3
        # F1 = 2 * 1 * (1/3) / (1 + 1/3) = 0.5
        pred = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c"]}]
        truth = [{"fragment_ids": ["a", "b", "c"]}]
        result = clustering_f1(pred, truth)
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(1 / 3)
        assert result["f1"] == pytest.approx(0.5, abs=0.01)

    def test_returns_pair_counts(self):
        pred = [{"fragment_ids": ["a", "b", "c"]}]
        truth = [{"fragment_ids": ["a", "b"]}, {"fragment_ids": ["c"]}]
        result = clustering_f1(pred, truth)
        assert "tp" in result
        assert "fp" in result
        assert "fn" in result
        # Pred pairs: (a,b), (a,c), (b,c) = 3
        # Truth pairs: (a,b) = 1
        # TP = 1, FP = 2, FN = 0
        assert result["tp"] == 1
        assert result["fp"] == 2
        assert result["fn"] == 0


# ─── evaluate_reconstruction_page & evaluate_classification_page ────────────


class TestEvaluateReconstructionPage:
    def test_perfect_prediction(self):
        pred = [
            {"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"},
            {"fragment_ids": ["r_3"], "title": "B", "class": "advertisement"},
        ]
        truth = [
            {
                "uuid": "u1",
                "class": "article",
                "fragment_ids": ["r_1", "r_2"],
                "topics": [],
            },
            {
                "uuid": "u2",
                "class": "advertisement",
                "fragment_ids": ["r_3"],
                "topics": [],
            },
        ]
        result = evaluate_reconstruction_page(pred, truth)
        assert result["clustering_f1"] == 1.0
        assert result["ari"] == 1.0
        assert result["coverage"] == 1.0
        assert result["num_predicted_items"] == 2
        assert result["num_ground_truth_items"] == 2

    def test_coverage_unassigned_fragments(self):
        pred = [{"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"}]
        truth = [
            {
                "uuid": "u1",
                "class": "article",
                "fragment_ids": ["r_1", "r_2"],
                "topics": [],
            },
            {
                "uuid": "u2",
                "class": "advertisement",
                "fragment_ids": ["r_3"],
                "topics": [],
            },
        ]
        result = evaluate_reconstruction_page(pred, truth)
        # r_3 not assigned → coverage = 2/3
        assert result["coverage"] == pytest.approx(2 / 3)


class TestEvaluateClassificationPage:
    def test_class_accuracy_and_f1(self):
        pred = [
            {"id": "r_1", "predicted_class": "article"},
            {"id": "r_2", "predicted_class": "article"},
            {"id": "r_3", "predicted_class": "obituary"},
        ]
        truth = [
            {
                "uuid": "u1",
                "class": "article",
                "fragment_ids": ["r_1", "r_2"],
                "topics": [],
            },
            {
                "uuid": "u2",
                "class": "advertisement",
                "fragment_ids": ["r_3"],
                "topics": [],
            },
        ]
        result = evaluate_classification_page(pred, truth)
        # r_1: article == article (correct)
        # r_2: article == article (correct)
        # r_3: obituary != advertisement (wrong)
        # 2 correct out of 3 total
        assert result["num_fragments"] == 3


# ─── log_evaluation_experiment ─────────────────────────────────────────────────────────


class TestLogEvaluationRun:
    def test_logs_run_to_file(self, tmp_path):
        results = [
            {
                "page_id": "test_page",
                "metrics": {
                    "clustering_f1": 0.85,
                    "clustering_precision": 0.9,
                    "clustering_recall": 0.8,
                    "bcubed_f1": 0.88,
                    "bcubed_precision": 0.9,
                    "bcubed_recall": 0.87,
                    "ari": 0.89,
                    "coverage": 0.9,
                    "num_fragments": 10,
                    "num_predicted_items": 5,
                    "num_ground_truth_items": 5,
                },
                "predicted_items": [],
                "ground_truth_items": [],
            }
        ]
        config = {"provider": "openai", "model": "gpt-4o", "task": "reconstruction"}
        output_dir = str(tmp_path / "evaluations")
        path = log_evaluation_experiment(results, config, output_dir)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "experiment_id" in data
        assert "timestamp" in data
        assert data["config"]["model"] == "gpt-4o"
        assert len(data["pages"]) == 1
        assert "aggregate" in data

    def test_aggregate_metrics_reconstruction(self, tmp_path):
        results = [
            {
                "page_id": "p1",
                "metrics": {
                    "clustering_f1": 0.8,
                    "coverage": 0.9,
                    "clustering_precision": 0.8,
                    "clustering_recall": 0.8,
                    "bcubed_f1": 0.82,
                    "bcubed_precision": 0.83,
                    "bcubed_recall": 0.81,
                    "ari": 0.85,
                    "num_fragments": 10,
                    "num_predicted_items": 5,
                    "num_ground_truth_items": 5,
                },
                "predicted_items": [],
                "ground_truth_items": [],
            },
            {
                "page_id": "p2",
                "metrics": {
                    "clustering_f1": 0.6,
                    "coverage": 0.8,
                    "clustering_precision": 0.6,
                    "clustering_recall": 0.6,
                    "bcubed_f1": 0.62,
                    "bcubed_precision": 0.63,
                    "bcubed_recall": 0.61,
                    "ari": 0.65,
                    "num_fragments": 8,
                    "num_predicted_items": 4,
                    "num_ground_truth_items": 4,
                },
                "predicted_items": [],
                "ground_truth_items": [],
            },
        ]
        config = {"provider": "openai", "model": "gpt-4o", "task": "reconstruction"}
        output_dir = str(tmp_path / "evaluations")
        path = log_evaluation_experiment(results, config, output_dir)
        with open(path) as f:
            data = json.load(f)
        assert data["aggregate"]["mean_clustering_f1"] == pytest.approx(0.7)
        assert data["aggregate"]["mean_coverage"] == pytest.approx(0.85)
        assert data["aggregate"]["total_pages"] == 2

    def test_aggregate_metrics_classification(self, tmp_path):
        results = [
            {
                "page_id": "p1",
                "metrics": {
                    "weighted_precision": 0.8,
                    "weighted_recall": 0.8,
                    "weighted_f1": 0.8,
                    "num_fragments": 10,
                },
                "predicted_items": [],
                "ground_truth_items": [],
            },
            {
                "page_id": "p2",
                "metrics": {
                    "weighted_precision": 0.6,
                    "weighted_recall": 0.6,
                    "weighted_f1": 0.6,
                    "num_fragments": 10,
                },
                "predicted_items": [],
                "ground_truth_items": [],
            },
        ]
        config = {"provider": "openai", "model": "gpt-4o", "task": "classification"}
        output_dir = str(tmp_path / "evaluations")
        path = log_evaluation_experiment(results, config, output_dir)
        with open(path) as f:
            data = json.load(f)
        assert data["aggregate"]["mean_weighted_f1"] == pytest.approx(0.7)
        assert data["aggregate"]["total_pages"] == 2

    def test_includes_prompts_in_config(self, tmp_path):
        results = []
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "system_prompt": "You are...",
            "user_prompt_template": "Fragments: {fragments}",
        }
        output_dir = str(tmp_path / "evaluations")
        path = log_evaluation_experiment(results, config, output_dir)
        with open(path) as f:
            data = json.load(f)
        assert data["config"]["system_prompt"] == "You are..."
        assert data["config"]["user_prompt_template"] == "Fragments: {fragments}"

    def test_prompt_name_in_filename(self, tmp_path):
        results = []
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_name": "jawi_v2",
        }
        output_dir = str(tmp_path / "evaluations")
        path = log_evaluation_experiment(results, config, output_dir)
        assert "jawi_v2" in os.path.basename(path)
        with open(path) as f:
            data = json.load(f)
        assert data["experiment_id"].endswith("_jawi_v2")
        assert data["config"]["prompt_name"] == "jawi_v2"

    def test_prompt_name_defaults_to_default(self, tmp_path):
        results = []
        config = {"provider": "openai", "model": "gpt-4o"}
        output_dir = str(tmp_path / "evaluations")
        path = log_evaluation_experiment(results, config, output_dir)
        assert "_default" in os.path.basename(path)


# ─── load_ground_truth_dir ──────────────────────────────────────────────────────


class TestLoadGroundTruthDir:
    def test_loads_directory(self, tmp_path):
        d = tmp_path / "article_xml"
        d.mkdir()
        (d / "page1.xml").write_text(ARTICLE_XML_SAMPLE, encoding="utf-8")
        result = load_ground_truth_dir(str(d))
        assert "page1" in result
        assert len(result["page1"]) == 4
