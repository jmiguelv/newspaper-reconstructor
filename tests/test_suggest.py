import json

from src.newspaper_reconstructor.suggest import (
    _derive_fragments_dir,
    build_judge_prompt,
    generate_suggestions,
    identify_worst_pages,
)


def test_identify_worst_pages():
    run_data = {
        "config": {"model": "test"},
        "pages": [
            {"page_id": "p1", "metrics": {"clustering_f1": 0.8}},
            {"page_id": "p2", "metrics": {"clustering_f1": 0.5}},
            {"page_id": "p3", "metrics": {"clustering_f1": 0.9}},
            {"page_id": "p4", "metrics": {"clustering_f1": 0.6}},
        ],
    }
    worst = identify_worst_pages(run_data, top_k=2)
    assert len(worst) == 2
    assert worst[0]["page_id"] == "p2"
    assert worst[1]["page_id"] == "p4"


def test_identify_worst_pages_skips_pages_without_metrics():
    run_data = {
        "pages": [
            {"page_id": "p1", "metrics": {"clustering_f1": 0.4}},
            {"page_id": "p2"},
        ]
    }
    worst = identify_worst_pages(run_data, top_k=3)
    assert [p["page_id"] for p in worst] == ["p1"]


def test_identify_worst_pages_returns_empty_when_no_valid_pages():
    assert identify_worst_pages({"pages": [{"page_id": "p1"}]}) == []
    assert identify_worst_pages({}) == []


def test_derive_fragments_dir_from_reconstruction_input_folder():
    run_data = {"config": {"input_folder": "data/1_interim/ds-x/reconstructions/exp1"}}
    assert _derive_fragments_dir(run_data) == "data/1_interim/ds-x/fragments"


def test_derive_fragments_dir_from_classification_input_folder():
    run_data = {"config": {"input_folder": "data/1_interim/ds-x/classified/exp1"}}
    assert _derive_fragments_dir(run_data) == "data/1_interim/ds-x/fragments"


def test_derive_fragments_dir_legacy_layout_without_dataset_subdir():
    run_data = {"config": {"input_folder": "data/1_interim/reconstructions/exp1"}}
    assert _derive_fragments_dir(run_data) == "data/1_interim/fragments"


def test_derive_fragments_dir_returns_none_without_input_folder():
    assert _derive_fragments_dir({"config": {}}) is None
    assert _derive_fragments_dir({}) is None


def _write_page_fragments(fragments_dir, page_id, fragments):
    fragments_dir.mkdir(parents=True, exist_ok=True)
    (fragments_dir / f"{page_id}.json").write_text(
        json.dumps(fragments), encoding="utf-8"
    )


def test_build_judge_prompt_includes_fragment_text(tmp_path):
    fragments_dir = tmp_path / "fragments"
    _write_page_fragments(
        fragments_dir,
        "p2",
        [{"id": "r_1", "text": "Hello"}, {"id": "r_2", "text": "World"}],
    )

    run_data = {
        "config": {"system_prompt": "Test System", "user_prompt_template": "Test User"}
    }
    worst_pages = [
        {
            "page_id": "p2",
            "metrics": {"clustering_f1": 0.5},
            "ground_truth_items": [
                {"class": "article", "fragment_ids": ["r_1", "r_2"]}
            ],
            "predicted_items": [
                {"class": "article", "fragment_ids": ["r_1"]},
                {"class": "article", "fragment_ids": ["r_2"]},
            ],
        }
    ]

    prompt = build_judge_prompt(run_data, worst_pages, fragments_dir=str(fragments_dir))

    assert "Test System" in prompt
    assert "Test User" in prompt
    assert "Page: p2" in prompt
    assert "r_1: 'Hello" in prompt
    assert "r_2: 'World" in prompt


def test_build_judge_prompt_warns_when_fragment_cache_missing(tmp_path):
    run_data = {
        "config": {"system_prompt": "Test System", "user_prompt_template": "Test User"}
    }
    worst_pages = [
        {
            "page_id": "p_missing",
            "metrics": {"clustering_f1": 0.5},
            "ground_truth_items": [{"class": "article", "fragment_ids": ["r_1"]}],
            "predicted_items": [{"class": "article", "fragment_ids": ["r_1"]}],
        }
    ]

    prompt = build_judge_prompt(
        run_data, worst_pages, fragments_dir=str(tmp_path / "fragments")
    )

    assert "Fragment text cache missing for p_missing" in prompt


def test_generate_suggestions_loads_fragment_text_from_dataset_dir(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    dataset_dir = tmp_path / "data" / "1_interim" / "ds-test"
    _write_page_fragments(
        dataset_dir / "fragments",
        "p1",
        [{"id": "r_1", "text": "Hello"}, {"id": "r_2", "text": "World"}],
    )

    eval_dir = tmp_path / "reports" / "evaluations"
    eval_dir.mkdir(parents=True)
    log = {
        "experiment_id": "exp1",
        "config": {
            "input_folder": str(dataset_dir / "reconstructions" / "exp1"),
            "system_prompt": "S",
            "user_prompt_template": "U",
        },
        "pages": [
            {
                "page_id": "p1",
                "metrics": {"clustering_f1": 0.5},
                "ground_truth_items": [
                    {"class": "article", "fragment_ids": ["r_1", "r_2"]}
                ],
                "predicted_items": [
                    {"class": "article", "fragment_ids": ["r_1", "r_2"]}
                ],
            }
        ],
    }
    (eval_dir / "exp1.json").write_text(json.dumps(log), encoding="utf-8")

    class FakeClient:
        def complete(self, system, user):
            return "suggestions"

    exit_code = generate_suggestions(
        "exp1", str(eval_dir), FakeClient(), "test-model", "clustering"
    )

    assert exit_code == 0
    saved_prompt = (tmp_path / "reports" / "suggestions" / "exp1_prompt.md").read_text(
        encoding="utf-8"
    )
    assert "r_1: 'Hello" in saved_prompt
    assert "r_2: 'World" in saved_prompt
    saved_suggestions = (
        tmp_path / "reports" / "suggestions" / "exp1_suggestions.md"
    ).read_text(encoding="utf-8")
    assert "suggestions" in saved_suggestions


def test_generate_suggestions_fails_when_log_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    eval_dir = tmp_path / "reports" / "evaluations"
    eval_dir.mkdir(parents=True)

    class FakeClient:
        def complete(self, system, user):
            return "unused"

    exit_code = generate_suggestions(
        "nope", str(eval_dir), FakeClient(), "test-model", "clustering"
    )
    assert exit_code == 1
