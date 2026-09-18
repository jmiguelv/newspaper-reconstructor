import json

import numpy as np
import pytest

from scripts.train_continuation_classifier import (
    PairDataset,
    classification_metrics,
    load_records,
    positive_probabilities,
    tokenize_records,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text_a, text_b, **kwargs):
        self.calls.append((text_a, text_b, kwargs))
        return {
            "input_ids": [[1, 2, 3] for _ in text_a],
            "attention_mask": [[1, 1, 1] for _ in text_a],
        }


def test_load_records_reads_all_three_splits(tmp_path):
    record = {
        "page_id": "page-1",
        "a_id": "a",
        "b_id": "b",
        "text_a": "first",
        "text_b": "second",
        "label": 1,
    }
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    records = load_records(tmp_path)

    assert records == {
        "train": [record],
        "validation": [record],
        "test": [record],
    }


def test_load_records_can_exclude_the_test_split(tmp_path):
    record = {
        "page_id": "page-1",
        "a_id": "a",
        "b_id": "b",
        "text_a": "first",
        "text_b": "second",
        "label": 1,
    }
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
    (tmp_path / "test.jsonl").write_text("not json\n", encoding="utf-8")

    records = load_records(tmp_path, split_names=("train", "validation"))

    assert records == {"train": [record], "validation": [record]}


def test_tokenize_records_uses_standard_text_pair_tokenization():
    tokenizer = FakeTokenizer()
    records = [
        {"text_a": "first", "text_b": "second", "label": 1},
        {"text_a": "third", "text_b": "fourth", "label": 0},
    ]

    dataset = tokenize_records(records, tokenizer, max_length=128)

    assert tokenizer.calls == [
        (
            ["first", "third"],
            ["second", "fourth"],
            {"truncation": True, "max_length": 128},
        )
    ]
    assert dataset.items == [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 1},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 0},
    ]
    assert len(dataset) == 2
    assert dataset[1]["labels"] == 0


def test_positive_probabilities_uses_stable_softmax():
    probabilities = positive_probabilities(
        np.array([[1000.0, 1000.0], [1000.0, 1001.0]])
    )

    assert probabilities.tolist() == pytest.approx([0.5, 0.73105858])


def test_classification_metrics_reports_binary_metrics_and_confusion_counts():
    metrics = classification_metrics(
        probabilities=[0.1, 0.8, 0.6, 0.2], labels=[0, 1, 0, 1]
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == 0.75
    assert metrics["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_pair_dataset_exposes_items():
    dataset = PairDataset([{"labels": 1}])

    assert len(dataset) == 1
    assert dataset[0] == {"labels": 1}
