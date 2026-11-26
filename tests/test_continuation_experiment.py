import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "experiments" / "continuation_classifier.sh"


def _run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        check=False,
        text=True,
    )


def test_help_lists_prepare_train_validation_and_test_steps():
    result = _run("--help")

    assert result.returncode == 0
    assert "prepare" in result.stdout
    assert "train" in result.stdout
    assert "val" in result.stdout
    assert "test" in result.stdout


def test_prepare_dry_run_only_builds_data():
    result = _run("prepare", "--dataset", "example", "--dry-run")

    assert result.returncode == 0
    assert "build_continuation_dataset.py" in result.stdout
    assert "train_continuation_classifier.py" not in result.stdout
    assert "data-seed42-balanced" in result.stdout


def test_train_dry_run_only_trains_existing_data():
    result = _run("train", "--dataset", "example", "--dry-run")

    assert result.returncode == 0
    assert "build_continuation_dataset.py" not in result.stdout
    assert "train_continuation_classifier.py train" in result.stdout
    assert "data-seed42-balanced" in result.stdout


def test_val_dry_run_evaluates_only_validation_split():
    result = _run("val", "--dataset", "example", "--dry-run")

    assert result.returncode == 0
    assert "train_continuation_classifier.py evaluate" in result.stdout
    assert "--split validation" in result.stdout
    assert "--split test" not in result.stdout


def test_test_dry_run_evaluates_only_test_split():
    result = _run("test", "--dataset", "example", "--dry-run")

    assert result.returncode == 0
    assert "train_continuation_classifier.py evaluate" in result.stdout
    assert "--split test" in result.stdout
    assert "--split validation" not in result.stdout


def test_unknown_step_is_rejected():
    result = _run("unknown")

    assert result.returncode != 0
    assert "Unknown step" in result.stderr


def test_prepare_dry_run_with_variant_passes_split_options():
    result = _run(
        "prepare",
        "--variant",
        "split400",
        "--max-chars",
        "400",
        "--filter-gibberish",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "--max-region-chars 400" in result.stdout
    assert "--filter-gibberish" in result.stdout
    assert "data-seed42-split400" in result.stdout


def test_train_dry_run_accepts_equals_sign_option_form():
    result = _run("train", "--variant=balanced8192", "--dry-run")

    assert result.returncode == 0
    assert "data-seed42-balanced8192" in result.stdout


def test_train_dry_run_with_data_variant_and_batch_size():
    result = _run(
        "train",
        "--variant",
        "balanced8192",
        "--data-variant",
        "balanced",
        "--batch-size",
        "8",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "data-seed42-balanced " in result.stdout
    assert "data-seed42-balanced8192" not in result.stdout
    assert "balanced8192-seed42" in result.stdout
    assert "--train-batch-size 8 --eval-batch-size 8" in result.stdout


def test_train_dry_run_with_variant_and_max_length():
    result = _run(
        "train",
        "--variant",
        "split400",
        "--max-length",
        "8192",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "--max-length 8192" in result.stdout
    assert "split400-seed42" in result.stdout
