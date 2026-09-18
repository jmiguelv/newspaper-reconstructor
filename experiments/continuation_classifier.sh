#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: experiments/continuation_classifier.sh <prepare|train|val|test> [options]

Steps:
  prepare Build balanced, page-disjoint train/validation/test data.
  train   Fine-tune the classifier using prepared train and validation data.
  val     Evaluate the saved classifier on the validation split.
  test    Evaluate the validated classifier on the test split.

Options:
  --dataset NAME   Dataset under data/0_external
  --model NAME     Hugging Face model name or local path
  --seed N         Dataset and training seed
  --epochs N       Training epochs (train step only, default 3)
  --dry-run        Print commands without running them
  --help           Show this help
EOF
}

if [[ $# -eq 0 || $1 == "--help" ]]; then
    usage
    exit 0
fi

STEP=$1
shift
case "$STEP" in
    prepare|train|val|test) ;;
    *)
        echo "Unknown step: $STEP" >&2
        usage >&2
        exit 1
        ;;
esac

DATASET="ds-articlereconstruction-20260821"
MODEL="distilbert/distilbert-base-multilingual-cased"
SEED="42"
EPOCHS="3"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --dataset|--model|--seed|--epochs)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            case "$1" in
                --dataset) DATASET=$2 ;;
                --model) MODEL=$2 ;;
                --seed) SEED=$2 ;;
                --epochs) EPOCHS=$2 ;;
            esac
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

ARTICLES_DIR="data/0_external/${DATASET}/articles"
REGIONS_DIR="data/0_external/${DATASET}/regions"
BASE_DIR="data/1_interim/${DATASET}/continuation"
DATA_DIR="${BASE_DIR}/data-seed${SEED}-balanced"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/: ' '___')"
MODEL_DIR="${BASE_DIR}/models/${MODEL_SLUG}-balanced-seed${SEED}"

run() {
    if [[ $DRY_RUN == "true" ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

case "$STEP" in
    prepare)
        if [[ $DRY_RUN != "true" && (! -d $ARTICLES_DIR || ! -d $REGIONS_DIR) ]]; then
            echo "Expected ${ARTICLES_DIR} and ${REGIONS_DIR}" >&2
            exit 1
        fi
        run uv run python scripts/build_continuation_dataset.py \
            --articles-dir "$ARTICLES_DIR" \
            --regions-dir "$REGIONS_DIR" \
            --output-dir "$DATA_DIR" \
            --seed "$SEED"
        ;;
    train)
        if [[ $DRY_RUN != "true" && ! -f ${DATA_DIR}/manifest.json ]]; then
            echo "Missing ${DATA_DIR}/manifest.json; run the prepare step first" >&2
            exit 1
        fi
        if [[ $DRY_RUN != "true" ]]; then
            rm -f "$MODEL_DIR/validation-metrics.json" "$MODEL_DIR/test-metrics.json"
        fi
        run uv run --group local python scripts/train_continuation_classifier.py train \
            --data-dir "$DATA_DIR" \
            --output-dir "$MODEL_DIR" \
            --model "$MODEL" \
            --seed "$SEED" \
            --epochs "$EPOCHS"
        ;;
    val)
        if [[ $DRY_RUN != "true" && ! -f ${MODEL_DIR}/config.json ]]; then
            echo "Missing ${MODEL_DIR}; run the train step first" >&2
            exit 1
        fi
        run uv run --group local python scripts/train_continuation_classifier.py evaluate \
            --data-dir "$DATA_DIR" \
            --model-dir "$MODEL_DIR" \
            --split validation
        ;;
    test)
        if [[ $DRY_RUN != "true" && ! -f ${MODEL_DIR}/validation-metrics.json ]]; then
            echo "Missing validation metrics; run the val step first" >&2
            exit 1
        fi
        run uv run --group local python scripts/train_continuation_classifier.py evaluate \
            --data-dir "$DATA_DIR" \
            --model-dir "$MODEL_DIR" \
            --split test
        ;;
esac
