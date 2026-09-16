#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-ds-filteredUM1956alto}"
OCR_DIR="data/0_external/${DATASET}/json"
GROUND_TRUTH_DIR="data/0_external/${DATASET}/article_xml"
EVAL_DIR="reports/evaluations/${DATASET}"
VARIANTS=(singletons one-cluster xy-cut xy-cut-plus geometry geometry-ncd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

if [[ ! -d "$OCR_DIR" || ! -d "$GROUND_TRUTH_DIR" ]]; then
    echo "Expected ${OCR_DIR} and ${GROUND_TRUTH_DIR}" >&2
    exit 1
fi

for variant in "${VARIANTS[@]}"; do
    experiment_id="${DATASET}_baseline-${variant}"
    output_dir="data/1_interim/${DATASET}/reconstructions/${experiment_id}"

    echo "=== ${experiment_id} ==="
    uv run article-reconstruction baseline \
        --input-folder "$OCR_DIR" \
        --output-folder "$output_dir" \
        --variant "$variant"
    uv run article-reconstruction evaluate \
        --input-folder "$output_dir" \
        --ground-truth-folder "$GROUND_TRUTH_DIR" \
        --task reconstruction \
        --eval-dir "$EVAL_DIR" \
        --experiment-id "$experiment_id"
done
