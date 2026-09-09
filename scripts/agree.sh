#!/usr/bin/env bash
# agree.sh: Compares two annotators' article XML directories on region agreement
set -euo pipefail

ANNOTATOR_A=""
ANNOTATOR_B=""
NAME_A=""
NAME_B=""
MATCH_THRESHOLD="1.0"
AGREEMENT_THRESHOLD=""
OUTPUT="reports/agreement"
COPY_AGREED=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --annotator-a) ANNOTATOR_A="$2"; shift ;;
        --annotator-b) ANNOTATOR_B="$2"; shift ;;
        --name-a) NAME_A="$2"; shift ;;
        --name-b) NAME_B="$2"; shift ;;
        --match-threshold) MATCH_THRESHOLD="$2"; shift ;;
        --agreement-threshold) AGREEMENT_THRESHOLD="$2"; shift ;;
        --output) OUTPUT="$2"; shift ;;
        --copy-agreed) COPY_AGREED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$ANNOTATOR_A" ] || [ -z "$ANNOTATOR_B" ]; then
    echo "Usage: $0 --annotator-a <dir> --annotator-b <dir> [--name-a <label>] [--name-b <label>] [--match-threshold <0<T<=1>] [--output <dir>]"
    echo ""
    echo "Example: $0 --annotator-a data/0_external/ds_article_20260902/article_xml_frial --annotator-b data/0_external/ds_article_20260902/article_xml_syafiq"
    exit 1
fi

for dir in "$ANNOTATOR_A" "$ANNOTATOR_B"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Directory does not exist: $dir"
        exit 1
    fi
done

ARGS=(--annotator-a "$ANNOTATOR_A" --annotator-b "$ANNOTATOR_B" --output "$OUTPUT")
if [ -n "$NAME_A" ]; then
    ARGS+=(--name-a "$NAME_A")
fi
if [ -n "$NAME_B" ]; then
    ARGS+=(--name-b "$NAME_B")
fi
if [ "$MATCH_THRESHOLD" != "1.0" ]; then
    ARGS+=(--match-threshold "$MATCH_THRESHOLD")
fi
if [ -n "$AGREEMENT_THRESHOLD" ]; then
    ARGS+=(--agreement-threshold "$AGREEMENT_THRESHOLD")
fi
if [ -n "$COPY_AGREED" ]; then
    ARGS+=(--copy-agreed "$COPY_AGREED")
fi

echo "=== Comparing annotations: ${NAME_A:-$(basename "$ANNOTATOR_A")} vs ${NAME_B:-$(basename "$ANNOTATOR_B")} (threshold: $MATCH_THRESHOLD) ==="

uv run article-reconstruction agree "${ARGS[@]}"

echo "Comparison complete!"
echo "----------------------------------------"
