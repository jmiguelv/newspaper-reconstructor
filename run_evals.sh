#!/usr/bin/env bash
set -euo pipefail

PROMPTS=(
  "data/0_prompts/v00.json"
  "data/0_prompts/v01.json"
  "data/0_prompts/v02.json"
)

MODELS=(
  "arc:lite"
  "arc:nexus"
)

SAMPLE_SIZES=(
  16
  32
)

INPUT_DIR="data/0_external/alto"
GROUND_TRUTH_DIR="data/0_external/article_xml"
SEED=42

total=$(( ${#PROMPTS[@]} * ${#MODELS[@]} * ${#SAMPLE_SIZES[@]} ))
count=0

for prompt_file in "${PROMPTS[@]}"; do
  for model in "${MODELS[@]}"; do
    for sample_size in "${SAMPLE_SIZES[@]}"; do
      count=$(( count + 1 ))
      prompt_name=$(basename "$prompt_file" .json)
      echo "[$count/$total] prompt=$prompt_name model=$model sample=$sample_size"
      uv run python main.py \
        --input-dir "$INPUT_DIR" \
        --ground-truth-dir "$GROUND_TRUTH_DIR" \
        --evaluate \
        --prompt-file "$prompt_file" \
        --model "$model" \
        --sample-size "$sample_size" \
        --seed "$SEED"
      echo "---"
    done
  done
done

echo "Done: $count runs completed."
