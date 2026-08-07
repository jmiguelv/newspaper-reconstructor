#!/usr/bin/env bash
set -euo pipefail

PROMPTS=(
  "prompts/v00.md"
  "prompts/v01.md"
  "prompts/v02.md"
  "prompts/v03.md"
  "prompts/v04.md"
)

MODELS=(
  "arc:lite"
  "arc:nexus"
)

SAMPLE_SIZES=(
  16
  32
)

ALTO_DIR="data/0_external/alto"
FRAGMENTS_DIR="data/1_interim/fragments"
GROUND_TRUTH_DIR="data/0_external/article_xml"
EVAL_DIR="reports/evaluations"
SEED=42
TIMEOUT=60

# Step 1: Parse ALTO to JSON (Do this once, it's model-agnostic)
echo "=== Step 1: Parsing ALTO ==="
if [ ! -d "$FRAGMENTS_DIR" ]; then
    uv run python main.py parse -i "$ALTO_DIR" -o "$FRAGMENTS_DIR"
else
    echo "Fragments already parsed."
fi

total=$(( ${#PROMPTS[@]} * ${#MODELS[@]} * ${#SAMPLE_SIZES[@]} ))

count=0

echo "=== Step 2: Clustering & Evaluating ==="
for prompt_file in "${PROMPTS[@]}"; do
  for model in "${MODELS[@]}"; do
    for sample_size in "${SAMPLE_SIZES[@]}"; do
      count=$(( count + 1 ))
      prompt_name=$(basename "$prompt_file" .md)
      
      run_id="openai_${model}_${prompt_name}_sample${sample_size}_seed${SEED}"
      # Replace colons for folder names safely
      safe_run_id=$(echo "$run_id" | tr ':' '_')
      
      reconstructions_dir="data/1_interim/reconstructions/$safe_run_id"
      
      # For now, skip classify and go straight to cluster since we don't have classify prompts yet
      
      echo "[${count}/${total}] prompt=$prompt_name model=$model sample=$sample_size"
      
      echo "  Clustering..."
      uv run python main.py cluster \
        -i "$FRAGMENTS_DIR" \
        -o "$reconstructions_dir" \
        -p "$prompt_file" \
        --model "$model" \
        --sample-size "$sample_size" \
        --seed "$SEED" \
        --timeout "$TIMEOUT"
        
      echo "  Evaluating..."
      uv run python main.py evaluate \
        -i "$reconstructions_dir" \
        -g "$GROUND_TRUTH_DIR" \
        --eval-dir "$EVAL_DIR" \
        --run-id "$run_id"
        
      echo "---"
    done
  done
done

echo "Done: $count runs."
