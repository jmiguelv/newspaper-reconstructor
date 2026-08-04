#!/usr/bin/env bash
set -euo pipefail

PROMPTS=(
  "data/0_prompts/v00.md"
  "data/0_prompts/v01.md"
  "data/0_prompts/v02.md"
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
EVAL_DIR="data/2_evaluations"
SEED=42
TIMEOUT=60

total=$(( ${#PROMPTS[@]} * ${#MODELS[@]} * ${#SAMPLE_SIZES[@]} ))

# Check which runs are already completed
completed=0
for prompt_file in "${PROMPTS[@]}"; do
  for model in "${MODELS[@]}"; do
    for sample_size in "${SAMPLE_SIZES[@]}"; do
      prompt_name=$(basename "$prompt_file" .md)
      pattern="*_openai_${model}_${prompt_name}_sample${sample_size}_seed${SEED}.json"
      if ls "${EVAL_DIR}"/${pattern} 1>/dev/null 2>&1; then
        completed=$(( completed + 1 ))
      fi
    done
  done
done

SKIP_COMPLETED=false

if [ "$completed" -gt 0 ]; then
  remaining=$(( total - completed ))
  echo "Found $completed/$total runs already completed. $remaining remaining."
  echo -n "Continue in 5s... (Press Enter to re-run all, Ctrl-C to cancel) "
  if read -t 5 -n 1; then
    echo ""
    echo "Re-running all $total runs."
  else
    echo ""
    SKIP_COMPLETED=true
    echo "Skipping $completed completed runs."
  fi
else
  echo "No completed runs found. Running all $total combos."
fi

count=0
skipped=0

for prompt_file in "${PROMPTS[@]}"; do
  for model in "${MODELS[@]}"; do
    for sample_size in "${SAMPLE_SIZES[@]}"; do
      count=$(( count + 1 ))
      prompt_name=$(basename "$prompt_file" .md)

      if $SKIP_COMPLETED; then
        pattern="*_openai_${model}_${prompt_name}_sample${sample_size}_seed${SEED}.json"
        if ls "${EVAL_DIR}"/${pattern} 1>/dev/null 2>&1; then
          skipped=$(( skipped + 1 ))
          echo "[${count}/${total}] prompt=$prompt_name model=$model sample=$sample_size — SKIPPED"
          continue
        fi
      fi

      echo "[${count}/${total}] prompt=$prompt_name model=$model sample=$sample_size"
      model_timeout=$TIMEOUT
      uv run python main.py \
        --input-dir "$INPUT_DIR" \
        --ground-truth-dir "$GROUND_TRUTH_DIR" \
        --evaluate \
        --prompt-file "$prompt_file" \
        --model "$model" \
        --sample-size "$sample_size" \
        --timeout "$model_timeout" \
        --seed "$SEED"
      echo "---"
    done
  done
done

echo "Done: $count runs. $skipped skipped."
