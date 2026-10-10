#!/bin/bash

# Robustness benchmark: run 6 detection methods on RAID Robustness datasets.
# Usage (from project root):
#   bash scripts/run_robustness.sh
# Background:
#   nohup bash scripts/run_robustness.sh > run_robustness.log 2> run_robustness.err < /dev/null &

set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

export CUDA_VISIBLE_DEVICES=5,6

# Robustness data (ImBD format: {"human": [...], "llm": [...]})
DATA_DIR="data/raid_robustness"

DATASETS=(
    "attack/alternative_spelling.json"
    "attack/article_deletion.json"
    "attack/homoglyph.json"
    "attack/insert_paragraphs.json"
    "attack/none.json"
    "attack/number.json"
    "attack/paraphrase.json"
    "attack/perplexity_misspelling.json"
    "attack/synonym.json"
    "attack/upper_lower.json"
    "attack/whitespace.json"
    "attack/zero_width_space.json"
)

BASE_RESULTS_DIR="./results/robustness"
EVAL_BATCH_SIZE=8

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

# Use path-based names to avoid collisions (e.g. attack/none -> attack_none)
clean_name_from_path() {
    local file="$1"
    local base="${file%.json}"
    echo "${base//\//_}"
}

echo "Project root: $(pwd)"
echo "Data dir:     $DATA_DIR"
echo "Results dir:  $BASE_RESULTS_DIR"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Models:       $MODEL_BASE (observer/ref) | $MODEL_INSTRUCT (scoring/performer)"
echo ""

for FILE in "${DATASETS[@]}"; do
    if [[ ! -f "$DATA_DIR/$FILE" ]]; then
        echo "Error: missing dataset file: $DATA_DIR/$FILE"
        exit 1
    fi
done

echo "Running method: Fast-DetectGPT"
for FILE in "${DATASETS[@]}"; do
    clean_name="$(clean_name_from_path "$FILE")"
    python code/detectors/eval_fast_det_gpt.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT" \
        --save_file "Fast-DetectGPT_${clean_name}.json" \
        --eval_data_format ImBD
done

echo "Running method: Fast-DetectGPT_tcd"
for FILE in "${DATASETS[@]}"; do
    clean_name="$(clean_name_from_path "$FILE")"
    python code/detectors/eval_fast_det_gpt_tcd.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT_tcd" \
        --save_file "Fast-DetectGPT_tcd_${clean_name}.json" \
        --eval_data_format ImBD
done

echo "Running method: Binoculars"
for FILE in "${DATASETS[@]}"; do
    clean_name="$(clean_name_from_path "$FILE")"
    python code/detectors/eval_binoculars.py \
        --observer_model_path "$MODEL_BASE" \
        --performer_model_path "$MODEL_INSTRUCT" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --save_path "$BASE_RESULTS_DIR/Binoculars" \
        --save_file "Bino_${clean_name}.json" \
        --eval_data_format ImBD
done

echo "Running method: Binoculars_tcd"
for FILE in "${DATASETS[@]}"; do
    clean_name="$(clean_name_from_path "$FILE")"
    python code/detectors/eval_binoculars_tcd.py \
        --observer_model_path "$MODEL_BASE" \
        --performer_model_path "$MODEL_INSTRUCT" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --save_path "$BASE_RESULTS_DIR/Binoculars_tcd" \
        --save_file "Bino_tcd_${clean_name}.json" \
        --eval_data_format ImBD
done

echo "Running method: DNA-DetectLLM"
for FILE in "${DATASETS[@]}"; do
    clean_name="$(clean_name_from_path "$FILE")"
    python code/detectors/eval_dnadetectllm.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM" \
        --save_file "dnadetectllm_${clean_name}.json" \
        --observer_name_or_path "$MODEL_BASE" \
        --performer_name_or_path "$MODEL_INSTRUCT"
done

echo "Running method: DNA-DetectLLM_tcd"
for FILE in "${DATASETS[@]}"; do
    clean_name="$(clean_name_from_path "$FILE")"
    python code/detectors/eval_dnadetectllm_tcd.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM_tcd" \
        --save_file "dnadetectllm_tcd_${clean_name}.json" \
        --observer_name_or_path "$MODEL_BASE" \
        --performer_name_or_path "$MODEL_INSTRUCT"
done

echo ""
echo "All done. Results saved under: $BASE_RESULTS_DIR"
