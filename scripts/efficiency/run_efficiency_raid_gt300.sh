#!/bin/bash

set -euo pipefail

# 后台示例：
# nohup bash scripts/run_efficiency_raid_gt300.sh > run_efficiency_raid_gt300.log 2> run_efficiency_raid_gt300.err < /dev/null &

# 按需修改：
export CUDA_VISIBLE_DEVICES=5,6

DATA_FILE="data/Benchmarks_1k/Raid_gt300_balanced_1000_llama3.2-1B-Instruct.json"
DATA_FORMAT="ImBD"
BATCH_SIZE=1
MAX_TOKENS=300

# 模型路径（与 scripts/run_my.sh 保持一致风格）
MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

OUT_BASE="result/efficiency"

mkdir -p "$OUT_BASE"

echo "Dataset: $DATA_FILE"
echo "Output base: $OUT_BASE"


echo "[1/6] Fast-DetectGPT_tcd"
python code/efficiency/measure_fast_det_gpt_tcd_efficiency.py \
  --scoring_model_name "$MODEL_INSTRUCT" \
  --reference_model_name "$MODEL_BASE" \
  --discrepancy_analytic \
  --eval_data_path "$DATA_FILE" \
  --eval_data_format "$DATA_FORMAT" \
  --eval_batch_size $BATCH_SIZE \
  --max_token_observed $MAX_TOKENS \
  --save_path "$OUT_BASE/Fast-DetectGPT_tcd"


echo "[2/6] Binoculars_tcd"
python code/efficiency/measure_binoculars_tcd_efficiency.py \
  --observer_model_path "$MODEL_BASE" \
  --performer_model_path "$MODEL_INSTRUCT" \
  --eval_data_path "$DATA_FILE" \
  --eval_data_format "$DATA_FORMAT" \
  --eval_batch_size $BATCH_SIZE \
  --max_token_observed $MAX_TOKENS \
  --save_path "$OUT_BASE/Binoculars_tcd"


echo "[3/6] DNA-DetectLLM_tcd"
python code/efficiency/measure_dnadetectllm_tcd_efficiency.py \
  --observer_name_or_path "$MODEL_BASE" \
  --performer_name_or_path "$MODEL_INSTRUCT" \
  --eval_data_path "$DATA_FILE" \
  --eval_data_format "$DATA_FORMAT" \
  --eval_batch_size $BATCH_SIZE \
  --max_token_observed $MAX_TOKENS \
  --save_path "$OUT_BASE/DNA-DetectLLM_tcd"


echo "[4/6] Fast-DetectGPT"
python code/efficiency/measure_fast_det_gpt_efficiency.py \
  --scoring_model_name "$MODEL_INSTRUCT" \
  --reference_model_name "$MODEL_BASE" \
  --discrepancy_analytic \
  --eval_data_path "$DATA_FILE" \
  --eval_data_format "$DATA_FORMAT" \
  --eval_batch_size $BATCH_SIZE \
  --max_token_observed $MAX_TOKENS \
  --save_path "$OUT_BASE/Fast-DetectGPT"


echo "[5/6] Binoculars"
python code/efficiency/measure_binoculars_efficiency.py \
  --observer_model_path "$MODEL_BASE" \
  --performer_model_path "$MODEL_INSTRUCT" \
  --eval_data_path "$DATA_FILE" \
  --eval_data_format "$DATA_FORMAT" \
  --eval_batch_size $BATCH_SIZE \
  --max_token_observed $MAX_TOKENS \
  --save_path "$OUT_BASE/Binoculars"


echo "[6/6] DNA-DetectLLM"
python code/efficiency/measure_dnadetectllm_efficiency.py \
  --observer_name_or_path "$MODEL_BASE" \
  --performer_name_or_path "$MODEL_INSTRUCT" \
  --eval_data_path "$DATA_FILE" \
  --eval_data_format "$DATA_FORMAT" \
  --eval_batch_size $BATCH_SIZE \
  --max_token_observed $MAX_TOKENS \
  --save_path "$OUT_BASE/DNA-DetectLLM"


echo "Done."
