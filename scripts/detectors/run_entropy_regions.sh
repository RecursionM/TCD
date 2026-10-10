#!/bin/bash

# 后台指令
# nohup bash scripts/detectors/run_entropy_regions.sh > run_entropy_regions.log 2> run_entropy_regions.err < /dev/null &

# 按注意力熵把 token 分成低熵 0-25% 与高熵 75-100% 两段后分别打分，对应论文图 1。


export CUDA_VISIBLE_DEVICES=5,6


DATA_DIR="data/Benchmarks_1k"
DATASETS=(
    "RealDet.json"
)
BASE_RESULTS_DIR="./results/Fast-DetectGPT_tcd_2sections_draw"

echo "All results will be saved to: $BASE_RESULTS_DIR"


MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"


echo "Running method: Fast-DetectGPT_tcd_2sections_draw (图1)"
for FILE in "${DATASETS[@]}"; do    
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python code/detectors/eval_fast_det_gpt_tcd_2sections_draw.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR" \
        --save_file "Fast-DetectGPT_tcd_2sections_draw_${clean_name}.json" \
        --eval_data_format ImBD
done
