#!/bin/bash

# 后台指令
# nohup bash scripts/run_zero.sh > run_zero.log 2> run_zero.err < /dev/null &


export CUDA_VISIBLE_DEVICES=5,6


DATA_DIR="data/Benchmarks_1k"


# 数据集列表 (只需要修改这里即可应用到所有方法)
DATASETS=(
    "RealDet.json"
    "Raid.json"
    "M4.json"
)

# TIMESTAMP=$(date +"%Y%m%d_%H%M")


BASE_RESULTS_DIR="./results/official_surrogate/gpt"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="EleutherAI/gpt-neo-2.7B"
MODEL_BASE="EleutherAI/gpt-j-6b"


echo "Running method: Fast-DetectGPT"
for FILE in "${DATASETS[@]}"; do    
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python code/detectors/eval_fast_det_gpt.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT" \
        --save_file "Fast-DetectGPT_${clean_name}.json" \
        --eval_data_format ImBD
done


echo "Running method: Fast-DetectGPT_tcd"
for FILE in "${DATASETS[@]}"; do    
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python code/detectors/eval_fast_det_gpt_tcd.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT_tcd" \
        --save_file "Fast-DetectGPT_tcd_${clean_name}.json" \
        --eval_data_format ImBD
done
