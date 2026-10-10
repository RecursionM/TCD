#!/bin/bash

# 后台指令
# nohup bash scripts/run_zero.sh > run_zero.log 2> run_zero.err < /dev/null &

export CUDA_VISIBLE_DEVICES=5,6

DATA_DIR="data/Benchmarks_1k"

# 数据集列表
DATASETS=(
    "RealDet.json"
    "Raid.json"
    "M4.json"
)

# 定义要遍历的温度参数
TEMPERATURES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

BASE_RESULTS_DIR="./results/hyperparameters"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"


echo "Running method: Fast-DetectGPT_tcd"
for TEMP in "${TEMPERATURES[@]}"; do
    echo "--> Testing Fast-DetectGPT_tcd with temperature: $TEMP"
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
            --save_file "Fast-DetectGPT_tcd_${clean_name}_temp_${TEMP}.json" \
            --eval_data_format ImBD \
            --temperature "$TEMP"
    done
done


echo "Running method: Binoculars_tcd"
for TEMP in "${TEMPERATURES[@]}"; do
    echo "--> Testing Binoculars_tcd with temperature: $TEMP"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        python code/detectors/eval_binoculars_tcd.py \
            --observer_model_path "$MODEL_BASE" \
            --performer_model_path "$MODEL_INSTRUCT" \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/Binoculars_tcd" \
            --save_file "Bino_tcd_${clean_name}_temp_${TEMP}.json" \
            --eval_data_format ImBD \
            --temperature "$TEMP"
    done
done


echo "Running method: dnadetectllm_tcd"
for TEMP in "${TEMPERATURES[@]}"; do
    echo "--> Testing dnadetectllm_tcd with temperature: $TEMP"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        python code/detectors/eval_dnadetectllm_tcd.py \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_data_format ImBD \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM_tcd" \
            --save_file "dnadetectllm_tcd_${clean_name}_temp_${TEMP}.json" \
            --observer_name_or_path "$MODEL_BASE" \
            --performer_name_or_path "$MODEL_INSTRUCT" \
            --temperature "$TEMP"
    done
done

echo "All tasks completed!"
