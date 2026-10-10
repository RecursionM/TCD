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


BASE_RESULTS_DIR="./results/official_surrogate/falcon"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="tiiuae/falcon-7b-instruct"
MODEL_BASE="tiiuae/falcon-7b"


echo "Running method: Binoculars"
for FILE in "${DATASETS[@]}"; do
    filename=$(basename "$FILE")    
    clean_name="${filename%.json}"
    python code/detectors/eval_binoculars.py \
        --observer_model_path "$MODEL_BASE" \
        --performer_model_path "$MODEL_INSTRUCT" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Binoculars" \
        --save_file "Bino_${clean_name}.json" \
        --eval_data_format ImBD
done


echo "Running method: dnadetectllm"
for FILE in "${DATASETS[@]}"; do
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python code/detectors/eval_dnadetectllm.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM" \
        --save_file "dnadetectllm_${clean_name}.json" \
        --observer_name_or_path "$MODEL_BASE" \
        --performer_name_or_path "$MODEL_INSTRUCT"
done


echo "Running method: Binoculars_tcd"
for FILE in "${DATASETS[@]}"; do
    filename=$(basename "$FILE")    
    clean_name="${filename%.json}"
    python code/detectors/eval_binoculars_tcd.py \
        --observer_model_path "$MODEL_BASE" \
        --performer_model_path "$MODEL_INSTRUCT" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Binoculars_tcd" \
        --save_file "Bino_tcd_${clean_name}.json" \
        --eval_data_format ImBD
done


echo "Running method: dnadetectllm_tcd"
for FILE in "${DATASETS[@]}"; do
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python code/detectors/eval_dnadetectllm_tcd.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM_tcd" \
        --save_file "dnadetectllm_tcd_${clean_name}.json" \
        --observer_name_or_path "$MODEL_BASE" \
        --performer_name_or_path "$MODEL_INSTRUCT"
done
