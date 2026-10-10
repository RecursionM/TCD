#!/bin/bash

# 后台指令示例
# nohup bash scripts/run_zero.sh > run_zero.log 2> run_zero.err < /dev/null &

export CUDA_VISIBLE_DEVICES=5,6

# 1. 定义需要遍历的单词长度列表
WORD_LENGTHS=("40" "80" "120" "160" "200" "240" "280")

# 数据集列表
DATASETS=(
    "Raid.json"
)

# 模型路径
MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

# 2. 外层循环：遍历每一个长度的目录
for LENGTH in "${WORD_LENGTHS[@]}"; do
    
    # 动态构建当前长度的目录名称和路径
    CURRENT_WORD_DIR="${LENGTH}_words"
    DATA_DIR="data/vary_len/${CURRENT_WORD_DIR}"
    BASE_RESULTS_DIR="./results/text_length/${CURRENT_WORD_DIR}"

    echo "==============================================================="
    echo "开始处理文本长度: ${CURRENT_WORD_DIR}"
    echo "数据来源: $DATA_DIR"
    echo "结果保存至: $BASE_RESULTS_DIR"
    echo "==============================================================="

    # --- 以下为原有的各方法执行逻辑，路径已随外层循环自动更新 ---

    echo "Running method: Fast-DetectGPT_tcd for ${CURRENT_WORD_DIR}"
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

    echo "Running method: Binoculars_tcd for ${CURRENT_WORD_DIR}"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        echo "Processing file: $FILE"
        python code/detectors/eval_binoculars_tcd.py \
            --observer_model_path "$MODEL_BASE" \
            --performer_model_path "$MODEL_INSTRUCT" \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/Binoculars_tcd" \
            --save_file "Bino_tcd_${clean_name}.json" \
            --eval_data_format ImBD
    done

    echo "Running method: dnadetectllm_tcd for ${CURRENT_WORD_DIR}"
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

    echo "Running method: Fast-DetectGPT for ${CURRENT_WORD_DIR}"
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

    echo "Running method: Binoculars for ${CURRENT_WORD_DIR}"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        echo "Processing file: $FILE"
        python code/detectors/eval_binoculars.py \
            --observer_model_path "$MODEL_BASE" \
            --performer_model_path "$MODEL_INSTRUCT" \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/Binoculars" \
            --save_file "Bino_${clean_name}.json" \
            --eval_data_format ImBD
    done

    echo "Running method: dnadetectllm for ${CURRENT_WORD_DIR}"
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

    echo "完成处理文本长度: ${CURRENT_WORD_DIR}"
    echo "---------------------------------------------------------------"

done

echo "所有长度的目录已全部处理完毕！"
