#!/bin/bash

# 后台指令
# nohup bash scripts/run_zero.sh > run_zero.log 2> run_zero.err < /dev/null &


export CUDA_VISIBLE_DEVICES=5,6


# ===== 论文用到的三组数据集，改 GROUP 即可切换 =====
#   benchmark   表1、表10、表11、表12（RealDet / RAID / M4）
#   hart        表2、表8（HART Level 1~3 × arxiv/essay/news/writing）
#   raid_domain 表7、图3（RAID 八个体裁）
GROUP="benchmark"

case "$GROUP" in
    benchmark)
        DATA_DIR="data/Benchmarks_1k"
        DATASETS=(
            "RealDet.json"
            "Raid.json"
            "M4.json"
        )
        BASE_RESULTS_DIR="./results/main_benchmarks"
        # 换代理模型跑表10 / 表11 时，同时改下面的 MODEL_* 与此处输出目录：
        # BASE_RESULTS_DIR="./results/main_benchmarks_llama-3.2-3B"
        # BASE_RESULTS_DIR="./results/main_benchmarks_gemma-2-2b"
        ;;
    hart)
        DATA_DIR="data/hart"
        DATASETS=(
            "level-1/arxiv.test.level1.extracted.json"
            "level-1/essay.test.level1.extracted.json"
            "level-1/news.test.level1.extracted.json"
            "level-1/writing.test.level1.extracted.json"
            "level-2/arxiv.test.level2.extracted.json"
            "level-2/essay.test.level2.extracted.json"
            "level-2/news.test.level2.extracted.json"
            "level-2/writing.test.level2.extracted.json"
            "level-3/arxiv.test.level3.extracted.json"
            "level-3/essay.test.level3.extracted.json"
            "level-3/news.test.level3.extracted.json"
            "level-3/writing.test.level3.extracted.json"
        )
        BASE_RESULTS_DIR="./results/hart"
        ;;
    raid_domain)
        DATA_DIR="data/raid_multi_domain"
        DATASETS=(
            "abstracts.json"
            "books.json"
            "news.json"
            "poetry.json"
            "recipes.json"
            "reddit.json"
            "reviews.json"
            "wiki.json"
        )
        BASE_RESULTS_DIR="./results/raid_multi_domain"
        ;;
    *)
        echo "未知的 GROUP: $GROUP" >&2
        exit 1
        ;;
esac

echo "All results will be saved to: $BASE_RESULTS_DIR"


MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

# MODEL_INSTRUCT="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_BASE="meta-llama/Llama-3.2-3B"

# MODEL_INSTRUCT="google/gemma-2-2b-it"
# MODEL_BASE="google/gemma-2-2b"


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
