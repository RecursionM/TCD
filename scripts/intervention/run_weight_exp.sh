#!/bin/bash
# =============================================================================
# Reviewer w1 intervention experiments (weight_exp).
# Runs all SIX interventions, each over the three TCD detectors
# (Fast-DetectGPT / Binoculars / DNA-DetectLLM) and the three benchmark datasets.
#
# Run from the repository root:
#   nohup bash code/intervention/run_weight_exp.sh > weight_exp.log 2> weight_exp.err < /dev/null &
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=5,6

DATA_DIR="data/Benchmarks_1k"
DATASETS=("RealDet.json" "Raid.json" "M4.json")

SAVE_ROOT="./results/intervention"

MODEL_BASE="meta-llama/Llama-3.2-1B"
MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"

# Intervention strengths for the sweepable experiments.
MASK_RATIO=0.3
HEAD_RATIO=0.3
SEED=42

COMMON_ARGS=(
    --data_dir "$DATA_DIR"
    --datasets "${DATASETS[@]}"
    --eval_data_format ImBD
    --eval_batch_size 8
    --model_base "$MODEL_BASE"
    --model_instruct "$MODEL_INSTRUCT"
    --save_root "$SAVE_ROOT"
    --temperature 0.7
    --seed "$SEED"
)

echo "All results will be saved to: $SAVE_ROOT"

echo "===== Experiment 1: mask LOW-entropy tokens ====="
python code/intervention/exp1_mask_low_entropy.py "${COMMON_ARGS[@]}" --mask_ratio "$MASK_RATIO"

echo "===== Experiment 2: mask HIGH-entropy tokens ====="
python code/intervention/exp2_mask_high_entropy.py "${COMMON_ARGS[@]}" --mask_ratio "$MASK_RATIO"

echo "===== Experiment 3: RANDOMIZE (permute) weights ====="
python code/intervention/exp3_randomize_weights.py "${COMMON_ARGS[@]}"

echo "===== Experiment 4: SWAP (invert) weights ====="
python code/intervention/exp4_swap_weights.py "${COMMON_ARGS[@]}"

echo "===== Experiment 5: ABLATE low-entropy HEADS ====="
python code/intervention/exp5_ablate_low_entropy_heads.py "${COMMON_ARGS[@]}" --head_ratio "$HEAD_RATIO"

echo "===== Experiment 6: ABLATE low-entropy LAYERS ====="
python code/intervention/exp6_ablate_low_entropy_layers.py "${COMMON_ARGS[@]}"

echo "All weight_exp experiments completed!"
