# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Usage:
#   CALIBRATION_SOURCE=wiki ROT_BEFORE_PTQ=0 bash scripts/various_ASVD_eval_ptq_mmlu.sh \
#     MODEL_PATH W_BITS A_BITS K_BITS ROTATION_PATH SVD_CKPT [MMLU_LOG_PATH]
#
# CALIBRATION_SOURCE: wiki | mmlu | mixed | cola
# ROT_BEFORE_PTQ: 0 | 1

set -u

CALIBRATION_SOURCE="${CALIBRATION_SOURCE:-wiki}"
ROT_BEFORE_PTQ="${ROT_BEFORE_PTQ:-0}"
MASTER_PORT="${MASTER_PORT:-23479}"
PER_DEVICE_EVAL_BATCH_SIZE=4 
MODEL_MAX_LENGTH=2048

export HF_DATASETS_CACHE=/data1/wwy/WikiText-2
export HF_HOME=/data1/wwy/WikiText-2
export COLA_CALIBRATION_PATH=../COLA/cola_output/calibration_samples.json

if [[ "$#" -lt 6 || "$#" -gt 7 ]]; then
  echo "Usage: CALIBRATION_SOURCE=wiki|mmlu|mixed|cola ROT_BEFORE_PTQ=0|1 bash $0 MODEL_PATH W_BITS A_BITS K_BITS ROTATION_PATH SVD_CKPT [MMLU_LOG_PATH]" >&2
  exit 2
fi

case "$CALIBRATION_SOURCE" in
  wiki|mmlu|mixed|cola) ;;
  *)
    echo "Unsupported CALIBRATION_SOURCE=$CALIBRATION_SOURCE; use wiki, mmlu, mixed, or cola" >&2
    exit 2
    ;;
esac

if [[ "$ROT_BEFORE_PTQ" == "1" ]]; then
  ROT_ARG=(--rot_before_ptq)
  DEFAULT_MMLU_LOG_PATH="./mmlu_log/rot_before_ptq/${CALIBRATION_SOURCE}/$2"
else
  ROT_ARG=()
  DEFAULT_MMLU_LOG_PATH="./mmlu_log/${CALIBRATION_SOURCE}/$2"
fi

MMLU_LOG_PATH="${7:-$DEFAULT_MMLU_LOG_PATH}"

torchrun --master-port="$MASTER_PORT" --nnodes=1 --nproc_per_node=1 various_cal_ptq_mmlu.py \
--calibration_source "$CALIBRATION_SOURCE" \
"${ROT_ARG[@]}" \
--input_model "$1" \
--do_train False \
--do_eval True \
--per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
--model_max_length "$MODEL_MAX_LENGTH" \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits "$2" \
--a_bits "$3" \
--k_bits "$4" \
--v_bits "$4" \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--optimized_rotation_path "$5" \
--svd_llm_ckpt "$6" \
--mmlu_log_path "$MMLU_LOG_PATH"
