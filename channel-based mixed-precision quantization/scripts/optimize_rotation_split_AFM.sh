# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

# export HF_DATASETS_CACHE=/data1/wwy/WikiText-2
# export HF_HOME=/data1/wwy/WikiText-2

# # --learning_rate 1.5 \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master-port=33334 --nnodes=1 --nproc_per_node=4 ./AFM_optimize_rotation_split.py \
--input_model /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9  \
--output_rotation_path "./train_rotation" \
--output_dir "./train_output" \
--logging_dir "./log" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits 16 \
--a_bits 8 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize 32 \
--a_groupsize 32 \
--svd_llm_ckpt /data1/ljs/SpinQuant_split_spin_middle_activation/AFM_models/0.05_q_k_Layerwise \
