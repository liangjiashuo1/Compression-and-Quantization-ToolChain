CUDA_VISIBLE_DEVICES=4 python calibration/build_task_calibration.py \
  --model_name_or_path /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --output_dir /data1/ljs/calibration_outputs/cola_mixed_0.15_0.45_0.4_QAanswer \
  --num_samples 1024  \
  --candidate_multiplier 4.0 \
  --sequence_length 2048 \
  --selection_method activation_clustering \
  --device cuda \
  --batch_size 4 \
  --wikitext2_ratio 0.15 \
  --commonsense_ratio 0.45 \
  --mmlu_ratio 0.4

CUDA_VISIBLE_DEVICES=4 python calibration/build_task_calibration.py \
  --model_name_or_path /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --output_dir /data1/ljs/calibration_outputs/cola_mmlu \
  --num_samples 2048  \
  --candidate_multiplier 4.0 \
  --sequence_length 2048 \
  --selection_method activation_clustering \
  --device cuda \
  --batch_size 4 \
  --wikitext2_ratio 0.0 \
  --commonsense_ratio 0.0 \
  --mmlu_ratio 1.0


CUDA_VISIBLE_DEVICES=5 python calibration/build_task_calibration.py \
  --model_name_or_path /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --output_dir /data1/ljs/calibration_outputs/cola_commonsense \
  --num_samples 2048 \
  --candidate_multiplier 4.0 \
  --sequence_length 2048 \
  --selection_method activation_clustering \
  --device cuda \
  --batch_size 4 \
  --wikitext2_ratio 0.0 \
  --commonsense_ratio 1.0 \
  --mmlu_ratio 0.0

CUDA_VISIBLE_DEVICES=7 python compare_channel.py \
  --input_model /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --svd_llm_ckpt /data1/ljs/SpinQuant_split_spin_middle_activation/AFM_models/0.05_q_k_Layerwise \
  --rotate \
  --optimized_rotation_path /data1/ljs/SpinQuant_split_spin_middle_activation/llama_rotation_split_AFM/5%_q_k_Layerwise_groupsize32_lr_1_5.bin \
  --bf16 \
  --score_mode logits_aware \
  --retained_ratio 0.40 \
  --use_global_budget \
  --score_nsamples 2048 \
  --w_bits 3 \
  --w_groupsize 64 \
  --w_clip \
  --mmlu_cola_path /data1/ljs/calibration_outputs/cola_mmlu/calibration_samples.json \
  --commonsense_cola_path /data1/ljs/calibration_outputs/cola_commonsense/calibration_samples.json \
  --mixed_cola_path /data1/ljs/calibration_outputs/cola_mixed/calibration_samples.json \
  --output_dir /data1/ljs/channel_compare_outputs_gpu \
  --score_compute_on_gpu