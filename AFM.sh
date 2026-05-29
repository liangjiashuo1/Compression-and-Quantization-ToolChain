CUDA_VISIBLE_DEVICES=4,5 python AAFM_LLaMA_Per_Layer_fixed_greedy.py \
  --compress_rate 0.05 \
  --model_name_or_path /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --cola_calibration_path /data1/ljs/calibration_outputs/cola_mixed_0.15_0.45_0.4_QAanswer/calibration_samples.json \
  --output_dir ./AFM_models/AAFM_quant_friendly_llama2_7b_cr005_2048 \
  --rank_candidates 1152,1280,1408,1536,1664,1792,1920,2048 \
  --rank_score_mode qerror_logits_aware \
  --qerror_w_bits 3 \
  --qerror_w_groupsize 64 \
  --qerror_w_clip


  


CUDA_VISIBLE_DEVICES=2,6 python AAFM_LLaMA_Per_Layer_fixed_greedy.py \
  --compress_rate 0.08 \
  --model_name_or_path /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --cola_calibration_path /data1/ljs/calibration_outputs/cola_mixed_0.15_0.45_0.4_QAanswer/calibration_samples.json \
  --output_dir ./AFM_models/8% \
  --rank_candidates 128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048 \
  --rank_score_mode qerror_logits_aware \
  --qerror_w_bits 3 \
  --qerror_w_groupsize 128 \
  --qerror_w_clip