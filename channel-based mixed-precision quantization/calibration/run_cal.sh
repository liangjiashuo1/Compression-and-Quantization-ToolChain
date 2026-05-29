CUDA_VISIBLE_DEVICES=6 \
CALIBRATION_SOURCE=mixed \
ROT_BEFORE_PTQ=1 \
bash scripts/various_ASVD_eval_ptq_mmlu.sh \
  /data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  4 \
  8 \
  16 \
  /data2/wwy/Repo_of_LeeX/SVD_LLM_SpinQuant_standard_llama/svd_llm_llama_rotation/5%_16_8_16_R.bin \
  /data1/lsl/QASVDLab/model/last_layer/state_dict.pt