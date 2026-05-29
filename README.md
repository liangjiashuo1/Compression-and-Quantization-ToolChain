# AFM-Guided Channel-Based Mixed-Precision Quantization for LLaMA

这是一个面向 LLaMA 系列模型压缩的实验性代码仓库，核心目标是在较高压缩率下尽量保留模型能力。仓库围绕以下三部分展开：

- 基于 AFM 的逐层低秩搜索，用于为 `q_proj / k_proj` 分配共享 rank
- 基于 SpinQuant 思路的旋转矩阵优化，用于进一步降低量化误差
- 面向 MMLU / commonsense 任务的后训练量化（PTQ）评测与 retained-channel mixed precision 策略

仓库同时包含一个 CoLA calibration data curation 子模块，用于生成更适合压缩场景的校准数据。

## 项目特点

- 支持 LLaMA 模型的 `q/k` 低秩压缩与逐层贪心搜索
- 支持结合 `COLA` 校准样本进行 AFM 统计与通道打分
- 支持旋转优化后的 PTQ 流程
- 支持 `MMLU 5-shot` 和 `commonsense 0-shot` 评测脚本
- 支持 retained-channel mixed precision，用少量高精度通道保留关键能力

## 方法流程

整个流程可以概括为 4 步：

1. 生成校准数据  
   使用 `COLA` 或仓库中的 `calibration/build_task_calibration.py` 构造 `calibration_samples.json`

2. 搜索低秩配置并导出低秩模型  
   运行根目录下的 [`AAFM_LLaMA_Per_Layer_fixed_greedy.py`](./AAFM_LLaMA_Per_Layer_fixed_greedy.py)  
   该脚本会：
   - 收集 AFM 统计量
   - 对每一层的 `q/k` 共享 rank 做候选搜索
   - 在目标压缩率约束下执行 greedy selection
   - 输出 `rank_tied_qk.json`
   - 保存低秩模型到 `output_dir/ckpt`

3. 训练或优化旋转矩阵  
   运行 [`channel-based mixed-precision quantization/AFM_optimize_rotation_split.py`](./channel-based%20mixed-precision%20quantization/AFM_optimize_rotation_split.py)  
   该阶段会输出：
   - `layerwise_rmid_best.bin`
   - `layerwise_rmid_final.bin`
   - `training_metrics.jsonl`
   - `run_config.json`

4. 进行 PTQ 与下游评测  
   运行：
   - [`channel-based mixed-precision quantization/ptq_AFM_LM_eval_mmlu5shot.py`](./channel-based%20mixed-precision%20quantization/ptq_AFM_LM_eval_mmlu5shot.py)
   - [`channel-based mixed-precision quantization/ptq_AFM_LM_eval_commonsense0shot.py`](./channel-based%20mixed-precision%20quantization/ptq_AFM_LM_eval_commonsense0shot.py)

## 目录结构

```text
.
├── AAFM_LLaMA_Per_Layer_fixed_greedy.py        # AFM 统计、逐层 q/k 共享 rank 搜索、低秩模型导出
├── AFM.sh                                      # 低秩搜索示例命令
├── README.md
└── channel-based mixed-precision quantization/
    ├── AFM_optimize_rotation_split.py          # 旋转矩阵优化
    ├── ptq_AFM_LM_eval_mmlu5shot.py           # MMLU 5-shot PTQ 评测
    ├── ptq_AFM_LM_eval_commonsense0shot.py    # Commonsense 0-shot PTQ 评测
    ├── calibration/                           # 校准数据构建脚本
    ├── cola/                                  # CoLA 数据筛选模块
    ├── eval_utils/                            # 通道选择、GPTQ、旋转等评测工具
    ├── train_utils/                           # 旋转训练与量化线性层实现
    ├── utils/                                 # 参数、数据、模型与量化工具
    └── scripts/                               # 常用 shell 启动脚本
```

## 环境要求

本仓库更接近研究代码，当前没有完整整理好的 `requirements.txt`。根据现有源码，建议至少准备以下环境：

- Python 3.9+
- PyTorch
- Transformers
- Datasets
- Safetensors
- tqdm
- scikit-learn
- sentence-transformers
- lm-eval
- pyarrow

推荐运行环境：

- Linux
- CUDA GPU
- `torchrun` 可用的多卡环境

一个可参考的安装方式如下：

```bash
pip install torch transformers datasets safetensors tqdm scikit-learn sentence-transformers pyarrow
pip install lm-eval
```

如果你需要使用 `CoLA` 子模块，也可以进入对应目录执行：

```bash
cd "channel-based mixed-precision quantization/cola"
pip install -e .
```

## 快速开始

### 1. 构造校准数据

可以直接使用脚本 [`channel-based mixed-precision quantization/scripts/cola.sh`](./channel-based%20mixed-precision%20quantization/scripts/cola.sh) 中的命令作为模板，例如：

```bash
python "channel-based mixed-precision quantization/calibration/build_task_calibration.py" \
  --model_name_or_path /path/to/llama \
  --output_dir ./calibration_outputs/cola_mixed \
  --num_samples 1024 \
  --candidate_multiplier 4.0 \
  --sequence_length 2048 \
  --selection_method activation_clustering \
  --device cuda \
  --batch_size 4 \
  --wikitext2_ratio 0.15 \
  --commonsense_ratio 0.45 \
  --mmlu_ratio 0.40
```

生成结果中最重要的文件通常是：

```text
calibration_outputs/cola_mixed/calibration_samples.json
```

### 2. 搜索逐层共享 q/k rank

可参考 [`AFM.sh`](./AFM.sh)：

```bash
python AAFM_LLaMA_Per_Layer_fixed_greedy.py \
  --compress_rate 0.05 \
  --model_name_or_path /path/to/llama \
  --cola_calibration_path ./calibration_outputs/cola_mixed/calibration_samples.json \
  --output_dir ./AFM_models/llama2_7b_cr005 \
  --rank_candidates 1152,1280,1408,1536,1664,1792,1920,2048 \
  --rank_score_mode qerror_logits_aware \
  --qerror_w_bits 3 \
  --qerror_w_groupsize 64 \
  --qerror_w_clip
```

输出通常包括：

- `output_dir/rank_tied_qk.json`
- `output_dir/ckpt/`
- `output_dir/run.log`

### 3. 优化旋转矩阵

可参考 [`channel-based mixed-precision quantization/scripts/optimize_rotation_split_AFM.sh`](./channel-based%20mixed-precision%20quantization/scripts/optimize_rotation_split_AFM.sh)：

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
  "channel-based mixed-precision quantization/AFM_optimize_rotation_split.py" \
  --input_model /path/to/llama \
  --output_rotation_path ./train_rotation \
  --output_dir ./train_output \
  --model_max_length 2048 \
  --bf16 True \
  --learning_rate 1.5 \
  --max_steps 100 \
  --w_bits 16 \
  --a_bits 8 \
  --k_bits 16 \
  --v_bits 16 \
  --rotate \
  --svd_llm_ckpt ./AFM_models/llama2_7b_cr005/ckpt
```

### 4. 运行 PTQ 评测

可参考：

- [`channel-based mixed-precision quantization/scripts/test_for_lmhead_mmlu.sh`](./channel-based%20mixed-precision%20quantization/scripts/test_for_lmhead_mmlu.sh)
- [`channel-based mixed-precision quantization/scripts/test_for_lmhead_cs.sh`](./channel-based%20mixed-precision%20quantization/scripts/test_for_lmhead_cs.sh)

MMLU 示例：

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
  "channel-based mixed-precision quantization/ptq_AFM_LM_eval_mmlu5shot.py" \
  --input_model /path/to/llama \
  --per_device_eval_batch_size 32 \
  --model_max_length 2048 \
  --bf16 True \
  --w_bits 3 \
  --a_bits 8 \
  --k_bits 16 \
  --v_bits 16 \
  --w_clip \
  --a_asym \
  --k_asym \
  --v_asym \
  --k_groupsize 128 \
  --v_groupsize 128 \
  --w_groupsize 128 \
  --a_groupsize 128 \
  --rotate \
  --nsamples 1024 \
  --optimized_rotation_path ./train_rotation/your_run/layerwise_rmid_final.bin \
  --svd_llm_ckpt ./AFM_models/llama2_7b_cr005/ckpt
```

## CoLA 子模块

仓库中包含一个相对独立的 CoLA 实现，位于 [`channel-based mixed-precision quantization/cola`](./channel-based%20mixed-precision%20quantization/cola)。

它的目标是为压缩任务构造更高质量的 calibration data，核心包括：

- 数据集选择
- 数据处理与格式增强
- 基于激活空间的样本筛选

示例脚本：

```bash
python "channel-based mixed-precision quantization/cola/run_cola.py" \
  --model_name_or_path /path/to/llama \
  --output_dir ./cola_output \
  --num_samples 128 \
  --sequence_length 2048 \
  --target_capabilities commonsense math code \
  --datasets wikitext c4 DKYoon/slimpajama-200k
```


