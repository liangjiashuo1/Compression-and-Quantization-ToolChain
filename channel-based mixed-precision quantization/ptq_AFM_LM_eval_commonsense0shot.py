import datetime
import json
import os
from logging import Logger

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
import transformers
from transformers import LlamaTokenizerFast
from tqdm import tqdm

import lm_eval
from lm_eval.models.huggingface import HFLM

from eval_utils.main import ptq_model
from ptq import (
    KEEP_QUANTIZER_STATE_AFTER_PTQ,
    PRINT_QUANTIZER_SHAPES_AFTER_EVAL,
    build_model_and_config,
    enable_quantizer_state_preservation,
    log_quantizer_shapes,
    maybe_apply_trained_rmid,
    prepare_run_output_dir,
    repair_empty_split_biases,
    save_final_eval_results,
    save_lowrank_qk_retained_channel_table,
    save_run_config,
)
from utils import data_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")

PPL_EVAL_SEQLEN = 2048
PPL_EVAL_BATCH_SIZE = 1
KEEP_QUANTIZER_STATE_FOR_PPL = False

# ============================================================================
# Low-rank q/k mixed-precision policy (commensense0shot 专用)
# 说明:
# 1. 这一组参数只服务 commensense0shot 入口，不继承 ptq.py 的同名默认值。
# 2. 修改这里的超参数，只影响 commensense0shot，不影响 ptq.py / mmlu5shot。
# ============================================================================

# --- A. 是否启用 low-rank q/k retained-channel mixed precision ---
EVAL_LOWRANK_QK_FP16_MIXED_ENABLED = True

# --- B. retained channel 打分与 bit 分配 ---
# score mode 定义如何衡量每个 bottleneck channel 的重要性
EVAL_LOWRANK_QK_RETAIN_SELECTION_MODE = "logits_aware"

# retained ratio: 选中多少比例的 bottleneck channel 走高精度
EVAL_LOWRANK_QK_RETAINED_CHANNEL_RATIO = 0.40

# 非 retained channel 的基础 bit-width
EVAL_LOWRANK_QK_QUANT_BITS = 3

# retained channel 的高精度 bit-width
EVAL_LOWRANK_QK_HIGHPREC_BITS = 4

# True: 跨所有层/所有 qk channel 做全局预算分配
EVAL_LOWRANK_QK_USE_GLOBAL_BUDGET = True

# --- C. channel 重排与 block 对齐 ---
# True: 按 score 将高分 channel 重排成连续前缀
EVAL_LOWRANK_QK_CHANNEL_REORDER_ENABLED = True

# 当前只支持 "score_desc"
EVAL_LOWRANK_QK_CHANNEL_REORDER_MODE = "score_desc"

# True: retained 前缀向上补齐到 block size 的整数倍
EVAL_LOWRANK_QK_BLOCK_ALIGNMENT_ENABLED = True

# block 对齐大小，通常与 GPTQ block size 对齐
EVAL_LOWRANK_QK_BLOCK_ALIGNMENT_SIZE = 128

# 对齐方式: "ceil" = 向上补齐
EVAL_LOWRANK_QK_BLOCK_ALIGNMENT_ROUNDING = "ceil"

# --- D. score calibration 数据来源 ---
# score 统计使用的 calibration 样本数
EVAL_LOWRANK_QK_SCORE_CALIBRATION_NSAMPLES = 1024

# calibration source:
# - "wikitext2"
# - "wikitext2_commonsense_mmlu"  (兼容历史命名，字符串里仍保留 commonsense)
# - "cola_json"
EVAL_LOWRANK_QK_SCORE_CALIBRATION_SOURCE = "cola_json"#"wikitext2"#"wikitext2_commonsense_mmlu"

# 当 source == "cola_json" 时，这个 path 才会真正生效
EVAL_LOWRANK_QK_SCORE_CALIBRATION_PATH = "/data1/ljs/calibration_outputs/cola_mixed_0.15_0.45_0.4_QAanswer/calibration_samples.json"

# 当 source == "wikitext2_commonsense_mmlu" 时，下面三个 ratio 才用于在线混合构造 calibration
# 注意: 下面字段名里的 "COMMONSENSE" 是历史兼容命名，这里实际指 commensense 混合部分
EVAL_LOWRANK_QK_SCORE_CALIBRATION_WIKITEXT2_RATIO = 0.30
EVAL_LOWRANK_QK_SCORE_CALIBRATION_COMMONSENSE_RATIO = 0.40
EVAL_LOWRANK_QK_SCORE_CALIBRATION_MMLU_RATIO = 0.30

# commensense 混合部分从以下任务中平均抽样
EVAL_LOWRANK_QK_SCORE_CALIBRATION_COMMONSENSE_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "boolq",
    "openbookqa",
    "social_iqa",
)
# --- E. score 计算时的随机旋转 / 运行方式 / 可视化 ---
# 是否在 score 计算前，对 low-rank bottleneck basis 做 RandomSpin
EVAL_LOWRANK_QK_SCORE_RANDOMSPIN_ENABLED = True

# RandomSpin 随机种子
EVAL_LOWRANK_QK_SCORE_RANDOMSPIN_SEED = 42

# True: score 统计尽量放到 GPU 上累计
EVAL_LOWRANK_QK_SCORE_COMPUTE_ON_GPU = True

# True: 显示 score 统计进度条
EVAL_LOWRANK_QK_SCORE_SHOW_PROGRESS = True

# --- F. 作用模块范围 ---
# 指定 retained-channel mixed precision 作用于哪些 low-rank 模块
EVAL_LOWRANK_QK_TARGET_PATTERNS = (
    "self_attn.q_proj_1.module",
    "self_attn.q_proj_2.module",
    "self_attn.k_proj_1.module",
    "self_attn.k_proj_2.module",
)

# 下游 commensense 0-shot 评测任务列表。
# 这组任务是最终 lm_eval 的任务，不等于 score calibration 的配比本身。
COMMENSENSE_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "boolq",
    "openbookqa",
    "social_iqa",
    "mmlu"
]


def apply_eval_lowrank_qk_policy(ptq_args) -> None:
    ptq_args.lowrank_qk_fp16_mixed_enabled = EVAL_LOWRANK_QK_FP16_MIXED_ENABLED
    ptq_args.lowrank_qk_retained_ratio = EVAL_LOWRANK_QK_RETAINED_CHANNEL_RATIO
    ptq_args.lowrank_qk_score_mode = EVAL_LOWRANK_QK_RETAIN_SELECTION_MODE
    ptq_args.lowrank_qk_fp16_quant_bits = EVAL_LOWRANK_QK_QUANT_BITS
    ptq_args.lowrank_qk_highprec_bits = EVAL_LOWRANK_QK_HIGHPREC_BITS
    ptq_args.lowrank_qk_use_global_budget = EVAL_LOWRANK_QK_USE_GLOBAL_BUDGET
    ptq_args.lowrank_qk_channel_reorder_enabled = EVAL_LOWRANK_QK_CHANNEL_REORDER_ENABLED
    ptq_args.lowrank_qk_channel_reorder_mode = EVAL_LOWRANK_QK_CHANNEL_REORDER_MODE
    ptq_args.lowrank_qk_block_alignment_enabled = EVAL_LOWRANK_QK_BLOCK_ALIGNMENT_ENABLED
    ptq_args.lowrank_qk_block_alignment_size = EVAL_LOWRANK_QK_BLOCK_ALIGNMENT_SIZE
    ptq_args.lowrank_qk_block_alignment_rounding = EVAL_LOWRANK_QK_BLOCK_ALIGNMENT_ROUNDING
    ptq_args.lowrank_qk_score_calibration_nsamples = EVAL_LOWRANK_QK_SCORE_CALIBRATION_NSAMPLES
    ptq_args.lowrank_qk_score_calibration_source = EVAL_LOWRANK_QK_SCORE_CALIBRATION_SOURCE
    ptq_args.lowrank_qk_score_calibration_path = EVAL_LOWRANK_QK_SCORE_CALIBRATION_PATH
    ptq_args.lowrank_qk_score_calibration_wikitext2_ratio = (
        EVAL_LOWRANK_QK_SCORE_CALIBRATION_WIKITEXT2_RATIO
    )
    ptq_args.lowrank_qk_score_calibration_commonsense_ratio = (
        EVAL_LOWRANK_QK_SCORE_CALIBRATION_COMMONSENSE_RATIO
    )
    ptq_args.lowrank_qk_score_calibration_mmlu_ratio = (
        EVAL_LOWRANK_QK_SCORE_CALIBRATION_MMLU_RATIO
    )
    ptq_args.lowrank_qk_score_calibration_commonsense_tasks = (
        EVAL_LOWRANK_QK_SCORE_CALIBRATION_COMMONSENSE_TASKS
    )
    ptq_args.lowrank_qk_score_randomspin_enabled = EVAL_LOWRANK_QK_SCORE_RANDOMSPIN_ENABLED
    ptq_args.lowrank_qk_score_randomspin_seed = EVAL_LOWRANK_QK_SCORE_RANDOMSPIN_SEED
    ptq_args.lowrank_qk_score_compute_on_gpu = EVAL_LOWRANK_QK_SCORE_COMPUTE_ON_GPU
    ptq_args.lowrank_qk_score_show_progress = EVAL_LOWRANK_QK_SCORE_SHOW_PROGRESS
    ptq_args.lowrank_qk_fp16_target_patterns = EVAL_LOWRANK_QK_TARGET_PATTERNS
    ptq_args.lowrank_qk_retained_indices_map = {}


@torch.inference_mode()
def low_memory_evaluator(model, testenc, dev, args) -> float:
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // model.seqlen
    input_ids = input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen)

    batch_size = max(1, int(args.bsz))
    input_batches = [
        input_ids[i : i + batch_size].contiguous()
        for i in range(0, nsamples, batch_size)
    ]
    nbatches = len(input_batches)

    inps = [None] * nbatches
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.detach().cpu()
            cache["i"] += 1
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            cache["attention_mask"] = (
                attention_mask.detach() if attention_mask is not None else None
            )
            cache["position_ids"] = (
                position_ids.detach() if position_ids is not None else None
            )
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in input_batches:
        try:
            model(batch.to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    outs = [None] * nbatches

    for layer_idx in tqdm(range(len(layers)), desc="(PPL Eval) Layers"):
        layer = layers[layer_idx].to(dev)
        for batch_idx in range(nbatches):
            hidden_states = inps[batch_idx].to(dev, non_blocking=True)
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            outs[batch_idx] = layer_out.detach().cpu()
            del hidden_states
            del layer_out
        layers[layer_idx] = layer.cpu()
        del layer
        inps, outs = outs, [None] * nbatches
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for batch_idx in range(nbatches):
        hidden_states = inps[batch_idx].to(dev, non_blocking=True)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_batches[batch_idx][:, 1:].to(dev, non_blocking=True)
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1).cpu()
        nlls.append(neg_log_likelihood)
        del hidden_states
        del lm_logits
        del shift_logits
        del shift_labels
        del loss
        del neg_log_likelihood
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    log.info(f"\n WikiText2 PPL (low-memory): {ppl.item():.3f}")
    return ppl.item()


def evaluate_wikitext2_low_memory(
    model,
    tokenizer,
    ptq_args,
    seq_len: int = PPL_EVAL_SEQLEN,
) -> float:
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=seq_len,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    dataset_ppl = low_memory_evaluator(model, testloader, utils.DEV, ptq_args)
    log.info(
        f"wiki2 ppl (seq_len={seq_len}, batch_size={ptq_args.bsz}) is: {dataset_ppl}"
    )
    return dataset_ppl


def evaluate_lm_tasks(model, tokenizer):
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size="auto",
        device="cuda",
    )
    return lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=COMMENSENSE_TASKS,
        num_fewshot=0,
        log_samples=False,
    )


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    model_args, training_args, ptq_args = process_args_ptq()
    apply_eval_lowrank_qk_policy(ptq_args)
    local_rank = utils.get_local_rank()

    run_output_dir, timestamp = prepare_run_output_dir(
        model_args, training_args, ptq_args, local_rank
    )
    if local_rank == 0:
        log.info(f"Current LM-eval run output directory: {run_output_dir}")
        log.info(
            "Low-rank q/k retain policy for this eval: mode=%s, ratio=%.4f",
            ptq_args.lowrank_qk_score_mode,
            ptq_args.lowrank_qk_retained_ratio,
        )
        log.info(
            "Low-rank q/k mixed precision for this eval: highprec_bits=%s, global_budget=%s",
            ptq_args.lowrank_qk_highprec_bits,
            ptq_args.lowrank_qk_use_global_budget,
        )
        log.info(
            "Low-rank q/k calibration for this eval: source=%s, path=%s",
            ptq_args.lowrank_qk_score_calibration_source,
            ptq_args.lowrank_qk_score_calibration_path,
        )
        save_run_config(
            run_output_dir,
            timestamp,
            model_args,
            training_args,
            ptq_args,
        )
    dist.barrier()

    model, config, state_dict, process_word_embeddings = build_model_and_config(
        model_args,
        training_args,
        ptq_args,
    )
    repair_empty_split_biases(model, state_dict)

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # from ptq import apply_lowrank_rotation_offline
    # apply_lowrank_rotation_offline(model, config)
    maybe_apply_trained_rmid(model, config)

    model.cuda()
    model = ptq_model(ptq_args, model, model_args)
    if KEEP_QUANTIZER_STATE_AFTER_PTQ and KEEP_QUANTIZER_STATE_FOR_PPL:
        enable_quantizer_state_preservation(model)
    model.seqlen = training_args.model_max_length
    model.config.use_cache = True
    model.cuda()
    if local_rank == 0:
        save_lowrank_qk_retained_channel_table(run_output_dir, ptq_args)

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )

    if local_rank == 0:
        log.info("Starting LM Evaluation Harness (commensense 0-shot)...")
        results = evaluate_lm_tasks(model, tokenizer)
        log.info("\n" + lm_eval.utils.make_table(results))
    else:
        results = None

    model.seqlen = PPL_EVAL_SEQLEN
    ptq_args.bsz = PPL_EVAL_BATCH_SIZE
    torch.cuda.empty_cache()
    wiki2_ppl = evaluate_wikitext2_low_memory(model, tokenizer, ptq_args)
    if (
        KEEP_QUANTIZER_STATE_AFTER_PTQ
        and KEEP_QUANTIZER_STATE_FOR_PPL
        and PRINT_QUANTIZER_SHAPES_AFTER_EVAL
    ):
        log_quantizer_shapes(model)

    if local_rank == 0:
        output_payload = {
            "metadata": {
                "model_path": model_args.input_model,
                "svd_ckpt_path": ptq_args.svd_llm_ckpt,
                "w_bits": getattr(ptq_args, "w_bits", "unknown"),
                "a_bits": getattr(ptq_args, "a_bits", 16),
                "w_groupsize": getattr(ptq_args, "w_groupsize", "none"),
                "rotation": getattr(ptq_args, "rotate", False),
                "lowrank_qk_score_mode": ptq_args.lowrank_qk_score_mode,
                "lowrank_qk_retained_ratio": ptq_args.lowrank_qk_retained_ratio,
                "lowrank_qk_highprec_bits": ptq_args.lowrank_qk_highprec_bits,
                "lowrank_qk_use_global_budget": ptq_args.lowrank_qk_use_global_budget,
                "lowrank_qk_score_calibration_source": ptq_args.lowrank_qk_score_calibration_source,
                "lowrank_qk_score_calibration_path": ptq_args.lowrank_qk_score_calibration_path,
                "lowrank_qk_score_compute_on_gpu": getattr(
                    ptq_args, "lowrank_qk_score_compute_on_gpu", False
                ),
                "lowrank_qk_score_show_progress": getattr(
                    ptq_args, "lowrank_qk_score_show_progress", False
                ),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tasks": COMMENSENSE_TASKS,
                "num_fewshot": 0,
                "ppl_eval_seq_len": PPL_EVAL_SEQLEN,
                "ppl_eval_batch_size": PPL_EVAL_BATCH_SIZE,
                "keep_quantizer_state_for_ppl": KEEP_QUANTIZER_STATE_FOR_PPL,
            },
            "eval_results": results.get("results", {}),
            "lm_eval_version": results.get("versions", {}),
            "wikitext2_ppl": wiki2_ppl,
        }

        save_final_eval_results(
            run_output_dir,
            timestamp,
            output_payload,
        )

        json_name = (
            f"{os.path.basename(os.path.normpath(ptq_args.svd_llm_ckpt))}"
            f"_W{getattr(ptq_args, 'w_bits', 'unknown')}"
            f"A{getattr(ptq_args, 'a_bits', 16)}"
            f"_G{getattr(ptq_args, 'w_groupsize', 'none')}"
            "_commensense0shot_lm_eval.json"
        )
        save_path = os.path.join(run_output_dir, json_name)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        log.info(f"LM-eval results saved to {save_path}")

    dist.barrier()


if __name__ == "__main__":
    main()
