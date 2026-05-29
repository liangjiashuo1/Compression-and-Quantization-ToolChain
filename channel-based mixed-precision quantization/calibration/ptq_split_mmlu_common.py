# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import traceback
from logging import Logger

import torch
import transformers
from transformers import LlamaTokenizerFast

from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils


log: Logger = utils.get_logger("spinquant")


def svd_llm_standandized_to_ptq_structure(st):
    rename_map = {
        "q_proj.ALinear": "q_proj_2",
        "q_proj.BLinear": "q_proj_1",
        "k_proj.ALinear": "k_proj_2",
        "k_proj.BLinear": "k_proj_1",
        "v_proj.ALinear": "v_proj_2",
        "v_proj.BLinear": "v_proj_1",
        "o_proj.ALinear": "o_proj_2",
        "o_proj.BLinear": "o_proj_1",
        "gate_proj.ALinear": "gate_proj_2",
        "gate_proj.BLinear": "gate_proj_1",
        "up_proj.ALinear": "up_proj_2",
        "up_proj.BLinear": "up_proj_1",
        "down_proj.ALinear": "down_proj_2",
        "down_proj.BLinear": "down_proj_1",
    }

    new_st = {}
    for key, value in st.items():
        new_key = key
        for old, new in rename_map.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
                break
        new_st[new_key] = value
    return new_st


def get_rank_and_flag(st, key_prefix):
    lowrank_key = f"{key_prefix}_1.weight"
    full_key = f"{key_prefix}.weight"

    if lowrank_key in st:
        return st[lowrank_key].shape[0], True
    if full_key in st:
        return None, False
    raise KeyError(f"{key_prefix} not found in state_dict")


def _configure_lowrank_metadata(config, st):
    config.q_rank = [None] * config.num_hidden_layers
    config.k_rank = [None] * config.num_hidden_layers
    config.v_rank = [None] * config.num_hidden_layers
    config.o_rank = [None] * config.num_hidden_layers
    config.gate_rank = [None] * config.num_hidden_layers
    config.up_rank = [None] * config.num_hidden_layers
    config.down_rank = [None] * config.num_hidden_layers

    config.q_lowrank = [False] * config.num_hidden_layers
    config.k_lowrank = [False] * config.num_hidden_layers
    config.v_lowrank = [False] * config.num_hidden_layers
    config.o_lowrank = [False] * config.num_hidden_layers
    config.gate_lowrank = [False] * config.num_hidden_layers
    config.up_lowrank = [False] * config.num_hidden_layers
    config.down_lowrank = [False] * config.num_hidden_layers

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"

        config.q_rank[layer_idx], config.q_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.self_attn.q_proj"
        )
        config.k_rank[layer_idx], config.k_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.self_attn.k_proj"
        )
        config.v_rank[layer_idx], config.v_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.self_attn.v_proj"
        )
        config.o_rank[layer_idx], config.o_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.self_attn.o_proj"
        )
        config.gate_rank[layer_idx], config.gate_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.mlp.gate_proj"
        )
        config.up_rank[layer_idx], config.up_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.mlp.up_proj"
        )
        config.down_rank[layer_idx], config.down_lowrank[layer_idx] = get_rank_and_flag(
            st, f"{prefix}.mlp.down_proj"
        )


def load_lowrank_model(model_args, training_args, ptq_args):
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model,
        token=model_args.access_token,
    )

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    st = torch.load(ptq_args.svd_llm_ckpt, map_location="cpu")
    st = svd_llm_standandized_to_ptq_structure(st)
    _configure_lowrank_metadata(config, st)

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        state_dict=st,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    return model


def build_tokenizer(model_args, training_args):
    return LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )


def log_gpu_mem(tag: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    log.info("[GPU MEM] %s: allocated=%.2f MB, reserved=%.2f MB", tag, alloc, reserv)


def reset_cuda_memory_stats():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _resolve_eval_log_base(ptq_args, script_path):
    if ptq_args.mmlu_log_path:
        return ptq_args.mmlu_log_path

    if ptq_args.optimized_rotation_path:
        return "{}_{}_{}_{}".format(
            ptq_args.optimized_rotation_path,
            ptq_args.w_bits,
            ptq_args.a_bits,
            ptq_args.k_bits,
        )

    script_stem = os.path.splitext(os.path.basename(script_path))[0]
    return os.path.join(
        os.path.dirname(script_path),
        "mmlu_log",
        "{}_{}_{}_{}".format(
            script_stem,
            ptq_args.w_bits,
            ptq_args.a_bits,
            ptq_args.k_bits,
        ),
    )


def _resolve_eval_log_path(base_path, suffix):
    stem, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".log"
    log_path = f"{stem}_{suffix}{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    return log_path


def run_standard_evaluations(model, tokenizer, ptq_args, script_path):
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: %s", dataset_ppl)

    eval_log_base = _resolve_eval_log_base(ptq_args, script_path)
    try:
        import test_comm

        suite_results = test_comm.evaluate_default_suites(
            model=model,
            tokenizer=tokenizer,
        )
        for suite_name, results in suite_results.items():
            formatted_results = test_comm.format_results(results)
            log.info("%s results:\n%s", suite_name, formatted_results)

            log_path = _resolve_eval_log_path(eval_log_base, suite_name)
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(formatted_results)
                handle.write("\n")
            log.info("%s log saved to %s", suite_name, log_path)
    except ImportError as exc:
        log.warning("Skip test_comm evaluation because lm_eval is unavailable: %s", exc)
    except Exception as exc:
        log.warning("test_comm evaluation failed: %s", exc)
        log.warning("test_comm traceback:\n%s", traceback.format_exc())

    return dataset_ppl
