# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import json
import os
import random

import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils_UV, hadamard_utils, quant_utils, utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)


def _resolve_cola_calibration_path(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("COLA_CALIBRATION_PATH")
    if env_path:
        candidates.append(env_path)

    env_output_dir = os.environ.get("COLA_OUTPUT_DIR")
    if env_output_dir:
        candidates.append(os.path.join(env_output_dir, "calibration_samples.json"))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend(
        [
            os.path.join(base_dir, "..", "COLA", "cola_output", "calibration_samples.json"),
            os.path.join(base_dir, "..", "COLA_MMLU", "cola_mmlu_output", "calibration_samples.json"),
            os.path.join(base_dir, "cola_output", "calibration_samples.json"),
        ]
    )

    checked = []
    for path in candidates:
        norm_path = os.path.abspath(path)
        checked.append(norm_path)
        if os.path.isfile(norm_path):
            return norm_path

    raise FileNotFoundError(
        "COLA calibration_samples.json not found. Set COLA_CALIBRATION_PATH or "
        "COLA_OUTPUT_DIR, or place the file at one of: {}".format(", ".join(checked))
    )


def _pad_or_trim_input_ids(inp: torch.Tensor, seqlen: int, pad_token_id: int) -> torch.Tensor:
    if inp.dim() == 1:
        inp = inp.unsqueeze(0)
    elif inp.dim() != 2:
        raise ValueError(f"Expected input_ids to be 1D or 2D, got shape {tuple(inp.shape)}")

    if inp.shape[1] > seqlen:
        return inp[:, :seqlen]
    if inp.shape[1] == seqlen:
        return inp

    pad_width = seqlen - inp.shape[1]
    pad = torch.full((inp.shape[0], pad_width), pad_token_id, dtype=inp.dtype)
    return torch.cat([inp, pad], dim=1)


def get_cola_calibration_loader_svdllm(
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    cache_dir=None,
    calibration_path=None,
):
    del cache_dir  # Unused, kept for interface compatibility.

    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    resolved_path = _resolve_cola_calibration_path(calibration_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        raise ValueError(f"No calibration samples found in {resolved_path}")

    rng = random.Random(seed)
    ordered_indices = list(range(len(samples)))
    rng.shuffle(ordered_indices)
    ordered_samples = [samples[idx] for idx in ordered_indices]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    trainloader = []
    for sample_idx in range(nsamples):
        sample = ordered_samples[sample_idx % len(ordered_samples)]
        if "input_ids" in sample and sample["input_ids"] is not None:
            inp = torch.tensor(sample["input_ids"], dtype=torch.long)
        elif "text" in sample:
            inp = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seqlen,
            ).input_ids
        else:
            raise KeyError(
                "Each COLA calibration sample must contain either 'input_ids' or 'text'."
            )

        inp = _pad_or_trim_input_ids(inp, seqlen=seqlen, pad_token_id=pad_token_id)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:
            trainloader = get_cola_calibration_loader_svdllm(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                cache_dir=getattr(model_args, "cache_dir", None),
            )
            if args.export_to_et:
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and "v_proj_1" not in name and args.v_bits < 16:
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name and "o_proj_2" not in name:
                layer_groupsize = head_dim

            if "lm_head" in name:
                layer_input_bits = 16

            if "down_proj" in name and "down_proj_2" not in name:
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    return model
