# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import os
import random

import pandas as pd
import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils_UV, hadamard_utils, quant_utils, utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)


def _pack_mmlu_example_svdllm(question, choices):
    choice_lines = "\n".join(
        f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices)
    )
    return f"Question:\n{question}\n\nChoices:\n{choice_lines}\n\nAnswer:"


def _load_local_mmlu_auxiliary_texts_svdllm():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        data_dir = data_utils.resolve_mmlu_data_dir(base_dir)
    except FileNotFoundError:
        return None

    for split in ("auxiliary_train", "dev"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue

        suffix = f"_{split}.csv"
        texts = []
        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith(suffix):
                continue
            df = pd.read_csv(os.path.join(split_dir, filename), header=None)
            for idx in range(df.shape[0]):
                question = df.iloc[idx, 0]
                choices = [df.iloc[idx, j + 1] for j in range(df.shape[1] - 2)]
                texts.append(_pack_mmlu_example_svdllm(question, choices))
        if texts:
            return texts

    return None


def _load_mmlu_auxiliary_texts_svdllm(cache_dir=None):
    local_texts = _load_local_mmlu_auxiliary_texts_svdllm()
    if local_texts is not None:
        return local_texts

    try:
        dataset = data_utils.datasets.load_dataset(
            "cais/mmlu",
            "all",
            split="auxiliary_train",
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load MMLU auxiliary data in SVD-LLM style. Either set "
            "MMLU_DATA_DIR to a local dataset root containing split folders such as "
            "dev/test, or make sure Hugging Face can access 'cais/mmlu'. "
            "Original error: {}".format(exc)
        ) from exc

    return [
        _pack_mmlu_example_svdllm(
            example.get("question", ""),
            example.get("choices", []),
        )
        for example in dataset
    ]


def get_mmlu_calibration_loader_svdllm(
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    cache_dir=None,
):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    tot_text = "\n\n".join(_load_mmlu_auxiliary_texts_svdllm(cache_dir=cache_dir))
    rng = random.Random(seed)

    trainloader = []
    while len(trainloader) < nsamples:
        i = rng.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        inp = trainenc.input_ids[:, :seqlen]
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
            trainloader = get_mmlu_calibration_loader_svdllm(
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
