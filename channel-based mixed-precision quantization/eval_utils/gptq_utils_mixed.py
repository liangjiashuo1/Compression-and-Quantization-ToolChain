# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import logging
import math
import pprint
import time

import torch
import torch.nn as nn
import tqdm

from utils import quant_utils, utils


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        w = layer.weight.data.clone()
        self.rows = w.shape[0]
        self.columns = w.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
    ):
        w = self.layer.weight.data.clone().float()
        scale_store = self.layer.weight.data.clone().float()
        w_int = self.layer.weight.data.clone().float()

        if not self.quantizer.ready():
            self.quantizer.find_params(w)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        w[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(w[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            w = w[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        losses = torch.zeros_like(w)
        q_out = torch.zeros_like(w)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            w1 = w[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            w_int1 = torch.zeros_like(w1)
            scale1 = torch.zeros_like(w1).to(scale_store.dtype)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = w1[:, i]
                d = Hinv1[i, i]
                original_col_idx = i1 + i

                if groupsize != -1:
                    if not static_groups:
                        if original_col_idx % groupsize == 0:
                            self.quantizer.find_params(
                                w[:, original_col_idx : (original_col_idx + groupsize)]
                            )
                    else:
                        idx = original_col_idx
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                effective_col_idx = original_col_idx
                if actorder:
                    effective_col_idx = int(perm[original_col_idx].item())
                retained_col_indices = getattr(
                    self.quantizer, "retained_col_indices", torch.zeros(0, dtype=torch.long)
                )
                preserve_full_column = (
                    retained_col_indices.numel() > 0
                    and torch.any(
                        retained_col_indices.to(self.dev) == effective_col_idx
                    ).item()
                )

                if preserve_full_column:
                    q, int_weight, scale = self.quantizer.fake_quantize(
                        col.unsqueeze(1), use_highprec_for_all=True
                    )
                else:
                    q, int_weight, scale = self.quantizer.fake_quantize(col.unsqueeze(1))
                q1[:, i] = q.flatten()
                q = q.flatten()
                w_int1[:, i] = int_weight.flatten()
                scale1[:, i] = scale.flatten()

                losses1[:, i] = (col - q) ** 2 / d**2

                local_err = (col - q) / d
                w1[:, i:] -= local_err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                err1[:, i] = local_err

            q_out[:, i1:i2] = q1
            w_int[:, i1:i2] = w_int1
            scale_store[:, i1:i2] = scale1
            losses[:, i1:i2] = losses1 / 2

            w[:, i2:] -= err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            q_out = q_out[:, invperm]

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", w_int.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", scale_store)
        self.layer.weight.data = q_out.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")
            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


def layer_uses_lowrank_qk_fp16_tail(name, args) -> bool:
    if not getattr(args, "lowrank_qk_fp16_mixed_enabled", False):
        return False

    target_patterns = getattr(
        args,
        "lowrank_qk_fp16_target_patterns",
        ("self_attn.q_proj_1.module", "self_attn.k_proj_1.module"),
    )
    return any(name.endswith(pattern) for pattern in target_patterns)


def resolve_retained_index_spec(args, name, layer_idx=None):
    retained_indices_map = getattr(args, "lowrank_qk_retained_indices_map", {})
    if not retained_indices_map:
        return None

    # The score-preparation path stores full module names such as
    # `model.layers.0.self_attn.q_proj_1.module`, while the local GPTQ loop
    # iterates with short names such as `self_attn.q_proj_1.module`.
    if layer_idx is not None:
        full_name = f"model.layers.{layer_idx}.{name}"
        if full_name in retained_indices_map:
            return retained_indices_map[full_name]

    return retained_indices_map.get(name)


def get_weight_quant_config(args, name, layer_idx=None):
    layer_weight_bits = args.w_bits
    retained_ratio = 0.0
    retained_row_indices = None
    retained_col_indices = None
    highprec_bits = None
    retention_mode = None

    if "lm_head" in name:
        return (
            getattr(args, "lm_head_w_bits", 16),
            retained_ratio,
            retained_row_indices,
            retained_col_indices,
            highprec_bits,
            retention_mode,
        )

    if args.int8_down_proj and "down_proj" in name:
        layer_weight_bits = 8

    if layer_uses_lowrank_qk_fp16_tail(name, args):
        layer_weight_bits = getattr(args, "lowrank_qk_fp16_quant_bits", layer_weight_bits)
        highprec_bits = getattr(args, "lowrank_qk_highprec_bits", None)
        retained_ratio = getattr(args, "lowrank_qk_retained_ratio", 0.0)
        retained_spec = resolve_retained_index_spec(args, name, layer_idx)
        if retained_spec is not None:
            retained_row_indices = retained_spec.get("row_indices")
            retained_col_indices = retained_spec.get("col_indices")
        retention_mode = getattr(args, "lowrank_qk_score_mode", None)
        if (
            retained_ratio > 0
            and getattr(args, "lowrank_qk_retained_indices_map", {})
            and retained_row_indices is None
            and retained_col_indices is None
        ):
            logging.warning(
                "No retained-channel indices found for %s (layer_idx=%s); "
                "falling back to default tail retention.",
                name,
                layer_idx,
            )

    return (
        layer_weight_bits,
        retained_ratio,
        retained_row_indices,
        retained_col_indices,
        highprec_bits,
        retention_mode,
    )


def proj_modules(module, name):
    if hasattr(module, "intermediate_size"):
        if hasattr(module, f"{name}_proj"):
            return [f"mlp.{name}_proj.module"]
        return [f"mlp.{name}_proj_1.module", f"mlp.{name}_proj_2.module"]

    if hasattr(module, f"{name}_proj"):
        return [f"self_attn.{name}_proj.module"]
    return [f"self_attn.{name}_proj_1.module", f"self_attn.{name}_proj_2.module"]


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    logging.info("-----GPTQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quantizers = {}
    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        sequential = [
            (
                proj_modules(layer.self_attn, "k")
                + proj_modules(layer.self_attn, "v")
                + proj_modules(layer.self_attn, "q")
            ),
            proj_modules(layer.self_attn, "o"),
            (
                proj_modules(layer.mlp, "up")
                + proj_modules(layer.mlp, "gate")
            ),
            proj_modules(layer.mlp, "down"),
        ]
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_sym = not (args.w_asym)
                (
                    layer_weight_bits,
                    retained_ratio,
                    retained_row_indices,
                    retained_col_indices,
                    highprec_bits,
                    retention_mode,
                ) = get_weight_quant_config(args, name, layer_idx=i)
                if layer_weight_bits == 16:
                    continue

                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                    retained_ratio=retained_ratio,
                    retained_row_indices=retained_row_indices,
                    retained_col_indices=retained_col_indices,
                    highprec_bits=highprec_bits,
                    retention_mode=retention_mode,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for handle in handles:
                handle.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args, custom_layers=None):
    if custom_layers:
        layers = custom_layers
    else:
        layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, torch.nn.Embedding]
        )

        for name in subset:
            w_groupsize = args.w_groupsize
            (
                layer_weight_bits,
                retained_ratio,
                retained_row_indices,
                retained_col_indices,
                highprec_bits,
                retention_mode,
            ) = get_weight_quant_config(args, name, layer_idx=i)
            if layer_weight_bits == 16:
                continue
            if args.export_to_et:
                layer_weight_bits = 8
                w_groupsize = -1
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
                retained_ratio=retained_ratio,
                retained_row_indices=retained_row_indices,
                retained_col_indices=retained_col_indices,
                highprec_bits=highprec_bits,
                retention_mode=retention_mode,
            )
            w = subset[name].weight.data
            quantizer.find_params(w)
            q, int_weight, scale = quantizer.fake_quantize(w)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)
            quantizers[f"model.layers.{i}.{name}"] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
