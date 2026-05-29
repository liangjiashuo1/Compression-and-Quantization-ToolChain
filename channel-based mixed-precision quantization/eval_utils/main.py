# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers

from eval_utils.vis_activation import visualize_save_and_clear_cache
from eval_utils.channel_selection import (
    apply_lowrank_qk_channel_reorder_inplace,
    compute_lowrank_qk_retained_indices,
    maybe_apply_lowrank_qk_randomspin_inplace,
)
from eval_utils import gptq_utils, rotation_utils, gptq_utils_mixed, gptq_utils_vis_InputActivations, gptq_utils_vis_OutputActivations
from utils import data_utils, fuse_norm_utils_UV, hadamard_utils, quant_utils, utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)


def move_weight_quantizers_to_cpu(weight_quantizers):
    return {
        key: quantizer.cpu() if hasattr(quantizer, "cpu") else quantizer
        for key, quantizer in weight_quantizers.items()
    }


def serialize_mixed_precision_weights(model, weight_quantizers):
    serialized = {}

    for key, quantizer in weight_quantizers.items():
        if not isinstance(quantizer, quant_utils.WeightQuantizer):
            continue
        if not getattr(quantizer, "mixed_precision", False):
            continue
        if getattr(quantizer, "highprec_bits", None) not in (None, 16):
            continue

        module = model.get_submodule(key)
        weight = module.weight.detach().cpu().float()
        retained_row_indices = getattr(quantizer, "retained_row_indices", None)
        retained_col_indices = getattr(quantizer, "retained_col_indices", None)
        if retained_row_indices is None and retained_col_indices is None:
            continue
        retained_row_indices = (
            retained_row_indices.detach().cpu().long().flatten()
            if retained_row_indices is not None
            else torch.zeros(0, dtype=torch.long)
        )
        retained_col_indices = (
            retained_col_indices.detach().cpu().long().flatten()
            if retained_col_indices is not None
            else torch.zeros(0, dtype=torch.long)
        )
        if retained_row_indices.numel() <= 0 and retained_col_indices.numel() <= 0:
            continue

        scale = quantizer.scale.detach().cpu().float().contiguous()
        zero = quantizer.zero.detach().cpu().float().contiguous()
        maxq = quantizer.maxq.detach().cpu()

        if getattr(quantizer, "sym", True):
            int_weight = torch.clamp(
                torch.round(weight / scale),
                -(maxq + 1),
                maxq,
            ).to(torch.int8)
            zero_to_save = None
        else:
            int_weight = torch.clamp(
                torch.round(weight / scale) + zero,
                0,
                maxq,
            ).to(torch.uint8)
            zero_to_save = zero

        serialized[key] = {
            "shape": list(weight.shape),
            "bits": getattr(quantizer, "bits", None),
            "sym": getattr(quantizer, "sym", True),
            "retained_ratio": getattr(quantizer, "retained_ratio", 0.0),
            "retention_mode": getattr(quantizer, "retention_mode", None),
            "retained_row_indices": retained_row_indices.contiguous(),
            "retained_col_indices": retained_col_indices.contiguous(),
            "int_weight": int_weight.contiguous(),
            "scale": scale.contiguous(),
        }
        if retained_row_indices.numel() > 0:
            serialized[key]["fp16_retained_rows"] = (
                weight[retained_row_indices].to(torch.float16).contiguous()
            )
        if retained_col_indices.numel() > 0:
            serialized[key]["fp16_retained_cols"] = (
                weight[:, retained_col_indices].to(torch.float16).contiguous()
            )
        if zero_to_save is not None:
            serialized[key]["zero"] = zero_to_save.contiguous()

    return serialized


def restore_mixed_precision_weights(model, serialized_mixed_weights):
    restored_count = 0

    for key, payload in serialized_mixed_weights.items():
        module = model.get_submodule(key)

        shape = tuple(payload["shape"])
        sym = bool(payload.get("sym", True))
        retained_row_indices = payload.get("retained_row_indices")
        retained_col_indices = payload.get("retained_col_indices")
        if retained_row_indices is None:
            tail_start = int(payload["tail_start"])
            tail_count = int(payload["tail_count"])
            retained_row_indices = torch.arange(
                tail_start, tail_start + tail_count, dtype=torch.long
            )
        if retained_row_indices is None:
            retained_row_indices = torch.zeros(0, dtype=torch.long)
        if retained_col_indices is None:
            retained_col_indices = torch.zeros(0, dtype=torch.long)
        retained_row_indices = retained_row_indices.to(module.weight.device).long()
        retained_col_indices = retained_col_indices.to(module.weight.device).long()

        int_weight = payload["int_weight"].to(module.weight.device).float()
        scale = payload["scale"].to(module.weight.device).float()
        if sym:
            full_weight = scale * int_weight
        else:
            zero = payload["zero"].to(module.weight.device).float()
            full_weight = scale * (int_weight - zero)

        full_weight = full_weight.reshape(shape)
        if retained_row_indices.numel() > 0:
            retained_rows = payload.get(
                "fp16_retained_rows",
                payload.get("fp16_retained_weight", payload.get("fp16_tail_weight")),
            )
            retained_rows = retained_rows.to(module.weight.device).to(full_weight.dtype)
            full_weight[retained_row_indices] = retained_rows
        if retained_col_indices.numel() > 0:
            retained_cols = payload["fp16_retained_cols"].to(module.weight.device).to(
                full_weight.dtype
            )
            full_weight[:, retained_col_indices] = retained_cols

        module.weight.data.copy_(full_weight.to(module.weight.dtype))
        restored_count += 1

    return restored_count


def maybe_prepare_lowrank_qk_retained_indices(args, model, model_args) -> None:
    if not getattr(args, "lowrank_qk_fp16_mixed_enabled", False):
        args.lowrank_qk_retained_indices_map = {}
        return
    if getattr(args, "load_qmodel_path", None):
        args.lowrank_qk_retained_indices_map = {}
        return
    if model_args is None:
        args.lowrank_qk_retained_indices_map = {}
        return

    retained_ratio = float(getattr(args, "lowrank_qk_retained_ratio", 0.0))
    score_mode = getattr(args, "lowrank_qk_score_mode", None)
    if retained_ratio <= 0 or score_mode is None:
        args.lowrank_qk_retained_indices_map = {}
        return

    retained_indices_map = compute_lowrank_qk_retained_indices(
        model=model,
        model_args=model_args,
        ptq_args=args,
        score_mode=score_mode,
        retain_ratio=retained_ratio,
    )
    args.lowrank_qk_retained_indices_map = retained_indices_map
    if retained_indices_map and getattr(args, "lowrank_qk_channel_reorder_enabled", False):
        reordered_pairs = apply_lowrank_qk_channel_reorder_inplace(
            model,
            retained_indices_map,
        )
        if reordered_pairs > 0:
            print(
                f"Applied low-rank q/k channel reorder to {reordered_pairs} projection pairs "
                f"with mode={getattr(args, 'lowrank_qk_channel_reorder_mode', 'score_desc')}."
            )
    if retained_indices_map:
        calibration_source = getattr(
            args, "lowrank_qk_score_calibration_source", "wikitext2"
        )
        summary = {
            name: {
                "row_count": int(
                    value.get("row_indices", torch.zeros(0, dtype=torch.long)).numel()
                ),
                "col_count": int(
                    value.get("col_indices", torch.zeros(0, dtype=torch.long)).numel()
                ),
                "score_row_count": int(
                    value.get(
                        "score_selected_row_indices", torch.zeros(0, dtype=torch.long)
                    ).numel()
                ),
                "score_col_count": int(
                    value.get(
                        "score_selected_col_indices", torch.zeros(0, dtype=torch.long)
                    ).numel()
                ),
                "reordered": bool(value.get("reorder_enabled", False)),
                "block_aligned": bool(value.get("block_alignment_enabled", False)),
            }
            for name, value in retained_indices_map.items()
        }
        print(
            f"Prepared low-rank q/k retained channel indices with mode={score_mode}, "
            f"ratio={retained_ratio:.4f}, calibration_source={calibration_source}: {summary}"
        )


def maybe_apply_lowrank_qk_randomspin(args, model) -> None:
    if getattr(args, "load_qmodel_path", None):
        return

    applied = maybe_apply_lowrank_qk_randomspin_inplace(model, args)
    if applied:
        applied_pairs = getattr(model, "_lowrank_qk_randomspin_pair_count", 0)
        randomspin_seed = int(
            getattr(args, "lowrank_qk_score_randomspin_seed", 0)
        )
        print(
            f"Applied in-memory RandomSpin to {applied_pairs} low-rank q/k pairs "
            f"before score computation and GPTQ (seed={randomspin_seed})."
        )

#===================================================第一种：原版的普通量化========================================================
def get_gptq_calibration_loader(args, model_args):
    calibration_source = getattr(
        args,
        "lowrank_qk_score_calibration_source",
        "wikitext2",
    )
    calibration_path = getattr(
        args,
        "lowrank_qk_score_calibration_path",
        None,
    )
    trainloader = data_utils.get_channel_selection_calibration_loader(
        calibration_source=calibration_source,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=2048,
        model=model_args.input_model,
        cache_dir=getattr(model_args, "cache_dir", None),
        wikitext2_ratio=getattr(
            args,
            "lowrank_qk_score_calibration_wikitext2_ratio",
            0.34,
        ),
        commonsense_ratio=getattr(
            args,
            "lowrank_qk_score_calibration_commonsense_ratio",
            0.33,
        ),
        mmlu_ratio=getattr(
            args,
            "lowrank_qk_score_calibration_mmlu_ratio",
            0.33,
        ),
        commonsense_tasks=getattr(
            args,
            "lowrank_qk_score_calibration_commonsense_tasks",
            data_utils.COMMONSENSE_CALIBRATION_TASKS,
        ),
        calibration_path=calibration_path,
    )
    print(
        "Using GPTQ calibration loader aligned with score calibration: "
        f"source={calibration_source}, path={calibration_path}, nsamples={args.nsamples}."
    )
    return trainloader


def quantize_named_module_rtn(
    module_name,
    module,
    dev,
    args,
    weight_quantizers,
    bits_attr_name="lm_head_w_bits",
    groupsize_attr_name="lm_head_w_groupsize",
    bits=None,
    groupsize=None,
):
    target_bits = getattr(args, bits_attr_name, 16) if bits is None else bits
    if target_bits >= 16:
        return

    target_groupsize = groupsize
    if target_groupsize is None:
        target_groupsize = getattr(args, groupsize_attr_name, -1)
    if target_groupsize == 0:
        target_groupsize = -1
    elif target_groupsize < 0:
        target_groupsize = getattr(args, "w_groupsize", -1)

    print(
        f"Quantizing {module_name} with RTN weight quantization: "
        f"bits={target_bits}, groupsize={target_groupsize}.",
        flush=True,
    )
    module = module.to(dev)
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        target_bits,
        perchannel=True,
        sym=not getattr(args, "w_asym", False),
        mse=getattr(args, "w_clip", False),
        weight_groupsize=target_groupsize,
    )
    weight = module.weight.data
    quantizer.find_params(weight)
    qweight, int_weight, scale = quantizer.fake_quantize(weight)
    module.weight.data = qweight.to(module.weight.data.dtype)

    if getattr(args, "export_to_et", False):
        if "int_weight" in module._buffers:
            module._buffers["int_weight"] = int_weight
        else:
            module.register_buffer("int_weight", int_weight)
        if "scale" in module._buffers:
            module._buffers["scale"] = scale
        else:
            module.register_buffer("scale", scale)

    weight_quantizers[module_name] = quantizer.cpu()
    module.cpu()
    torch.cuda.empty_cache()


def maybe_quantize_lm_head(args, model, model_args, weight_quantizers, dev="cuda"):
    lm_head_bits = getattr(args, "lm_head_w_bits", 16)
    if lm_head_bits >= 16:
        return
    lm_head_groupsize = getattr(args, "lm_head_w_groupsize", -1)
    if lm_head_groupsize == 0:
        lm_head_groupsize = -1
    elif lm_head_groupsize < 0:
        lm_head_groupsize = getattr(args, "w_groupsize", -1)

    print(
        f"Quantizing lm_head with GPTQ weight quantization: "
        f"bits={lm_head_bits}, groupsize={lm_head_groupsize}.",
        flush=True,
    )

    trainloader = get_gptq_calibration_loader(args, model_args)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    model = model.to(dev)
    gptq = gptq_utils.GPTQ(model.lm_head)
    gptq.quantizer = quant_utils.WeightQuantizer()
    gptq.quantizer.configure(
        lm_head_bits,
        perchannel=True,
        sym=not getattr(args, "w_asym", False),
        mse=getattr(args, "w_clip", False),
    )

    def add_batch(_, inp, out):
        gptq.add_batch(inp[0].data, out.data)

    handle = model.lm_head.register_forward_hook(add_batch)
    try:
        for batch in trainloader:
            model(batch[0].to(dev))
    finally:
        handle.remove()
        model.config.use_cache = use_cache

    gptq.fasterquant(
        percdamp=args.percdamp,
        groupsize=lm_head_groupsize,
        actorder=args.act_order,
        static_groups=False,
        export_to_et=args.export_to_et,
    )
    weight_quantizers["lm_head"] = gptq.quantizer.cpu()
    gptq.free()
    model.lm_head.cpu()
    torch.cuda.empty_cache()


def maybe_quantize_embed_tokens(args, model, weight_quantizers, dev="cuda"):
    embed_bits = getattr(args, "embed_w_bits", 16)
    if embed_bits >= 16:
        return
    quantize_named_module_rtn(
        "model.embed_tokens",
        model.model.embed_tokens,
        dev,
        args,
        weight_quantizers,
        bits_attr_name="embed_w_bits",
        groupsize_attr_name="embed_w_groupsize",
        bits=embed_bits,
    )


def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    weight_quantizers = {}

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model) # fix
        rotation_utils.rotate_model(model, args) # fix
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name: #fix
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    maybe_apply_lowrank_qk_randomspin(args, model)
    maybe_prepare_lowrank_qk_retained_indices(args, model, model_args)

    if (
        args.w_bits < 16
        or getattr(args, "lm_head_w_bits", 16) < 16
        or getattr(args, "embed_w_bits", 16) < 16
    ):
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            restored_mixed_count = restore_mixed_precision_weights(
                model,
                save_dict.get("mixed_precision_weights", {}),
            )
            if restored_mixed_count:
                print(f"Restored {restored_mixed_count} mixed-precision low-rank Q/K weights from serialized checkpoint.")
            weight_quantizers = save_dict.get("w_quantizers", {})

        else:
            if args.w_bits < 16:
                if not args.w_rtn:  # GPTQ Weight Quantization
                    trainloader = get_gptq_calibration_loader(args, model_args)
                    quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
                    weight_quantizers.update(quantizers)
                else:  # RTN Weight Quantization
                    quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
                    weight_quantizers.update(quantizers)

            if args.export_to_et and getattr(args, "embed_w_bits", 16) >= 16:
                quantize_named_module_rtn(
                    "model.embed_tokens",
                    model.model.embed_tokens,
                    "cuda",
                    args,
                    weight_quantizers,
                    bits_attr_name="embed_w_bits",
                    groupsize_attr_name="embed_w_groupsize",
                    bits=8,
                    groupsize=-1,
                )
            maybe_quantize_embed_tokens(args, model, weight_quantizers, dev="cuda")
            maybe_quantize_lm_head(args, model, model_args, weight_quantizers, dev="cuda")
            save_dict["w_quantizers"] = weight_quantizers

        if args.save_qmodel_path:
            quantizers_to_save = move_weight_quantizers_to_cpu(weight_quantizers)
            save_dict["model"] = model.state_dict()
            save_dict["w_quantizers"] = quantizers_to_save
            save_dict["mixed_precision_weights"] = serialize_mixed_precision_weights(
                model,
                quantizers_to_save,
            )
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    model.weight_quantizers = move_weight_quantizers_to_cpu(weight_quantizers)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0: # -1
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            # print(f"args.a_groupsize {args.a_groupsize}") # -1
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads # 32
            model_dim = model.config.hidden_size # 4096
            head_dim = model_dim // num_heads # 128

            if "v_proj" in name and "v_proj_1" not in name and args.v_bits < 16:  # Set the v_proj precision   #fix (in,so not _2)
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name and "o_proj_2" not in name: # fix same
                layer_groupsize = head_dim # 128

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name and "down_proj_2" not in name:  # Set the down_proj precision # fix same
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize # -1 

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
#============================================================================================================================




#===================================================第二种：混合精度量化========================================================
# 具体决定哪个层怎么量化，可能需要去 gptq_utils_mixed.py 里进行修改
def ptq_model_mixed(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    weight_quantizers = {}

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model) # fix
        rotation_utils.rotate_model(model, args) # fix
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name: #fix
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    maybe_apply_lowrank_qk_randomspin(args, model)
    maybe_prepare_lowrank_qk_retained_indices(args, model, model_args)

    if (
        args.w_bits < 16
        or getattr(args, "lm_head_w_bits", 16) < 16
        or getattr(args, "embed_w_bits", 16) < 16
    ):
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            restored_mixed_count = restore_mixed_precision_weights(
                model,
                save_dict.get("mixed_precision_weights", {}),
            )
            if restored_mixed_count:
                print(f"Restored {restored_mixed_count} mixed-precision low-rank Q/K weights from serialized checkpoint.")
            weight_quantizers = save_dict.get("w_quantizers", {})

        else:
            if args.w_bits < 16:
                if not args.w_rtn:  # GPTQ Weight Quantization
                    trainloader = get_gptq_calibration_loader(args, model_args)
                    quantizers = gptq_utils_mixed.gptq_fwrd(model, trainloader, "cuda", args)
                    weight_quantizers.update(quantizers)
                else:  # RTN Weight Quantization
                    quantizers = gptq_utils_mixed.rtn_fwrd(model, "cuda", args)
                    weight_quantizers.update(quantizers)

            if args.export_to_et and getattr(args, "embed_w_bits", 16) >= 16:
                quantize_named_module_rtn(
                    "model.embed_tokens",
                    model.model.embed_tokens,
                    "cuda",
                    args,
                    weight_quantizers,
                    bits_attr_name="embed_w_bits",
                    groupsize_attr_name="embed_w_groupsize",
                    bits=8,
                    groupsize=-1,
                )
            maybe_quantize_embed_tokens(args, model, weight_quantizers, dev="cuda")
            maybe_quantize_lm_head(args, model, model_args, weight_quantizers, dev="cuda")
            save_dict["w_quantizers"] = weight_quantizers

        if args.save_qmodel_path:
            quantizers_to_save = move_weight_quantizers_to_cpu(weight_quantizers)
            save_dict["model"] = model.state_dict()
            save_dict["w_quantizers"] = quantizers_to_save
            save_dict["mixed_precision_weights"] = serialize_mixed_precision_weights(
                model,
                quantizers_to_save,
            )
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    model.weight_quantizers = move_weight_quantizers_to_cpu(weight_quantizers)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0: # -1
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            # print(f"args.a_groupsize {args.a_groupsize}") # -1
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads # 32
            model_dim = model.config.hidden_size # 4096
            head_dim = model_dim // num_heads # 128

            if "v_proj" in name and "v_proj_1" not in name and args.v_bits < 16:  # Set the v_proj precision   #fix (in,so not _2)
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name and "o_proj_2" not in name: # fix same
                layer_groupsize = head_dim # 128

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name and "down_proj_2" not in name:  # Set the down_proj precision # fix same
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize # -1 

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

#===================================================第三种：量化前收集所有FP16的激活值========================================================
# 没有侵入到 gptq_fwrd 中
def ptq_model_vis_activations(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model) # fix
        rotation_utils.rotate_model(model, args) # fix
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name: #fix
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = get_gptq_calibration_loader(args, model_args)

            # ==========================================
            # ⬇️⬇️⬇️ 将这整块 Hook 代码粘贴到这里 ⬇️⬇️⬇️
            # ==========================================
            print(">>> 开始收集量化前激活值...")
            activation_cache = {}
            
            # 定义 Hook 函数
            def get_activation(name):
                def hook(module, hook_input):
                    if name not in activation_cache:
                        activation_cache[name] = hook_input[0].detach().cpu()
                return hook

            # 找到需要 Hook 的层（复用你代码里的工具函数）
            qlayers_for_hook = quant_utils.find_qlayers(model)
            hooks = []
            
            # 注册 Hooks（建议过滤一下层数防止爆内存，比如只看第0层和第31层）
            for name, layer in qlayers_for_hook.items():
                    hooks.append(layer.register_forward_pre_hook(get_activation(name)))

            model = model.to("cuda")
            # 取出一个 Batch 的真实校准数据进行前向传播，触发 Hooks
            # 直接强制分配到 GPU，与后续的 gptq_fwrd 保持一致
            dummy_input = trainloader[0][0].to("cuda") 
            
            with torch.no_grad():
                # 加上 input_ids= 关键字传参，对 Hugging Face Llama 模型更安全
                model(input_ids=dummy_input)

            # 收集完毕，务必移除 Hooks，以免影响后续 GPTQ 流程
            for h in hooks:
                h.remove()
                
            print(f">>> 激活值收集完成！已缓存以下层的输入：\n{list(activation_cache.keys())}")
            # 现在你可以在这里打断点（breakpoint()），或者打印 activation_cache 里的张量来分析异常值了
            all_layer_names = list(activation_cache.keys())
            
            # 2. 遍历每一个层，挨个丢给画图函数
            for layer_name in all_layer_names:
                visualize_save_and_clear_cache(activation_cache, layer_name)

            # ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️ ❤️

            # ==========================================
            # ⬆️⬆️⬆️ Hook 代码到此结束 ⬆️⬆️⬆️
            # ==========================================


            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd( # fix
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0: # -1
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            # print(f"args.a_groupsize {args.a_groupsize}") # -1
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads # 32
            model_dim = model.config.hidden_size # 4096
            head_dim = model_dim // num_heads # 128

            if "v_proj" in name and "v_proj_1" not in name and args.v_bits < 16:  # Set the v_proj precision   #fix (in,so not _2)
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name and "o_proj_2" not in name: # fix same
                layer_groupsize = head_dim # 128

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name and "down_proj_2" not in name:  # Set the down_proj precision # fix same
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize # -1 

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

#===================================================第四种：量化一层，收集一层激活值========================================================
# 需要侵入到 gptq_fwrd 中，所以创建了一个 gptq_utils_vis.py 文件进行修改

def ptq_model_vis_activations_layer_by_layer(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model) # fix
        rotation_utils.rotate_model(model, args) # fix
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name: #fix
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = get_gptq_calibration_loader(args, model_args)
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd( # fix
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils_vis_InputActivations.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0: # -1
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            # print(f"args.a_groupsize {args.a_groupsize}") # -1
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads # 32
            model_dim = model.config.hidden_size # 4096
            head_dim = model_dim // num_heads # 128

            if "v_proj" in name and "v_proj_1" not in name and args.v_bits < 16:  # Set the v_proj precision   #fix (in,so not _2)
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name and "o_proj_2" not in name: # fix same
                layer_groupsize = head_dim # 128

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name and "down_proj_2" not in name:  # Set the down_proj precision # fix same
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize # -1 

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

#===================================================第五种：直接收集激活值，返回到上一层画图========================================================
def collect_activations_FP16(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    
    activation_cache = {}

    # 1. 旋转权重 (保持保留，因为旋转后的激活值分布会改变，这是我们想观察的)
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model) # fix
        rotation_utils.rotate_model(model, args) # fix
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name: #fix
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) 

    # 2. 准备校准数据用于前向传播
    trainloader = get_gptq_calibration_loader(args, model_args)

    # 3. 挂载 Hook 收集激活值
    print(">>> 开始收集前向传播激活值 (跳过量化流程)...")
    
    def get_activation(name):
        def hook(module, hook_input):
            if name not in activation_cache:
                # 移到 CPU 防止 GPU OOM
                activation_cache[name] = hook_input[0].detach().cpu()
        return hook

    qlayers_for_hook = quant_utils.find_qlayers(model)
    hooks = []
    
    # ⚠️ 安全限制：只收集前 4 层。如需收集全部，将这段改为简单的 for 循环即可
    MAX_HOOK_LAYERS = 4 
    count = 0
    for name, layer in qlayers_for_hook.items():
        if count >= MAX_HOOK_LAYERS:
            break
        hooks.append(layer.register_forward_pre_hook(get_activation(name)))
        count += 1

    # 4. 执行一次前向传播触发 Hook
    model = model.to("cuda")
    dummy_input = trainloader[0][0].to("cuda") 
    
    with torch.no_grad():
        model(input_ids=dummy_input)

    # 5. 卸载 Hooks 并清理
    for h in hooks:
        h.remove()
        
    print(f">>> 激活值收集完成！共收集了 {len(activation_cache)} 层的输入。")

    return model, activation_cache

#=================================================== 第六种：逐层量化，收集激活值 ========================================================


def collect_OutputActivations_quant(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    gptq_layer_activations = {}
    # Rotate the weights
    if args.rotate:
        fuse_norm_utils_UV.fuse_layer_norms(model) # fix
        rotation_utils.rotate_model(model, args) # fix
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name and "down_proj_2" not in name: #fix
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = get_gptq_calibration_loader(args, model_args)
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd( # fix
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers, gptq_layer_activations = gptq_utils_vis_OutputActivations.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0: # -1
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            # print(f"args.a_groupsize {args.a_groupsize}") # -1
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads # 32
            model_dim = model.config.hidden_size # 4096
            head_dim = model_dim // num_heads # 128

            if "v_proj" in name and "v_proj_1" not in name and args.v_bits < 16:  # Set the v_proj precision   #fix (in,so not _2)
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name and "o_proj_2" not in name: # fix same
                layer_groupsize = head_dim # 128

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name and "down_proj_2" not in name:  # Set the down_proj precision # fix same
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize # -1 

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

    return model, gptq_layer_activations



#=================================================== 第七种：只 R1 R2 旋转并返回模型========================================================
def ptq_model_Rot_only(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    if args.rotate:
        # 融合 LayerNorm 到权重中
        fuse_norm_utils_UV.fuse_layer_norms(model) 
        # 执行权重旋转操作
        rotation_utils.rotate_model(model, args) 
        utils.cleanup_memory(verbos=True)

        # 添加 Activation Wrapper，Hadamard 变换依赖于这个外壳
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
        # 即使不旋转，也套上 Wrapper，保证下游代码在前向传播时不会报错
        quant_utils.add_actquant(model)

    return model
