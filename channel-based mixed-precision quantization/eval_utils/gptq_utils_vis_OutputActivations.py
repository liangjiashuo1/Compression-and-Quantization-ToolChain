# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from eval_utils.vis_activation import visualize_save_and_clear_cache

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
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
        # blocksize=2,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

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

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Scale1 = torch.zeros_like(W1).to(Scale.dtype)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q, int_weight, scale = self.quantizer.fake_quantize(w.unsqueeze(1))
                Q1[:, i] = q.flatten()
                q = q.flatten()
                W_int1[:, i] = int_weight.flatten()
                Scale1[:, i] = scale.flatten()

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Scale[:, i1:i2] = Scale1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", W_int.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
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

def proj_modules(attn, name):
    if getattr(attn, f"{name}_lowrank"):
        return [
            f"self_attn.{name}_proj_1.module",
            f"self_attn.{name}_proj_2.module",
        ]
    else:
        return [f"self_attn.{name}_proj.module"]

def mlp_modules(mlp, name):
    if getattr(mlp, f"{name}_lowrank"):
        return [
            f"mlp.{name}_proj_1.module",
            f"mlp.{name}_proj_2.module",
        ]
    else:
        return [f"mlp.{name}_proj.module"]

#=================================================== 原始版本的gptq_fwrd ==========================================
# @torch.no_grad()
# def gptq_fwrd(model, dataloader, dev, args):
#     """
#     From GPTQ repo
#     """
#     logging.info("-----GPTQ Quantization-----")

#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.layers

#     model.model.embed_tokens = model.model.embed_tokens.to(dev)
#     model.model.norm = model.model.norm.to(dev)
#     layers[0] = layers[0].to(dev)

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {"i": 0, "attention_mask": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#             if hasattr(module, "attention_type"):
#                 self.attention_type = module.attention_type

#         def forward(self, inp, **kwargs):
#             inps[cache["i"]] = inp
#             cache["i"] += 1
#             cache["attention_mask"] = kwargs["attention_mask"]
#             cache["position_ids"] = kwargs["position_ids"]
#             cache['position_embeddings'] = kwargs['position_embeddings']
#             raise ValueError

#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         # print(f"the shape of batch[0] is{batch[0].shape}")
#         try:
#             model(batch[0].to(dev))
#         except ValueError:
#             pass
#     layers[0] = layers[0].module

#     layers[0] = layers[0].cpu()
#     model.model.embed_tokens = model.model.embed_tokens.cpu()
#     model.model.norm = model.model.norm.cpu()
#     torch.cuda.empty_cache()

#     outs = torch.zeros_like(inps)
#     attention_mask = cache["attention_mask"]
#     position_ids = cache["position_ids"]
#     position_embeddings = cache['position_embeddings']


#     quantizers = {}
#     # attn = layers[0].self_attn
#     # mlp = layers[0].mlp
#     # print(full.keys())
#     # sequential = [
#     #     (
#     #         proj_modules(attn, "k")
#     #         + proj_modules(attn, "v")
#     #         + proj_modules(attn, "q")
#     #     ),
#     #     proj_modules(attn, "o"),
#     #     (
#     #         mlp_modules(mlp, "up")
#     #         + mlp_modules(mlp, "gate")
#     #     ),
#     #     mlp_modules(mlp, "down"),
#     # ]
#     for i in range(len(layers)):
#         print(f"\nLayer {i}:", flush=True, end=" ")
#         layer = layers[i].to(dev)

#         attn = layer.self_attn
#         mlp = layer.mlp

#         sequential = [
#             (
#                 proj_modules(attn, "k")
#                 + proj_modules(attn, "v")
#                 + proj_modules(attn, "q")
#             ),
#             proj_modules(attn, "o"),
#             (
#                 mlp_modules(mlp, "up")
#                 + mlp_modules(mlp, "gate")
#             ),
#             mlp_modules(mlp, "down"),
#         ]

#         full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
#         for names in sequential:
#             subset = {n: full[n] for n in names}

#             gptq = {}
#             for name in subset:
#                 print(f"{name}", end="  ", flush=True)
#                 layer_weight_bits = args.w_bits
#                 layer_weight_sym = not (args.w_asym)
#                 if "lm_head" in name:
#                     layer_weight_bits = 16
#                     continue
#                 if args.int8_down_proj and "down_proj" in name:
#                     layer_weight_bits = 8
#                 gptq[name] = GPTQ(subset[name])
#                 gptq[name].quantizer = quant_utils.WeightQuantizer()
#                 gptq[name].quantizer.configure(
#                     layer_weight_bits,
#                     perchannel=True,
#                     sym=layer_weight_sym,
#                     mse=args.w_clip,
#                 )

#             def add_batch(name):
#                 def tmp(_, inp, out):
#                     gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

#                 return tmp

#             handles = []
#             for name in subset:
#                 handles.append(subset[name].register_forward_hook(add_batch(name)))
#             for j in range(args.nsamples):
#                 outs[j] = layer(
#                     inps[j].unsqueeze(0),
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     # position_embeddings=position_embeddings
#                 )[0]
#             for h in handles:
#                 h.remove()

#             for name in subset:
#                 layer_w_groupsize = args.w_groupsize
#                 gptq[name].fasterquant(
#                     percdamp=args.percdamp,
#                     groupsize=layer_w_groupsize,
#                     actorder=args.act_order,
#                     static_groups=False,
#                     export_to_et=args.export_to_et,
#                 )
#                 quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
#                 gptq[name].free()

#         for j in range(args.nsamples):
#             outs[j] = layer(
#                 inps[j].unsqueeze(0),
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 # position_embeddings=position_embeddings
#             )[0]

#         layers[i] = layer.cpu()
#         del layer
#         del gptq
#         torch.cuda.empty_cache()

#         inps, outs = outs, inps

#     model.config.use_cache = use_cache
#     utils.cleanup_memory(verbos=True)
#     logging.info("-----GPTQ Quantization Done-----\n")
#     return quantizers
#=====================================================================================================


#=================================================修改后的版本⬇️（收集量化前的激活值）===============================================
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    """
    From GPTQ repo
    """
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
            cache['position_embeddings'] = kwargs['position_embeddings']
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
    position_embeddings = cache['position_embeddings']

    quantizers = {}

    ### ====== 新增：可视化配置字典 ====== ###
    activation_cache = {}
    # 这里设置你想观察的层数，防止每层都画图导致太慢或图片太多
    # 比如观察最浅层(0)、中间层(15)、深层(31)的误差累积情况
    # target_vis_layers = [0, 10, 15, 31] 
    target_vis_layers = list(range(32))
    ### ================================== ###

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)

        attn = layer.self_attn
        mlp = layer.mlp

        sequential = [
            (
                proj_modules(attn, "k")
                + proj_modules(attn, "v")
                + proj_modules(attn, "q")
            ),
            proj_modules(attn, "o"),
            (
                mlp_modules(mlp, "up")
                + mlp_modules(mlp, "gate")
            ),
            mlp_modules(mlp, "down"),
        ]

        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

            ### ================= 收集输入激活值 =============== ###
            # def add_batch(name, layer_idx):
            #     def tmp(_, inp, out):
            #         # GPTQ 原本的逻辑：计算 Hessian 矩阵
            #         gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

            #         # 拦截激活值：如果当前层在我们的观察列表中
            #         if layer_idx in target_vis_layers:
            #             cache_key = f"model.layers.{layer_idx}.{name}"
            #             # 为了画图不爆内存，我们只抓取每个子模块的第 1 个 Batch 的数据
            #             if cache_key not in activation_cache:
            #                 activation_cache[cache_key] = inp[0].data.detach().cpu()
            #     return tmp
            ### ================================================ ###

            ### ====================== 收集输出激活值 ====================== ###
            def add_batch(name, layer_idx):
                def tmp(_, inp, out):
                    # GPTQ 原本的逻辑：计算 Hessian 矩阵 (⚠️ 必须保留 inp[0]，千万别改)
                    gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    # 拦截输出激活值：如果当前层在我们的观察列表中
                    if layer_idx in target_vis_layers:
                        cache_key = f"model.layers.{layer_idx}.{name}"
                        if cache_key not in activation_cache:
                            # 👇 修改点在这里 👇
                            # 对于标准的 nn.Linear，out 直接就是一个 Tensor
                            # 为了代码极其稳健，防止某些特殊模型架构返回元组，可以加个判断：
                            if isinstance(out, tuple):
                                activation_cache[cache_key] = out[0].data.detach().cpu()
                            else:
                                activation_cache[cache_key] = out.data.detach().cpu()
                return tmp
            ### ================================================ ###

            handles = []
            for name in subset:
                # 传入当前的层数 i，以便 Hook 内部判断
                handles.append(subset[name].register_forward_hook(add_batch(name, i)))
                
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    # position_embeddings=position_embeddings
                )[0]
                
            for h in handles:
                h.remove()

            ### ====== 修改：在获取完数据后，计算统计信息、写入txt并画图 ====== ###
            if i in target_vis_layers:
                # 以追加模式 (a) 打开 txt 文件，如果不存在会自动创建
                stats_file_path = "activation_stats.txt"
                with open(stats_file_path, "a", encoding="utf-8") as f:
                    # 如果是第一次写入这一层，可以打个分割线
                    f.write(f"\n{'='*20} Processing Layer {i} {'='*20}\n")
                    
                    for name in subset:
                        cache_key = f"model.layers.{i}.{name}"
                        if cache_key in activation_cache:
                            # 1. 提取张量并转为浮点型以保证计算精度
                            act_tensor = activation_cache[cache_key].float()
                            
                            # 2. 计算统计指标
                            var_val = act_tensor.var().item()  # 方差
                            max_abs_val = act_tensor.abs().max().item()  # 最大绝对值
                            
                            # 计算峰度 Kurtosis: E[(X - mu)^4] / (sigma^2)^2
                            # 加上 1e-9 防止方差为 0 导致除零错误
                            mean_val = act_tensor.mean()
                            kurtosis_val = ((act_tensor - mean_val)**4).mean().item() / (var_val**2 + 1e-9)
                            
                            # 3. 格式化日志字符串
                            log_str = (f"Module: {cache_key:<40} | "
                                       f"Variance: {var_val:>10.4f} | "
                                       f"Kurtosis: {kurtosis_val:>10.4f} | "
                                       f"Max_Abs: {max_abs_val:>10.4f}\n")
                            
                            # 4. 打印到控制台并写入 txt 文件
                            print(f"  -> Stats: {log_str.strip()}")
                            f.write(log_str)
                            
                            # 5. 调用画图并清空缓存
                            # visualize_save_and_clear_cache(activation_cache, cache_key)

            ### ==========================仅对激活值进行可视化，没有统计方差等数据⬇️================================== ###
            # if i in target_vis_layers:
            #     for name in subset:
            #         cache_key = f"model.layers.{i}.{name}"
            #         if cache_key in activation_cache:
            #             visualize_save_and_clear_cache(activation_cache, cache_key)
            ### ============================================================================================== ###

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        # 使用量化后的当前层，生成输入给下一层的真实、带噪的激活值！
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                # position_embeddings=position_embeddings
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers, activation_cache

#=================================================修改后的版本⬆️===============================================

@torch.no_grad()
def rtn_fwrd(model, dev, args, custom_layers=None):
    """
    From GPTQ repo
    """
    # assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
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
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
            if args.export_to_et:
                layer_weight_bits = 8  # all per channel 8 bits for executorch export
                w_groupsize = -1
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
