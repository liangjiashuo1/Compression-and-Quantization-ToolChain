# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import transformers

from train_utils.quant_linear import QuantizeLinear
from train_utils.quant_linear_split import QuantizeLinear as Q2
from utils import hadamard_utils
from utils.utils import HadamardTransform


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16
        self.keep_last_state = False

    def free(self) -> None:
        if self.keep_last_state:
            return
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x) -> None:
        init_shape = x.shape
        # print(f'x.shape is {x.shape}')
        # print(f' self.groupsize is {self.groupsize}') # 128 
        # self.groupsize = 2
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        # if R1 is not None:
        if R1 is not None or R2 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.register_buffer("highprec_maxq", torch.tensor(0))
        self.register_buffer("highprec_scale", torch.zeros(shape))
        self.register_buffer("highprec_zero", torch.zeros(shape))
        self.register_buffer("retained_row_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("retained_col_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("retained_indices", torch.zeros(0, dtype=torch.long))
        self.retained_ratio = 0.0
        self.retained_count = 0
        self.retained_row_count = 0
        self.retained_col_count = 0
        self.retention_mode = None
        self.highprec_bits = None
        self.use_highprec_channel_quant = False
        self.mixed_precision = False

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
        retained_ratio: float = 0.0,
        retained_row_indices=None,
        retained_col_indices=None,
        highprec_bits: int = None,
        retention_mode: str = None,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        self.retained_ratio = retained_ratio
        self.retention_mode = retention_mode
        self.highprec_bits = highprec_bits
        self._set_retained_indices(
            retained_row_indices=retained_row_indices,
            retained_col_indices=retained_col_indices,
        )
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def _normalize_retained_indices(self, indices) -> torch.Tensor:
        if indices is None:
            return torch.zeros(0, dtype=torch.long)
        normalized = torch.as_tensor(indices, dtype=torch.long).flatten()
        if normalized.numel() > 0:
            normalized = torch.sort(torch.unique(normalized.cpu())).values
        else:
            normalized = torch.zeros(0, dtype=torch.long)
        return normalized

    def _refresh_retained_metadata(self) -> None:
        self.retained_row_count = int(self.retained_row_indices.numel())
        self.retained_col_count = int(self.retained_col_indices.numel())
        self.retained_count = max(self.retained_row_count, self.retained_col_count)
        if self.retained_row_count > 0:
            self.retained_indices = self.retained_row_indices.clone()
        else:
            self.retained_indices = self.retained_col_indices.clone()
        self.mixed_precision = (self.retained_row_count > 0) or (
            self.retained_col_count > 0
        )
        self.use_highprec_channel_quant = (
            self.mixed_precision
            and self.highprec_bits is not None
            and self.highprec_bits > self.bits
            and self.highprec_bits < 16
        )

    def _set_retained_indices(self, retained_row_indices=None, retained_col_indices=None) -> None:
        self.retained_row_indices = self._normalize_retained_indices(retained_row_indices)
        self.retained_col_indices = self._normalize_retained_indices(retained_col_indices)
        self._refresh_retained_metadata()

    def _maybe_build_default_retained_indices(self, x) -> None:
        if self.mixed_precision or self.retained_ratio <= 0:
            self._refresh_retained_metadata()
            return

        rows = x.shape[0] if x.ndim > 0 else 0
        if rows <= 0:
            self.retained_row_indices = torch.zeros(0, dtype=torch.long)
            self.retained_col_indices = torch.zeros(0, dtype=torch.long)
            self._refresh_retained_metadata()
            return

        retained_count = min(rows, max(1, math.ceil(rows * self.retained_ratio)))
        start = max(0, rows - retained_count)
        self.retained_row_indices = torch.arange(start, rows, dtype=torch.long)
        self.retained_col_indices = torch.zeros(0, dtype=torch.long)
        self._refresh_retained_metadata()

    def _compute_quant_triplet(self, x, scale, zero, maxq):
        scale = scale.to(x.device)
        maxq = maxq.to(x.device)
        if self.sym:
            q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
            fake_q = scale * q
        else:
            zero = zero.to(x.device)
            q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
            fake_q = scale * (q - zero)
        return fake_q, q, scale

    def _build_highprec_params(self, original_x) -> None:
        if not self.use_highprec_channel_quant:
            self.highprec_scale = torch.zeros_like(self.scale)
            self.highprec_zero = torch.zeros_like(self.zero)
            self.highprec_maxq = torch.tensor(0, device=original_x.device)
            return

        temp_quantizer = WeightQuantizer()
        temp_quantizer.configure(
            self.highprec_bits,
            perchannel=self.perchannel,
            sym=self.sym,
            mse=self.mse,
            norm=self.norm,
            grid=self.grid,
            maxshrink=self.maxshrink,
            weight_groupsize=self.weight_groupsize,
        )
        temp_quantizer.find_params(original_x)
        self.highprec_scale = temp_quantizer.scale.detach().clone()
        self.highprec_zero = temp_quantizer.zero.detach().clone()
        self.highprec_maxq = temp_quantizer.maxq.detach().clone()

    def _get_highprec_quant_triplet(self, x):
        if not self.use_highprec_channel_quant:
            return None, None, None
        return self._compute_quant_triplet(
            x,
            self.highprec_scale,
            self.highprec_zero,
            self.highprec_maxq,
        )

    def _apply_retained_channel_mixed_precision(
        self,
        quantized_x,
        int_weight=None,
        scale=None,
        highprec_quantized_x=None,
        highprec_int_weight=None,
        highprec_scale=None,
    ):
        if not self.mixed_precision:
            return quantized_x, int_weight, scale

        if self.retained_row_indices.numel() > 0:
            retained_row_indices = self.retained_row_indices.to(quantized_x.device)
            if highprec_quantized_x is not None:
                quantized_x[retained_row_indices] = highprec_quantized_x[
                    retained_row_indices
                ].to(quantized_x.dtype)
            if int_weight is not None and highprec_int_weight is not None:
                int_weight[retained_row_indices] = highprec_int_weight[
                    retained_row_indices
                ].to(int_weight.dtype)
            if scale is not None and highprec_scale is not None:
                scale[retained_row_indices] = highprec_scale[retained_row_indices].to(
                    scale.dtype
                )

        if self.retained_col_indices.numel() > 0 and quantized_x.ndim >= 2:
            retained_col_indices = self.retained_col_indices.to(quantized_x.device)
            retained_col_indices = retained_col_indices[
                retained_col_indices < quantized_x.shape[1]
            ]
            if retained_col_indices.numel() == 0:
                return quantized_x, int_weight, scale
            if highprec_quantized_x is not None:
                quantized_x[:, retained_col_indices] = highprec_quantized_x[
                    :, retained_col_indices
                ].to(quantized_x.dtype)
            if int_weight is not None and highprec_int_weight is not None:
                int_weight[:, retained_col_indices] = highprec_int_weight[
                    :, retained_col_indices
                ].to(int_weight.dtype)
            if scale is not None and highprec_scale is not None:
                scale[:, retained_col_indices] = highprec_scale[
                    :, retained_col_indices
                ].to(scale.dtype)

        return quantized_x, int_weight, scale

    def find_params_weight_groupwise(self, x) -> None:
        self._maybe_build_default_retained_indices(x)
        original_x = x
        init_shape = x.shape
        x = x.reshape(
            x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize
        )

        xmax = torch.amax(x, dim=-1, keepdim=True)
        xmin = torch.amin(x, dim=-1, keepdim=True)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, self.weight_groupsize)
        self.zero = self.zero.repeat(1, 1, self.weight_groupsize)

        if self.mse:
            best = torch.full(
                [x.shape[0], x.shape[1]], float("inf"), device=x.device
            ).type_as(x)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = sym_quant_dequant(x, scale1, self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = asym_quant_dequant(x, scale1, zero1, self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, -1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(init_shape)
        self.zero = self.zero.reshape(init_shape)
        self._build_highprec_params(original_x)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)
        self._maybe_build_default_retained_indices(x)
        original_x = x

        shape = x.shape

        if self.weight_groupsize > 0:
            # group-wise per-token quantization
            self.find_params_weight_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        self._build_highprec_params(original_x)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            base_quantized_x, _, base_scale = self._compute_quant_triplet(
                x, self.scale, self.zero, self.maxq
            )
            highprec_quantized_x, _, highprec_scale = self._get_highprec_quant_triplet(x)
            quantized_x, _, _ = self._apply_retained_channel_mixed_precision(
                base_quantized_x.to(x_dtype),
                None,
                base_scale,
                highprec_quantized_x=(
                    highprec_quantized_x.to(x_dtype)
                    if highprec_quantized_x is not None
                    else None
                ),
                highprec_scale=highprec_scale,
            )
            return quantized_x
        return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x, use_highprec_for_all: bool = False):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if use_highprec_for_all and self.use_highprec_channel_quant:
                highprec_fake_q, highprec_q, highprec_scale = self._get_highprec_quant_triplet(x)
                return highprec_fake_q.to(x_dtype), highprec_q, highprec_scale

            fake_q, q, scale = self._compute_quant_triplet(
                x, self.scale, self.zero, self.maxq
            )
            highprec_fake_q, highprec_q, highprec_scale = self._get_highprec_quant_triplet(x)
            fake_q, q, scale = self._apply_retained_channel_mixed_precision(
                fake_q.to(x_dtype),
                q,
                scale,
                highprec_quantized_x=(
                    highprec_fake_q.to(x_dtype)
                    if highprec_fake_q is not None
                    else None
                ),
                highprec_int_weight=highprec_q,
                highprec_scale=highprec_scale,
            )
            return fake_q, q, scale
        else:
            return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        Q2,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
) -> None:
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) is torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) is torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
