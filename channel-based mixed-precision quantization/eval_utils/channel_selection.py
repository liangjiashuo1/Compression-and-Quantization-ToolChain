# coding=utf-8

import math
from typing import Dict, Optional

import torch
from tqdm import tqdm

from utils import data_utils, quant_utils
from utils import utils


MAX_SCORE_TOKENS_PER_MODULE = None
SCORE_CALIBRATION_NSAMPLES = 256
STORE_SCORE_ACTIVATIONS_AS_FLOAT16 = True
ATTENTION_SCORE_RANK_CHUNK = 128
SUPPORTED_SCORE_MODES = {
    "activation_aware",
    "attention_aware",
    "proj1_propagated_activation_aware",
    "logits_aware",
}


def sync_rotary_buffers_to_device(model, device: torch.device) -> None:
    for module in model.modules():
        if hasattr(module, "inv_freq"):
            inv_freq = getattr(module, "inv_freq")
            if isinstance(inv_freq, torch.Tensor) and inv_freq.device != device:
                persistent = "inv_freq" not in getattr(
                    module, "_non_persistent_buffers_set", set()
                )
                module.register_buffer(
                    "inv_freq", inv_freq.to(device), persistent=persistent
                )
        if hasattr(module, "original_inv_freq"):
            original_inv_freq = getattr(module, "original_inv_freq")
            if (
                isinstance(original_inv_freq, torch.Tensor)
                and original_inv_freq.device != device
            ):
                module.original_inv_freq = original_inv_freq.to(device)


def get_underlying_weight_module(module):
    return getattr(module, "module", module)


def quantize_weight_for_score(w1: torch.Tensor, ptq_args) -> torch.Tensor:
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        ptq_args.w_bits,
        perchannel=True,
        sym=not ptq_args.w_asym,
        mse=ptq_args.w_clip,
        weight_groupsize=ptq_args.w_groupsize,
    )

    w1 = w1.float()
    if ptq_args.w_bits < 16:
        quantizer.find_params(w1)
        return quantizer.quantize(w1).float()
    return w1.clone()


def infer_attention_shape(config, q_out_dim: int, k_out_dim: int):
    num_q_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(
        getattr(config, "num_key_value_heads", num_q_heads) or num_q_heads
    )
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)

    if num_q_heads > 0 and hidden_size > 0:
        head_dim = hidden_size // num_q_heads
    elif num_q_heads > 0:
        head_dim = q_out_dim // num_q_heads
    else:
        num_q_heads = 1
        head_dim = q_out_dim

    if head_dim <= 0 or q_out_dim % head_dim != 0 or k_out_dim % head_dim != 0:
        return 1, 1, min(q_out_dim, k_out_dim)

    return q_out_dim // head_dim, k_out_dim // head_dim, head_dim


def compute_lowrank_output(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    chunk_tokens: int = 1024,
    keep_on_device: bool = False,
) -> torch.Tensor:
    x = x.float()
    w1 = w1.float()
    w2 = w2.float()
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_tokens):
            x_chunk = x[start : start + chunk_tokens]
            z = x_chunk.matmul(w1.t())
            y = z.matmul(w2.t())
            outputs.append(y if keep_on_device else y.cpu())
    return torch.cat(outputs, dim=0)


def absorb_random_spin_into_lowrank_pair(
    w1: torch.Tensor,
    w2: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = int(w1.shape[0])
    compute_device = w1.device
    if w1.device != w2.device:
        compute_device = w1.device if w1.is_cuda else w2.device

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_mat = torch.randn(rank, rank, generator=generator, dtype=torch.float32)
    q_mat, _ = torch.linalg.qr(random_mat)
    rotation = q_mat.to(device=compute_device, dtype=torch.float32)
    w1_work = w1.to(device=compute_device, dtype=torch.float32)
    w2_work = w2.to(device=compute_device, dtype=torch.float32)
    w1_rot = rotation.t().matmul(w1_work)
    w2_rot = w2_work.matmul(rotation)
    return w1_rot, w2_rot


@torch.no_grad()
def maybe_apply_lowrank_qk_randomspin_inplace(model, ptq_args) -> bool:
    randomspin_enabled = bool(
        getattr(ptq_args, "lowrank_qk_score_randomspin_enabled", False)
    )
    if not randomspin_enabled:
        return False
    if getattr(model, "_lowrank_qk_randomspin_applied", False):
        return False
    if getattr(ptq_args, "load_qmodel_path", None):
        return False

    randomspin_seed = int(getattr(ptq_args, "lowrank_qk_score_randomspin_seed", 0))
    applied_pairs = 0

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj_name, seed_offset in (("q", 0), ("k", 1)):
            if not getattr(attn, f"{proj_name}_lowrank", False):
                continue

            proj1 = get_underlying_weight_module(getattr(attn, f"{proj_name}_proj_1"))
            proj2 = get_underlying_weight_module(getattr(attn, f"{proj_name}_proj_2"))
            w1 = proj1.weight.detach()
            w2 = proj2.weight.detach()
            rotated_w1, rotated_w2 = absorb_random_spin_into_lowrank_pair(
                w1,
                w2,
                seed=randomspin_seed + layer_idx * 2 + seed_offset,
            )
            proj1.weight.data.copy_(
                rotated_w1.to(device=proj1.weight.device, dtype=proj1.weight.dtype)
            )
            proj2.weight.data.copy_(
                rotated_w2.to(device=proj2.weight.device, dtype=proj2.weight.dtype)
            )
            applied_pairs += 1

    model._lowrank_qk_randomspin_applied = True
    model._lowrank_qk_randomspin_pair_count = applied_pairs
    return applied_pairs > 0


def compute_attention_opposite_energy(
    opposite_output: torch.Tensor,
    target_w2: torch.Tensor,
    target_kind: str,
    config,
    keep_on_device: bool = False,
) -> torch.Tensor:
    opposite_output = opposite_output.float()
    target_w2 = target_w2.float()

    target_out_dim = target_w2.shape[0]
    opposite_out_dim = opposite_output.shape[1]

    if target_kind == "q":
        q_out_dim = target_out_dim
        k_out_dim = opposite_out_dim
    elif target_kind == "k":
        q_out_dim = opposite_out_dim
        k_out_dim = target_out_dim
    else:
        raise ValueError(f"Unsupported target_kind={target_kind!r}")

    num_q_heads, num_kv_heads, head_dim = infer_attention_shape(
        config, q_out_dim, k_out_dim
    )
    rank = target_w2.shape[1]
    energies = torch.zeros(rank, dtype=torch.float32, device=target_w2.device)

    if num_q_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        for start in range(0, rank, ATTENTION_SCORE_RANK_CHUNK):
            end = min(rank, start + ATTENTION_SCORE_RANK_CHUNK)
            proj = opposite_output.matmul(target_w2[:, start:end])
            energies[start:end] = proj.pow(2).sum(dim=0)
        return energies

    if target_kind == "q":
        group_size = max(1, num_q_heads // max(1, num_kv_heads))
        if num_q_heads % num_kv_heads != 0:
            for start in range(0, rank, ATTENTION_SCORE_RANK_CHUNK):
                end = min(rank, start + ATTENTION_SCORE_RANK_CHUNK)
                shared_dim = min(opposite_output.shape[1], target_w2.shape[0])
                proj = opposite_output[:, :shared_dim].matmul(
                    target_w2[:shared_dim, start:end]
                )
                energies[start:end] = proj.pow(2).sum(dim=0)
            return energies / max(float(head_dim), 1.0)

        k_heads = opposite_output.reshape(opposite_output.shape[0], num_kv_heads, head_dim)
        k_for_q_heads = k_heads.repeat_interleave(group_size, dim=1)
        w2_heads = target_w2.reshape(num_q_heads, head_dim, rank)

        for start in range(0, rank, ATTENTION_SCORE_RANK_CHUNK):
            end = min(rank, start + ATTENTION_SCORE_RANK_CHUNK)
            proj = torch.einsum(
                "thd,hdr->thr", k_for_q_heads, w2_heads[:, :, start:end]
            )
            energies[start:end] = proj.pow(2).sum(dim=(0, 1))
    else:
        group_size = max(1, num_q_heads // max(1, num_kv_heads))
        if num_q_heads % num_kv_heads != 0:
            for start in range(0, rank, ATTENTION_SCORE_RANK_CHUNK):
                end = min(rank, start + ATTENTION_SCORE_RANK_CHUNK)
                shared_dim = min(opposite_output.shape[1], target_w2.shape[0])
                proj = opposite_output[:, :shared_dim].matmul(
                    target_w2[:shared_dim, start:end]
                )
                energies[start:end] = proj.pow(2).sum(dim=0)
            return energies / max(float(head_dim), 1.0)

        q_heads = opposite_output.reshape(opposite_output.shape[0], num_q_heads, head_dim)
        q_grouped = q_heads.reshape(
            opposite_output.shape[0], num_kv_heads, group_size, head_dim
        )
        w2_heads = target_w2.reshape(num_kv_heads, head_dim, rank)

        for start in range(0, rank, ATTENTION_SCORE_RANK_CHUNK):
            end = min(rank, start + ATTENTION_SCORE_RANK_CHUNK)
            proj = torch.einsum(
                "tkgd,kdr->tkgr", q_grouped, w2_heads[:, :, start:end]
            )
            energies[start:end] = proj.pow(2).sum(dim=(0, 1, 2))

    energies = energies / max(float(head_dim), 1.0)
    return energies if keep_on_device else energies.cpu()


def compute_logits_aware_channel_score(
    activation_error: torch.Tensor,
    opposite_output: torch.Tensor,
    target_w2: torch.Tensor,
    target_kind: str,
    config,
    rank_chunk: int = ATTENTION_SCORE_RANK_CHUNK,
    keep_on_device: bool = False,
) -> torch.Tensor:
    activation_error = activation_error.float()
    opposite_output = opposite_output.float()
    target_w2 = target_w2.float()

    target_out_dim = target_w2.shape[0]
    opposite_out_dim = opposite_output.shape[1]

    if target_kind == "q":
        q_out_dim = target_out_dim
        k_out_dim = opposite_out_dim
    elif target_kind == "k":
        q_out_dim = opposite_out_dim
        k_out_dim = target_out_dim
    else:
        raise ValueError(f"Unsupported target_kind={target_kind!r}")

    num_q_heads, num_kv_heads, head_dim = infer_attention_shape(
        config, q_out_dim, k_out_dim
    )
    rank = target_w2.shape[1]
    scores = torch.zeros(rank, dtype=torch.float32, device=target_w2.device)

    if num_q_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        for start in range(0, rank, rank_chunk):
            end = min(rank, start + rank_chunk)
            proj = opposite_output.matmul(target_w2[:, start:end]) # [T, T_o, c]
            opposite_energy = proj.pow(2).sum(dim=1)
            dH = activation_error[:, start:end].pow(2)
            scores[start:end] = (dH * opposite_energy).sum(dim=0)
        return scores

    if target_kind == "q":
        group_size = max(1, num_q_heads // max(1, num_kv_heads))
        if num_q_heads % num_kv_heads != 0:
            for start in range(0, rank, rank_chunk):
                end = min(rank, start + rank_chunk)
                shared_dim = min(opposite_output.shape[1], target_w2.shape[0])
                proj = opposite_output[:, :shared_dim].matmul(
                    target_w2[:shared_dim, start:end]
                )
                opposite_energy = proj.pow(2).sum(dim=1)
                dH = activation_error[:, start:end].pow(2)
                scores[start:end] = (dH * opposite_energy).sum(dim=0)
            return scores / max(float(head_dim), 1.0)

        k_heads = opposite_output.reshape(opposite_output.shape[0], num_kv_heads, head_dim)
        k_for_q_heads = k_heads.repeat_interleave(group_size, dim=1)
        w2_heads = target_w2.reshape(num_q_heads, head_dim, rank)

        for start in range(0, rank, rank_chunk):
            end = min(rank, start + rank_chunk)
            proj = torch.einsum(
                "thd,hdr->thr", k_for_q_heads, w2_heads[:, :, start:end]
            )
            opposite_energy = proj.pow(2).sum(dim=1)
            dH = activation_error[:, start:end].pow(2)
            scores[start:end] = (dH * opposite_energy).sum(dim=0)

    else:
        group_size = max(1, num_q_heads // max(1, num_kv_heads))
        if num_q_heads % num_kv_heads != 0:
            for start in range(0, rank, rank_chunk):
                end = min(rank, start + rank_chunk)
                shared_dim = min(opposite_output.shape[1], target_w2.shape[0])
                proj = opposite_output[:, :shared_dim].matmul(
                    target_w2[:shared_dim, start:end]
                )
                opposite_energy = proj.pow(2).sum(dim=1)
                dH = activation_error[:, start:end].pow(2)
                scores[start:end] = (dH * opposite_energy).sum(dim=0)
            return scores / max(float(head_dim), 1.0)

        q_heads = opposite_output.reshape(opposite_output.shape[0], num_q_heads, head_dim)
        q_grouped = q_heads.reshape(
            opposite_output.shape[0], num_kv_heads, group_size, head_dim
        )
        w2_heads = target_w2.reshape(num_kv_heads, head_dim, rank)

        for start in range(0, rank, rank_chunk):
            end = min(rank, start + rank_chunk)
            proj = torch.einsum(
                "tkgd,kdr->tkgr", q_grouped, w2_heads[:, :, start:end]
            )
            opposite_energy = proj.pow(2).sum(dim=(1, 2))
            dH = activation_error[:, start:end].pow(2)
            scores[start:end] = (dH * opposite_energy).sum(dim=0)

    scores = scores / max(float(head_dim), 1.0)
    return scores if keep_on_device else scores.cpu()


def compute_score_curves_for_pair(
    w1: torch.Tensor,
    w2: torch.Tensor,
    activations: torch.Tensor,
    ptq_args,
    attention_opposite_energy: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    w1 = w1.float()
    w2 = w2.float()
    x = activations.float()

    q_w1 = quantize_weight_for_score(w1, ptq_args)
    delta_w1 = w1 - q_w1

    activation_error = x.matmul(delta_w1.t())
    activation_aware_tensor = activation_error.pow(2).sum(dim=0).cpu()

    curves = {
        "activation_aware": activation_aware_tensor,
    }

    if attention_opposite_energy is not None:
        attention_energy = attention_opposite_energy.float().cpu()
        curves["attention_aware"] = activation_aware_tensor * attention_energy

    return curves


def topk_channel_indices(score: torch.Tensor, ratio: float) -> torch.Tensor:
    n = int(score.numel())
    if n <= 0:
        return torch.zeros(0, dtype=torch.long)
    k = min(n, max(1, math.ceil(n * ratio)))
    indices = torch.topk(score, k=k, largest=True).indices.cpu().long()
    return torch.sort(indices).values


def build_descending_score_permutation(score: torch.Tensor) -> torch.Tensor:
    if score.numel() <= 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.argsort(score.float(), descending=True).cpu().long()


def remap_indices_after_permutation(
    original_indices: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = permutation.cpu().long().flatten()
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(
        permutation.numel(), dtype=torch.long
    )

    original_indices = torch.as_tensor(original_indices, dtype=torch.long).flatten()
    if original_indices.numel() <= 0:
        return torch.zeros(0, dtype=torch.long), inverse_permutation

    remapped = torch.sort(inverse_permutation[original_indices]).values
    return remapped, inverse_permutation


def align_prefix_indices_to_block(
    selected_count: int,
    total_count: int,
    block_size: int,
    rounding_mode: str,
) -> torch.Tensor:
    if selected_count <= 0:
        return torch.zeros(0, dtype=torch.long)

    if block_size <= 0 or rounding_mode in (None, "none"):
        aligned_count = selected_count
    elif rounding_mode == "ceil":
        aligned_count = int(math.ceil(selected_count / block_size) * block_size)
    else:
        raise ValueError(
            f"Unsupported low-rank q/k block-alignment rounding mode: {rounding_mode!r}."
        )

    aligned_count = min(total_count, aligned_count)
    return torch.arange(aligned_count, dtype=torch.long)


def build_score_module_map(model) -> Dict[int, torch.nn.Module]:
    module_map = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if getattr(attn, "q_lowrank", False):
            module_map[layer_idx] = attn.q_proj_1
            continue
        if getattr(attn, "k_lowrank", False):
            module_map[layer_idx] = attn.k_proj_1
    return module_map


def apply_lowrank_qk_channel_reorder_inplace(
    model,
    retained_indices_map: Dict[str, Dict[str, torch.Tensor]],
) -> int:
    if not retained_indices_map:
        return 0
    if getattr(model, "_lowrank_qk_channel_reorder_applied", False):
        return int(getattr(model, "_lowrank_qk_channel_reorder_pair_count", 0))

    def reorder_lowrank_pair(proj1_module, proj2_module, permutation: torch.Tensor) -> None:
        proj1 = get_underlying_weight_module(proj1_module)
        proj2 = get_underlying_weight_module(proj2_module)

        proj1_perm = permutation.to(proj1.weight.device)
        proj2_perm = permutation.to(proj2.weight.device)

        proj1.weight.data.copy_(proj1.weight.data.index_select(0, proj1_perm))
        if proj1.bias is not None:
            proj1.bias.data.copy_(proj1.bias.data.index_select(0, proj1_perm))
        proj2.weight.data.copy_(proj2.weight.data.index_select(1, proj2_perm))

    applied_pairs = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for kind in ("q", "k"):
            if not getattr(attn, f"{kind}_lowrank", False):
                continue

            proj1_key = f"model.layers.{layer_idx}.self_attn.{kind}_proj_1.module"
            proj2_key = f"model.layers.{layer_idx}.self_attn.{kind}_proj_2.module"
            proj1_spec = retained_indices_map.get(proj1_key, {})
            proj2_spec = retained_indices_map.get(proj2_key, {})

            permutation = proj1_spec.get("row_reorder_perm")
            if permutation is None:
                permutation = proj2_spec.get("col_reorder_perm")
            if permutation is None:
                continue

            permutation = torch.as_tensor(permutation, dtype=torch.long).flatten()
            if permutation.numel() <= 0:
                continue

            reorder_lowrank_pair(
                getattr(attn, f"{kind}_proj_1"),
                getattr(attn, f"{kind}_proj_2"),
                permutation,
            )
            applied_pairs += 1

    model._lowrank_qk_channel_reorder_applied = applied_pairs > 0
    model._lowrank_qk_channel_reorder_pair_count = applied_pairs
    return applied_pairs


def compute_lowrank_qk_retained_indices(
    model,
    model_args,
    ptq_args,
    score_mode: str,
    retain_ratio: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(
            f"Unsupported low-rank q/k score mode: {score_mode}. "
            f"Expected one of {sorted(SUPPORTED_SCORE_MODES)}."
        )

    module_map = build_score_module_map(model)
    if not module_map:
        return {}

    score_nsamples = int(
        max(
            getattr(ptq_args, "nsamples", 0),
            getattr(ptq_args, "lowrank_qk_score_calibration_nsamples", SCORE_CALIBRATION_NSAMPLES),
        )
    )
    score_compute_on_gpu = bool(
        getattr(ptq_args, "lowrank_qk_score_compute_on_gpu", False)
    ) and torch.cuda.is_available()
    score_storage_device = (
        torch.device("cuda", torch.cuda.current_device())
        if score_compute_on_gpu
        else torch.device("cpu")
    )
    randomspin_enabled = bool(
        getattr(ptq_args, "lowrank_qk_score_randomspin_enabled", False)
    ) and not getattr(model, "_lowrank_qk_randomspin_applied", False)
    randomspin_seed = int(
        getattr(ptq_args, "lowrank_qk_score_randomspin_seed", 0)
    )

    layer_specs = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        layer_spec = {}
        if getattr(attn, "q_lowrank", False):
            q_w1 = attn.q_proj_1.weight.detach().float().to(score_storage_device)
            q_w2 = attn.q_proj_2.weight.detach().float().to(score_storage_device)
            if randomspin_enabled:
                q_w1, q_w2 = absorb_random_spin_into_lowrank_pair(
                    q_w1,
                    q_w2,
                    seed=randomspin_seed + layer_idx * 2,
                )
            q_qw1 = quantize_weight_for_score(q_w1, ptq_args)
            layer_spec["q"] = {
                "w1": q_w1,
                "w2": q_w2,
                "delta_w1": q_w1 - q_qw1,
                "activation_sum": torch.zeros(q_w1.shape[0], dtype=torch.float32, device=score_storage_device),
                "attention_energy": torch.zeros(q_w1.shape[0], dtype=torch.float32, device=score_storage_device),
                "proj2_col_energy": q_w2.pow(2).sum(dim=0),
                "logits_sum": torch.zeros(q_w1.shape[0], dtype=torch.float32, device=score_storage_device),
            }
        if getattr(attn, "k_lowrank", False):
            k_w1 = attn.k_proj_1.weight.detach().float().to(score_storage_device)
            k_w2 = attn.k_proj_2.weight.detach().float().to(score_storage_device)
            if randomspin_enabled:
                k_w1, k_w2 = absorb_random_spin_into_lowrank_pair(
                    k_w1,
                    k_w2,
                    seed=randomspin_seed + layer_idx * 2 + 1,
                )
            k_qw1 = quantize_weight_for_score(k_w1, ptq_args)
            layer_spec["k"] = {
                "w1": k_w1,
                "w2": k_w2,
                "delta_w1": k_w1 - k_qw1,
                "activation_sum": torch.zeros(k_w1.shape[0], dtype=torch.float32, device=score_storage_device),
                "attention_energy": torch.zeros(k_w1.shape[0], dtype=torch.float32, device=score_storage_device),
                "proj2_col_energy": k_w2.pow(2).sum(dim=0),
                "logits_sum": torch.zeros(k_w1.shape[0], dtype=torch.float32, device=score_storage_device),
            }
        if layer_spec:
            layer_specs[layer_idx] = layer_spec

    captured_inputs: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook(_, hook_input):
            x = hook_input[0].detach().reshape(-1, hook_input[0].shape[-1])
            if not score_compute_on_gpu:
                x = x.cpu()
            x = x.to(torch.float16) if STORE_SCORE_ACTIVATIONS_AS_FLOAT16 else x.float()
            captured_inputs[layer_idx] = x

        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(layer_idx))
        for layer_idx, module in module_map.items()
    ]

    try:
        trainloader = data_utils.get_channel_selection_calibration_loader(
            calibration_source=getattr(
                ptq_args,
                "lowrank_qk_score_calibration_source",
                "wikitext2",
            ),
            nsamples=score_nsamples,
            seed=ptq_args.seed,
            model=model_args.input_model,
            seqlen=2048,
            cache_dir=getattr(model_args, "cache_dir", None),
            wikitext2_ratio=getattr(
                ptq_args,
                "lowrank_qk_score_calibration_wikitext2_ratio",
                0.34,
            ),
            commonsense_ratio=getattr(
                ptq_args,
                "lowrank_qk_score_calibration_commonsense_ratio",
                0.33,
            ),
            mmlu_ratio=getattr(
                ptq_args,
                "lowrank_qk_score_calibration_mmlu_ratio",
                0.33,
            ),
            commonsense_tasks=getattr(
                ptq_args,
                "lowrank_qk_score_calibration_commonsense_tasks",
                data_utils.COMMONSENSE_CALIBRATION_TASKS,
            ),
            calibration_path=getattr(
                ptq_args,
                "lowrank_qk_score_calibration_path",
                None,
            ),
        )
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model.to(device)
        sync_rotary_buffers_to_device(model, device)
        if score_compute_on_gpu:
            print(
                f"Running low-rank q/k score accumulation on GPU: storage_device={score_storage_device}, nsamples={score_nsamples}."
            )
        show_progress = bool(
            getattr(ptq_args, "lowrank_qk_score_show_progress", False)
        )
        progress_desc = getattr(
            ptq_args,
            "lowrank_qk_score_progress_desc",
            f"score[{getattr(ptq_args, 'lowrank_qk_score_calibration_source', 'calibration')}]",
        )
        loader_iter = trainloader
        if show_progress:
            loader_iter = tqdm(
                trainloader,
                total=len(trainloader) if hasattr(trainloader, "__len__") else None,
                desc=progress_desc,
            )
        with torch.no_grad():
            for batch in loader_iter:
                captured_inputs.clear()
                model(batch[0].to(device))

                for layer_idx, layer_input in captured_inputs.items():
                    if layer_idx not in layer_specs:
                        continue
                    x = layer_input.float()
                    spec = layer_specs[layer_idx]

                    if "q" in spec:
                        q_delta = spec["q"]["delta_w1"]
                        q_activation_error = x.matmul(q_delta.t())
                        spec["q"]["activation_sum"] += q_activation_error.pow(2).sum(dim=0)

                    if "k" in spec:
                        k_delta = spec["k"]["delta_w1"]
                        k_activation_error = x.matmul(k_delta.t())
                        spec["k"]["activation_sum"] += k_activation_error.pow(2).sum(dim=0)

                    if score_mode in ("attention_aware", "logits_aware") and "q" in spec and "k" in spec:
                        q_output = compute_lowrank_output(
                            x,
                            spec["q"]["w1"],
                            spec["q"]["w2"],
                            keep_on_device=score_compute_on_gpu,
                        )
                        k_output = compute_lowrank_output(
                            x,
                            spec["k"]["w1"],
                            spec["k"]["w2"],
                            keep_on_device=score_compute_on_gpu,
                        )
                        if score_mode == "attention_aware":
                            spec["q"]["attention_energy"] += compute_attention_opposite_energy(
                                opposite_output=k_output,
                                target_w2=spec["q"]["w2"],
                                target_kind="q",
                                config=model.config,
                                keep_on_device=score_compute_on_gpu,
                            )
                            spec["k"]["attention_energy"] += compute_attention_opposite_energy(
                                opposite_output=q_output,
                                target_w2=spec["k"]["w2"],
                                target_kind="k",
                                config=model.config,
                                keep_on_device=score_compute_on_gpu,
                            )
                        elif score_mode == "logits_aware":
                            q_logits_score = compute_logits_aware_channel_score(
                                activation_error=q_activation_error,
                                opposite_output=k_output,
                                target_w2=spec["q"]["w2"],
                                target_kind="q",
                                config=model.config,
                                keep_on_device=score_compute_on_gpu,
                            )
                            spec["q"]["logits_sum"] += q_logits_score
                            
                            k_logits_score = compute_logits_aware_channel_score(
                                activation_error=k_activation_error,
                                opposite_output=q_output,
                                target_w2=spec["k"]["w2"],
                                target_kind="k",
                                config=model.config,
                                keep_on_device=score_compute_on_gpu,
                            )
                            spec["k"]["logits_sum"] += k_logits_score
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)

    score_bank = []
    for layer_idx, spec in layer_specs.items():
        if "q" in spec:
            q_score = spec["q"]["activation_sum"]
            if score_mode == "proj1_propagated_activation_aware":
                q_score = q_score * spec["q"]["proj2_col_energy"]
            if score_mode == "attention_aware" and "k" in spec:
                q_score = q_score * spec["q"]["attention_energy"]
            if score_mode == "logits_aware":
                q_score = spec["q"]["logits_sum"]
            q_score_cpu = q_score.detach().float().cpu()
            spec["q"]["final_score"] = q_score_cpu
            for local_idx, score_value in enumerate(q_score_cpu.tolist()):
                score_bank.append(("q", layer_idx, local_idx, float(score_value)))

        if "k" in spec:
            k_score = spec["k"]["activation_sum"]
            if score_mode == "proj1_propagated_activation_aware":
                k_score = k_score * spec["k"]["proj2_col_energy"]
            if score_mode == "attention_aware" and "q" in spec:
                k_score = k_score * spec["k"]["attention_energy"]
            if score_mode == "logits_aware":
                k_score = spec["k"]["logits_sum"]
            k_score_cpu = k_score.detach().float().cpu()
            spec["k"]["final_score"] = k_score_cpu
            for local_idx, score_value in enumerate(k_score_cpu.tolist()):
                score_bank.append(("k", layer_idx, local_idx, float(score_value)))

    selected_indices = {}
    use_global_budget = bool(
        getattr(ptq_args, "lowrank_qk_use_global_budget", False)
    )
    if score_bank:
        if use_global_budget:
            total_channels = len(score_bank)
            topk_count = min(total_channels, max(1, math.ceil(total_channels * retain_ratio)))
            score_tensor = torch.tensor(
                [entry[3] for entry in score_bank], dtype=torch.float32
            )
            chosen_positions = torch.topk(
                score_tensor, k=topk_count, largest=True
            ).indices.tolist()
            for pos in chosen_positions:
                kind, layer_idx, local_idx, _ = score_bank[pos]
                selected_indices.setdefault((kind, layer_idx), []).append(local_idx)
        else:
            for layer_idx, spec in layer_specs.items():
                if "q" in spec:
                    selected_indices[("q", layer_idx)] = topk_channel_indices(
                        spec["q"]["final_score"], retain_ratio
                    ).tolist()
                if "k" in spec:
                    selected_indices[("k", layer_idx)] = topk_channel_indices(
                        spec["k"]["final_score"], retain_ratio
                    ).tolist()

    channel_reorder_enabled = bool(
        getattr(ptq_args, "lowrank_qk_channel_reorder_enabled", False)
    )
    channel_reorder_mode = getattr(
        ptq_args, "lowrank_qk_channel_reorder_mode", None
    )
    if channel_reorder_enabled and channel_reorder_mode not in (None, "score_desc"):
        raise ValueError(
            f"Unsupported low-rank q/k channel reorder mode: {channel_reorder_mode!r}."
        )
    block_alignment_enabled = bool(
        getattr(ptq_args, "lowrank_qk_block_alignment_enabled", False)
    )
    block_alignment_size = int(
        getattr(ptq_args, "lowrank_qk_block_alignment_size", 128)
    )
    block_alignment_rounding = getattr(
        ptq_args, "lowrank_qk_block_alignment_rounding", "ceil"
    )
    if block_alignment_enabled and not channel_reorder_enabled:
        raise ValueError(
            "low-rank q/k block alignment requires channel reorder to be enabled, "
            "because the alignment logic assumes score-descending local channel order."
        )

    retained_index_map: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_idx, spec in layer_specs.items():
        q_key = f"model.layers.{layer_idx}.self_attn.q_proj_1.module"
        q2_key = f"model.layers.{layer_idx}.self_attn.q_proj_2.module"
        k_key = f"model.layers.{layer_idx}.self_attn.k_proj_1.module"
        k2_key = f"model.layers.{layer_idx}.self_attn.k_proj_2.module"

        if "q" in spec:
            q_score_selected_original_indices = torch.tensor(
                sorted(selected_indices.get(("q", layer_idx), [])), dtype=torch.long
            )
            q_original_indices = q_score_selected_original_indices.clone()
            q_indices = q_score_selected_original_indices.clone()
            q_perm = None
            q_invperm = None
            if channel_reorder_enabled:
                q_perm = build_descending_score_permutation(spec["q"]["final_score"])
                q_indices, q_invperm = remap_indices_after_permutation(
                    q_score_selected_original_indices,
                    q_perm,
                )
                if block_alignment_enabled:
                    q_indices = align_prefix_indices_to_block(
                        selected_count=int(q_indices.numel()),
                        total_count=int(q_perm.numel()),
                        block_size=block_alignment_size,
                        rounding_mode=block_alignment_rounding,
                    )
                    q_original_indices = torch.sort(q_perm[q_indices]).values.cpu().long()
            retained_index_map[q_key] = {
                "row_indices": q_indices,
                "original_row_indices": q_original_indices,
                "score_selected_row_indices": q_score_selected_original_indices,
                "reorder_enabled": channel_reorder_enabled,
                "reorder_mode": channel_reorder_mode or "score_desc",
                "block_alignment_enabled": block_alignment_enabled,
                "block_alignment_size": block_alignment_size,
                "block_alignment_rounding": block_alignment_rounding,
            }
            retained_index_map[q2_key] = {
                "col_indices": q_indices.clone(),
                "original_col_indices": q_original_indices.clone(),
                "score_selected_col_indices": q_score_selected_original_indices.clone(),
                "reorder_enabled": channel_reorder_enabled,
                "reorder_mode": channel_reorder_mode or "score_desc",
                "block_alignment_enabled": block_alignment_enabled,
                "block_alignment_size": block_alignment_size,
                "block_alignment_rounding": block_alignment_rounding,
            }
            if q_perm is not None and q_invperm is not None:
                retained_index_map[q_key]["row_reorder_perm"] = q_perm
                retained_index_map[q_key]["row_inverse_reorder_perm"] = q_invperm
                retained_index_map[q2_key]["col_reorder_perm"] = q_perm.clone()
                retained_index_map[q2_key]["col_inverse_reorder_perm"] = q_invperm.clone()

        if "k" in spec:
            k_score_selected_original_indices = torch.tensor(
                sorted(selected_indices.get(("k", layer_idx), [])), dtype=torch.long
            )
            k_original_indices = k_score_selected_original_indices.clone()
            k_indices = k_score_selected_original_indices.clone()
            k_perm = None
            k_invperm = None
            if channel_reorder_enabled:
                k_perm = build_descending_score_permutation(spec["k"]["final_score"])
                k_indices, k_invperm = remap_indices_after_permutation(
                    k_score_selected_original_indices,
                    k_perm,
                )
                if block_alignment_enabled:
                    k_indices = align_prefix_indices_to_block(
                        selected_count=int(k_indices.numel()),
                        total_count=int(k_perm.numel()),
                        block_size=block_alignment_size,
                        rounding_mode=block_alignment_rounding,
                    )
                    k_original_indices = torch.sort(k_perm[k_indices]).values.cpu().long()
            retained_index_map[k_key] = {
                "row_indices": k_indices,
                "original_row_indices": k_original_indices,
                "score_selected_row_indices": k_score_selected_original_indices,
                "reorder_enabled": channel_reorder_enabled,
                "reorder_mode": channel_reorder_mode or "score_desc",
                "block_alignment_enabled": block_alignment_enabled,
                "block_alignment_size": block_alignment_size,
                "block_alignment_rounding": block_alignment_rounding,
            }
            retained_index_map[k2_key] = {
                "col_indices": k_indices.clone(),
                "original_col_indices": k_original_indices.clone(),
                "score_selected_col_indices": k_score_selected_original_indices.clone(),
                "reorder_enabled": channel_reorder_enabled,
                "reorder_mode": channel_reorder_mode or "score_desc",
                "block_alignment_enabled": block_alignment_enabled,
                "block_alignment_size": block_alignment_size,
                "block_alignment_rounding": block_alignment_rounding,
            }
            if k_perm is not None and k_invperm is not None:
                retained_index_map[k_key]["row_reorder_perm"] = k_perm
                retained_index_map[k_key]["row_inverse_reorder_perm"] = k_invperm
                retained_index_map[k2_key]["col_reorder_perm"] = k_perm.clone()
                retained_index_map[k2_key]["col_inverse_reorder_perm"] = k_invperm.clone()

    return retained_index_map
