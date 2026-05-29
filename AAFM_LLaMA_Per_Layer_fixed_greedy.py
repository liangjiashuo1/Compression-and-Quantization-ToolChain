import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPINQUANT_ROOT = None
for candidate_name in (
    "Codex_AFM_SpinQuant_mixed",
    "Codex_AFM_SpinQuant_mixed_Calibration_paddingmask",
    "Codex_AFM_SPINQuant_mixed_Calibration_padding_nomask",
):
    candidate_root = os.path.join(CURRENT_DIR, candidate_name)
    if os.path.isdir(candidate_root):
        SPINQUANT_ROOT = candidate_root
        break
if SPINQUANT_ROOT is None:
    raise FileNotFoundError(
        "Unable to locate the SpinQuant project root next to "
        f"{os.path.basename(__file__)}."
    )
if SPINQUANT_ROOT not in sys.path:
    sys.path.insert(0, SPINQUANT_ROOT)

from utils import data_utils as spin_data_utils
from eval_utils.channel_selection import (
    absorb_random_spin_into_lowrank_pair,
    compute_logits_aware_channel_score,
    compute_lowrank_output,
    quantize_weight_for_score,
)


LLAMA_LAYER_NUM = 32
LAYER_TYPES = ("q", "k")
LAYER_TYPE_NAME_MAP = {"q": "q_proj", "k": "k_proj"}
LAYER_TYPE_RANK_DICT = {"q": 4096 + 4096, "k": 4096 + 4096}
LAYER_TYPE_CONST_DICT = {"q": 4096, "k": 4096}


class TwoLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        rank: int,
        out_features: int,
        bias2_val: Optional[torch.Tensor],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.linear2 = nn.Linear(
            rank,
            out_features,
            bias=(bias2_val is not None),
            dtype=dtype,
        )
        if bias2_val is not None:
            self.linear2.bias.data = bias2_val.to(dtype)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def AFM(layers_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    layers_y = layers_y.to(torch.float32)
    e_y_layer = torch.mean(layers_y, dim=0, keepdim=False)
    e_yyt_layer = torch.matmul(layers_y.T, layers_y) / layers_y.shape[0]
    if torch.isnan(e_y_layer).any() or torch.isinf(e_y_layer).any():
        raise ValueError("E_y_layer contains NaN or Inf values.")
    if torch.isnan(e_yyt_layer).any() or torch.isinf(e_yyt_layer).any():
        raise ValueError("E_yyT_layer contains NaN or Inf values.")
    return e_yyt_layer, e_y_layer


def update_afm_statistics(
    e_yyt_layer: torch.Tensor,
    e_y_layer: torch.Tensor,
    layer_output: torch.Tensor,
    count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    e_yyt, e_y = AFM(layer_output)
    count = count.to(e_y_layer.device)
    e_y_layer.mul_(count)
    e_y_layer.add_(e_y.to(e_y_layer.device))
    e_y_layer.div_(count + 1)

    e_yyt_layer.mul_(count)
    e_yyt_layer.add_(e_yyt.to(e_yyt_layer.device))
    e_yyt_layer.div_(count + 1)
    return e_yyt_layer, e_y_layer


def parse_rank_candidates(rank_candidates: str) -> List[int]:
    if not rank_candidates.strip():
        return [i for i in range(128, 2048 + 1, 128)]
    values = []
    for chunk in rank_candidates.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    values = sorted(set(values))
    return [rank for rank in values if rank > 0 and rank % 128 == 0]


def build_device_map(num_layers: int) -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {}
    gpu_count = max(1, torch.cuda.device_count())
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }
    for layer_idx in range(num_layers):
        device_map[f"model.layers.{layer_idx}"] = f"cuda:{layer_idx % gpu_count}"
    return device_map


def get_embed_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_eval_device() -> torch.device:
    if torch.cuda.is_available():
        gpu_idx = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")


def build_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return (input_ids != pad_token_id).long()


def get_model_inputs(
    batch,
    tokenizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Received an empty batch tuple/list.")
        input_ids = batch[0]
        attention_mask = None
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = build_attention_mask(input_ids, tokenizer.pad_token_id)
    else:
        attention_mask = attention_mask.to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def mask_logits_by_attention(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D [B, T, V], got shape {tuple(logits.shape)}")
    if attention_mask.shape != logits.shape[:2]:
        raise ValueError(
            "attention_mask shape does not match logits shape: "
            f"{tuple(attention_mask.shape)} vs {tuple(logits.shape[:2])}"
        )

    flat_mask = attention_mask.reshape(-1).bool()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    return flat_logits[flat_mask]


def load_cola_loader(
    tokenizer,
    model_name_or_path: str,
    calibration_path: Optional[str],
    nsamples: int,
    max_length: int,
    seed: int,
):
    return spin_data_utils.get_channel_selection_calibration_loader(
        calibration_source="cola_json",
        nsamples=nsamples,
        seed=seed,
        seqlen=max_length,
        model=model_name_or_path,
        tokenizer=tokenizer,
        calibration_path=calibration_path,
    )


def build_lowrank_weights(
    ori_model,
    layer_type: str,
    layer_idx: int,
    rank: int,
    e_y_dict,
    e_yyt_dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attr_name = LAYER_TYPE_NAME_MAP[layer_type]
    parent = ori_model.model.layers[layer_idx].self_attn
    full_weight = getattr(parent, attr_name).weight.detach()
    target_device = full_weight.device
    e_yyt = e_yyt_dict[layer_type][layer_idx].detach().float().to(target_device)
    e_y = e_y_dict[layer_type][layer_idx].detach().float().to(target_device)

    basis = e_yyt[:, -rank:]
    w1 = basis.T.float().matmul(full_weight.float().to(target_device))
    w2 = basis.float()
    b2 = e_y - basis.matmul(basis.T.matmul(e_y))
    return w1, w2, b2


def build_qerror_quant_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        w_bits=args.qerror_w_bits,
        w_asym=args.qerror_w_asym,
        w_clip=args.qerror_w_clip,
        w_groupsize=args.qerror_w_groupsize,
    )


@torch.no_grad()
def apply_compression(
    model_to_compress,
    ori_model,
    tune_dict,
    rank_dict,
    e_y_dict,
    e_yyt_dict,
    aafm_original_layers,
):
    layer_type_map = {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "o_proj",
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
    }
    for layer_idx in range(LLAMA_LAYER_NUM):
        for layer_type, attr_name in layer_type_map.items():
            mod_parent = (
                model_to_compress.model.layers[layer_idx].self_attn
                if layer_type in ["q", "k", "v", "o"]
                else model_to_compress.model.layers[layer_idx].mlp
            )
            ori_parent = (
                ori_model.model.layers[layer_idx].self_attn
                if layer_type in ["q", "k", "v", "o"]
                else ori_model.model.layers[layer_idx].mlp
            )

            key = f"{layer_type}_{layer_idx}"
            if key not in aafm_original_layers:
                aafm_original_layers[key] = getattr(mod_parent, attr_name)

            if tune_dict[layer_type][layer_idx]:
                rank = rank_dict[layer_type][layer_idx]
                if layer_type in LAYER_TYPES:
                    w1, w2, b2 = build_lowrank_weights(
                        ori_model=ori_model,
                        layer_type=layer_type,
                        layer_idx=layer_idx,
                        rank=rank,
                        e_y_dict=e_y_dict,
                        e_yyt_dict=e_yyt_dict,
                    )
                    full_weight = getattr(ori_parent, attr_name).weight
                    new_proj = TwoLinear(
                        full_weight.shape[1],
                        rank,
                        full_weight.shape[0],
                        b2,
                        dtype=full_weight.dtype,
                    )
                    new_proj.linear1.weight.data = w1.to(full_weight.dtype)
                    new_proj.linear2.weight.data = w2.to(full_weight.dtype)
                    new_proj = new_proj.to(aafm_original_layers[key].weight.device)
                    setattr(mod_parent, attr_name, new_proj)
                else:
                    setattr(mod_parent, attr_name, aafm_original_layers[key])
            else:
                setattr(mod_parent, attr_name, aafm_original_layers[key])
    return model_to_compress


def compute_rank_score_kl(
    tokenizer,
    ori_model,
    aafm_model,
    layer_idx: int,
    shared_rank: int,
    tune_dict,
    rank_dict,
    e_y_dict,
    e_yyt_dict,
    search_loader,
    aafm_original_layers,
) -> float:
    tune_dict["q"][layer_idx] = True
    tune_dict["k"][layer_idx] = True
    rank_dict["q"][layer_idx] = shared_rank
    rank_dict["k"][layer_idx] = shared_rank

    apply_compression(
        aafm_model,
        ori_model,
        tune_dict,
        rank_dict,
        e_y_dict,
        e_yyt_dict,
        aafm_original_layers,
    )
    aafm_model.eval()

    score = torch.tensor(0.0, device=get_eval_device())
    valid_token_count = 0
    with torch.no_grad():
        for batch in search_loader:
            ori_inputs = get_model_inputs(batch, tokenizer, get_embed_device())
            aafm_inputs = get_model_inputs(batch, tokenizer, get_eval_device())

            ori_outputs = ori_model(**ori_inputs)
            aafm_outputs = aafm_model(**aafm_inputs)

            attention_mask = aafm_inputs["attention_mask"].to(get_eval_device())
            if not attention_mask.any():
                continue

            ori_logits = ori_outputs.logits.to(get_eval_device(), dtype=torch.float32)
            aafm_logits = aafm_outputs.logits.to(dtype=torch.float32)
            ori_logits = mask_logits_by_attention(ori_logits, attention_mask)
            aafm_logits = mask_logits_by_attention(aafm_logits, attention_mask)
            if ori_logits.numel() == 0:
                continue

            ori_probs = F.softmax(ori_logits, dim=-1)
            aafm_log_probs = F.log_softmax(aafm_logits, dim=-1)

            score = score + F.kl_div(
                aafm_log_probs,
                ori_probs,
                reduction="batchmean",
            )
            valid_token_count += int(attention_mask.sum().item())

    tune_dict["q"][layer_idx] = False
    tune_dict["k"][layer_idx] = False
    if valid_token_count <= 0:
        raise ValueError("No valid tokens were found in search_loader after applying attention_mask.")
    return float(score.item())


@torch.no_grad()
def compute_rank_score_qerror_logits_aware(
    tokenizer,
    ori_model,
    layer_idx: int,
    shared_rank: int,
    e_y_dict,
    e_yyt_dict,
    search_loader,
    args,
) -> float:
    q_w1, q_w2, _ = build_lowrank_weights(
        ori_model=ori_model,
        layer_type="q",
        layer_idx=layer_idx,
        rank=shared_rank,
        e_y_dict=e_y_dict,
        e_yyt_dict=e_yyt_dict,
    )
    k_w1, k_w2, _ = build_lowrank_weights(
        ori_model=ori_model,
        layer_type="k",
        layer_idx=layer_idx,
        rank=shared_rank,
        e_y_dict=e_y_dict,
        e_yyt_dict=e_yyt_dict,
    )

    if args.qerror_randomspin_enabled:
        q_w1, q_w2 = absorb_random_spin_into_lowrank_pair(
            q_w1,
            q_w2,
            seed=args.qerror_randomspin_seed + layer_idx * 2,
        )
        k_w1, k_w2 = absorb_random_spin_into_lowrank_pair(
            k_w1,
            k_w2,
            seed=args.qerror_randomspin_seed + layer_idx * 2 + 1,
        )

    quant_args = build_qerror_quant_args(args)
    q_qw1 = quantize_weight_for_score(q_w1, quant_args).to(q_w1.device)
    k_qw1 = quantize_weight_for_score(k_w1, quant_args).to(k_w1.device)
    q_delta_w1 = q_w1 - q_qw1
    k_delta_w1 = k_w1 - k_qw1

    captured_inputs: List[torch.Tensor] = []
    attn_module = ori_model.model.layers[layer_idx].self_attn

    def capture_attn_input(_, hook_input):
        layer_input = hook_input[0].detach().reshape(-1, hook_input[0].shape[-1])
        captured_inputs.append(layer_input)

    handle = attn_module.q_proj.register_forward_pre_hook(capture_attn_input)
    total_score = 0.0
    valid_token_count = 0
    try:
        for batch in search_loader:
            captured_inputs.clear()
            ori_inputs = get_model_inputs(batch, tokenizer, get_embed_device())
            attention_mask = ori_inputs["attention_mask"]
            ori_model(**ori_inputs)

            if not captured_inputs:
                raise RuntimeError(
                    f"Failed to capture q_proj input for layer {layer_idx} during qerror scoring."
                )

            x = captured_inputs[-1].float()
            token_mask = attention_mask.reshape(-1).to(
                device=x.device,
                dtype=torch.float32,
            )
            if token_mask.numel() != x.shape[0]:
                raise ValueError(
                    "Attention mask token count does not match captured attention input: "
                    f"{token_mask.numel()} vs {x.shape[0]}."
                )
            valid_tokens = int(token_mask.sum().item())
            if valid_tokens <= 0:
                continue

            x = x * token_mask.unsqueeze(1)
            q_activation_error = x.matmul(q_delta_w1.t())
            k_activation_error = x.matmul(k_delta_w1.t())
            q_output = compute_lowrank_output(x, q_w1, q_w2, keep_on_device=True)
            k_output = compute_lowrank_output(x, k_w1, k_w2, keep_on_device=True)

            q_score = compute_logits_aware_channel_score(
                activation_error=q_activation_error,
                opposite_output=k_output,
                target_w2=q_w2,
                target_kind="q",
                config=ori_model.config,
                keep_on_device=True,
            )
            k_score = compute_logits_aware_channel_score(
                activation_error=k_activation_error,
                opposite_output=q_output,
                target_w2=k_w2,
                target_kind="k",
                config=ori_model.config,
                keep_on_device=True,
            )
            total_score += float(q_score.mean().item() + k_score.mean().item())
            valid_token_count += valid_tokens
    finally:
        handle.remove()

    if valid_token_count <= 0:
        raise ValueError(
            "No valid tokens were found in search_loader for qerror logits-aware scoring."
        )
    return total_score / float(valid_token_count)


def compute_rank_score(
    tokenizer,
    ori_model,
    aafm_model,
    layer_idx: int,
    shared_rank: int,
    tune_dict,
    rank_dict,
    e_y_dict,
    e_yyt_dict,
    search_loader,
    aafm_original_layers,
    args,
) -> float:
    if args.rank_score_mode == "kl":
        return compute_rank_score_kl(
            tokenizer=tokenizer,
            ori_model=ori_model,
            aafm_model=aafm_model,
            layer_idx=layer_idx,
            shared_rank=shared_rank,
            tune_dict=tune_dict,
            rank_dict=rank_dict,
            e_y_dict=e_y_dict,
            e_yyt_dict=e_yyt_dict,
            search_loader=search_loader,
            aafm_original_layers=aafm_original_layers,
        )
    if args.rank_score_mode == "qerror_logits_aware":
        return compute_rank_score_qerror_logits_aware(
            tokenizer=tokenizer,
            ori_model=ori_model,
            layer_idx=layer_idx,
            shared_rank=shared_rank,
            e_y_dict=e_y_dict,
            e_yyt_dict=e_yyt_dict,
            search_loader=search_loader,
            args=args,
        )
    raise ValueError(f"Unsupported rank_score_mode: {args.rank_score_mode!r}")


def greedy_selection(options_per_layer, max_total_params: int):
    """
    options_per_layer[layer] = [(score, total_params, shared_rank), ...]
    sorted by total_params descending before entering this function.
    """
    sorted_options = [
        sorted(layer_options, key=lambda item: item[1], reverse=True)
        for layer_options in options_per_layer
    ]

    selected_indices = [0] * len(sorted_options)
    current_params_sum = sum(options[0][1] for options in sorted_options)

    if current_params_sum <= max_total_params:
        return [options[0] for options in sorted_options]

    heap = []
    for layer_idx, options in enumerate(sorted_options):
        if len(options) <= 1:
            continue
        curr_score, curr_params, _ = options[0]
        next_score, next_params, _ = options[1]
        delta_score = next_score - curr_score
        delta_params = curr_params - next_params
        if delta_params > 0:
            cost = delta_score / delta_params
            heap.append((cost, layer_idx, 1))

    import heapq

    heapq.heapify(heap)
    while current_params_sum > max_total_params and heap:
        _, layer_idx, next_idx = heapq.heappop(heap)
        curr_idx = selected_indices[layer_idx]
        curr_score, curr_params, _ = sorted_options[layer_idx][curr_idx]
        next_score, next_params, _ = sorted_options[layer_idx][next_idx]

        current_params_sum -= curr_params - next_params
        selected_indices[layer_idx] = next_idx

        if next_idx + 1 < len(sorted_options[layer_idx]):
            new_curr_score, new_curr_params, _ = sorted_options[layer_idx][next_idx]
            new_next_score, new_next_params, _ = sorted_options[layer_idx][next_idx + 1]
            delta_score = new_next_score - new_curr_score
            delta_params = new_curr_params - new_next_params
            if delta_params > 0:
                cost = delta_score / delta_params
                heapq.heappush(heap, (cost, layer_idx, next_idx + 1))

    return [
        sorted_options[layer_idx][selected_indices[layer_idx]]
        for layer_idx in range(len(sorted_options))
    ]


def save_rank_json(output_dir: str, rank_dict: Dict[str, List[Optional[int]]]) -> str:
    output_path = os.path.join(output_dir, "rank_tied_qk.json")
    payload = {
        "q": rank_dict["q"],
        "k": rank_dict["k"],
        "shared_qk": rank_dict["q"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def AAFM_test_function(
    tokenizer,
    ori_model,
    aafm_model,
    tune_dict,
    rank_dict,
    e_y_dict,
    e_yyt_dict,
    args,
    ori_params,
    aafm_original_layers,
):
    apply_compression(
        aafm_model,
        ori_model,
        tune_dict,
        rank_dict,
        e_y_dict,
        e_yyt_dict,
        aafm_original_layers,
    )
    aafm_model.eval()

    compress_params = sum(p.numel() for p in aafm_model.parameters())
    real_compress_rate = 1 - compress_params / ori_params

    save_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(save_dir, exist_ok=True)

    layer_types = ["q", "k", "v", "o", "gate", "up", "down"]
    for layer_type in layer_types:
        lowrank_status = tune_dict[layer_type]
        ranks = [
            rank_dict[layer_type][i] if lowrank_status[i] else None
            for i in range(LLAMA_LAYER_NUM)
        ]
        setattr(aafm_model.config, f"{layer_type}_lowrank", lowrank_status)
        setattr(aafm_model.config, f"{layer_type}_rank", ranks)

    aafm_model.save_pretrained(save_dir)
    print(f"successfully saved to {save_dir}")

    total_loss = 0.0
    total_tokens = 0
    val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    val_iter = iter(val_dataset)

    with torch.no_grad():
        for _ in tqdm(
            range(args.test_samples if args.test_samples > 0 else 1000),
            desc="Evaluating Perplexity",
        ):
            try:
                sample = next(val_iter)
            except StopIteration:
                break
            text = sample["text"]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {key: val.to(get_eval_device()) for key, val in inputs.items()}
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            outputs = aafm_model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss > 0 else float("inf")
    print(f"Perplexity over validation samples: {ppl:.2f}")

    return ppl, real_compress_rate


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device_map = build_device_map(LLAMA_LAYER_NUM)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ori_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map if device_map else None,
    ).eval()
    aafm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=str(get_eval_device()) if torch.cuda.is_available() else None,
    ).eval()
    aafm_original_layers = {}

    e_y_dict = {
        key: [
            torch.zeros(
                (args.dim_map[key]),
                device=args.dev_map[key],
                dtype=torch.float32,
            )
            for _ in range(LLAMA_LAYER_NUM)
        ]
        for key in ["q", "k", "v", "o", "gate", "up", "down"]
    }
    e_yyt_dict = {
        key: [
            torch.zeros(
                (args.dim_map[key], args.dim_map[key]),
                device=args.dev_map[key],
                dtype=torch.float32,
            )
            for _ in range(LLAMA_LAYER_NUM)
        ]
        for key in ["q", "k", "v", "o", "gate", "up", "down"]
    }

    recon_loader = load_cola_loader(
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        calibration_path=args.cola_calibration_path,
        nsamples=args.train_samples,
        max_length=args.max_length,
        seed=args.seed,
    )
    search_loader = load_cola_loader(
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        calibration_path=args.cola_calibration_path,
        nsamples=args.search_samples,
        max_length=args.max_length,
        seed=args.seed + 17,
    )

    count = torch.tensor(0, device=get_embed_device(), dtype=torch.float64)
    global_activations = {}

    def get_activation(name):
        def hook(_, __, output):
            global_activations[name] = output.detach()

        return hook

    hooks = []
    for layer_idx in range(LLAMA_LAYER_NUM):
        for layer_type, attr_name in LAYER_TYPE_NAME_MAP.items():
            mod = getattr(ori_model.model.layers[layer_idx].self_attn, attr_name)
            hooks.append(mod.register_forward_hook(get_activation(f"{layer_type}_{layer_idx}")))

    print(f"Extracting AFM statistics from cola_json using {len(recon_loader)} samples...")
    with torch.no_grad():
        for batch in tqdm(recon_loader, desc="Extracting AFM"):
            inputs = get_model_inputs(batch, tokenizer, get_embed_device())
            ori_model(**inputs)

            attention_mask = inputs["attention_mask"].bool()
            batch_size = inputs["input_ids"].shape[0]
            for sample_idx in range(batch_size):
                sample_mask = attention_mask[sample_idx]
                for layer_type in LAYER_TYPES:
                    for layer_idx in range(LLAMA_LAYER_NUM):
                        act_key = f"{layer_type}_{layer_idx}"
                        layer_output = global_activations[act_key][sample_idx, :, :]
                        layer_mask = sample_mask.to(layer_output.device)
                        if layer_mask.numel() == layer_output.shape[0]:
                            valid_output = layer_output[layer_mask]
                        else:
                            valid_output = layer_output
                        if valid_output.numel() == 0 or valid_output.shape[0] == 0:
                            continue
                        e_yyt_dict[layer_type][layer_idx], e_y_dict[layer_type][layer_idx] = update_afm_statistics(
                            e_yyt_dict[layer_type][layer_idx],
                            e_y_dict[layer_type][layer_idx],
                            valid_output,
                            count,
                        )
                count += 1
            global_activations.clear()

    for hook in hooks:
        hook.remove()

    logging.info("AFM statistics extraction finished in %.2f seconds", time.time() - start_time)

    with torch.no_grad():
        epsilon = 3e-4
        for layer_type in LAYER_TYPES:
            for layer_idx in range(LLAMA_LAYER_NUM):
                e_yyt = e_yyt_dict[layer_type][layer_idx]
                e_y = e_y_dict[layer_type][layer_idx]
                y = e_y.unsqueeze(1)
                cov_y = e_yyt - y @ y.T
                cov_y = cov_y + epsilon * torch.eye(cov_y.shape[0], device=cov_y.device)
                _, svd = torch.linalg.eigh(cov_y.float())
                e_yyt_dict[layer_type][layer_idx].copy_(
                    svd.to(e_yyt_dict[layer_type][layer_idx].device)
                )

    tune_dict = {
        key: [False for _ in range(LLAMA_LAYER_NUM)]
        for key in ["q", "k", "v", "o", "gate", "up", "down"]
    }
    rank_dict = {
        key: [None for _ in range(LLAMA_LAYER_NUM)]
        for key in ["q", "k", "v", "o", "gate", "up", "down"]
    }

    ori_params = sum(p.numel() for p in ori_model.parameters())
    logging.info("Original parameters: %d", ori_params)

    if args.rank_path and os.path.exists(args.rank_path):
        logging.info("Loading precomputed tied qk ranks from %s", args.rank_path)
        with open(args.rank_path, "r", encoding="utf-8") as f:
            loaded_ranks = json.load(f)
        shared_ranks = (
            loaded_ranks.get("shared_qk")
            or loaded_ranks.get("q")
            or loaded_ranks.get("k")
        )
        if shared_ranks is None:
            raise KeyError("rank_path must contain 'shared_qk' or 'q'/'k'.")
        for layer_idx in range(LLAMA_LAYER_NUM):
            rank_dict["q"][layer_idx] = int(shared_ranks[layer_idx])
            rank_dict["k"][layer_idx] = int(shared_ranks[layer_idx])
            tune_dict["q"][layer_idx] = True
            tune_dict["k"][layer_idx] = True
    else:
        candidate_ranks = parse_rank_candidates(args.rank_candidates)
        options_per_layer = []
        for layer_idx in range(LLAMA_LAYER_NUM):
            layer_options = []
            for rank in tqdm(
                candidate_ranks,
                desc=f"Searching shared qk rank for layer {layer_idx}",
                leave=False,
            ):
                score = compute_rank_score(
                    tokenizer=tokenizer,
                    ori_model=ori_model,
                    aafm_model=aafm_model,
                    layer_idx=layer_idx,
                    shared_rank=rank,
                    tune_dict=tune_dict,
                    rank_dict=rank_dict,
                    e_y_dict=e_y_dict,
                    e_yyt_dict=e_yyt_dict,
                    search_loader=search_loader,
                    aafm_original_layers=aafm_original_layers,
                    args=args,
                )
                total_params = 0
                for layer_type in LAYER_TYPES:
                    total_params += (
                        rank * LAYER_TYPE_RANK_DICT[layer_type]
                        + LAYER_TYPE_CONST_DICT[layer_type]
                    )
                layer_options.append((score, total_params, rank))
                logging.info(
                    "Layer %d shared_rank %d -> score %.6f, params %d",
                    layer_idx,
                    rank,
                    score,
                    total_params,
                )
            options_per_layer.append(layer_options)

        original_layer_params = {"q": 4096 * 4096, "k": 4096 * 4096}
        tunable_ori_params = LLAMA_LAYER_NUM * (
            original_layer_params["q"] + original_layer_params["k"]
        )
        untunable_params = ori_params - tunable_ori_params
        max_params = ori_params * (1 - args.compress_rate) - untunable_params
        if max_params <= 0:
            raise ValueError(
                f"compress_rate={args.compress_rate} is too high for q/k-only compression."
            )

        selected_options = greedy_selection(options_per_layer, int(max_params))
        for layer_idx, (_, _, rank) in enumerate(selected_options):
            rank_dict["q"][layer_idx] = int(rank)
            rank_dict["k"][layer_idx] = int(rank)
            tune_dict["q"][layer_idx] = True
            tune_dict["k"][layer_idx] = True

        save_path = save_rank_json(args.output_dir, rank_dict)
        logging.info("Saved tied qk rank json to %s", save_path)

    logging.info("Final q_rank: %s", rank_dict["q"])
    logging.info("Final k_rank: %s", rank_dict["k"])

    ppl, real_compress_rate = AAFM_test_function(
        tokenizer=tokenizer,
        ori_model=ori_model,
        aafm_model=aafm_model,
        tune_dict=tune_dict,
        rank_dict=rank_dict,
        e_y_dict=e_y_dict,
        e_yyt_dict=e_yyt_dict,
        args=args,
        ori_params=ori_params,
        aafm_original_layers=aafm_original_layers,
    )
    logging.info("Real compress rate is %.6f", real_compress_rate)
    logging.info("Final ppl is %.6f", ppl)
    logging.info("Total time is %.2f seconds", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress_rate", type=float, required=True, help="Target compression rate.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Compressed model base path.",
    )
    parser.add_argument(
        "--cola_calibration_path",
        type=str,
        required=True,
        help="Path to cola calibration_samples.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/AAFM_tied_qk",
        help="Output directory for logs / ranks / ckpt.",
    )
    parser.add_argument(
        "--rank_candidates",
        type=str,
        default="1152,1280,1408,1536,1664",
        help="Comma-separated shared qk rank candidates, must be multiples of 128.",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000,
        help="cola_json samples used for AFM statistics.",
    )
    parser.add_argument(
        "--search_samples",
        type=int,
        default=1000,
        help="cola_json samples used for tied qk rank search.",
    )
    parser.add_argument("--test_samples", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--rank_path",
        type=str,
        default=None,
        help="Optional path to precomputed tied qk rank json.",
    )
    parser.add_argument(
        "--rank_score_mode",
        type=str,
        default="kl",
        choices=["kl", "qerror_logits_aware"],
        help="Scoring objective used during shared q/k rank search.",
    )
    parser.add_argument(
        "--qerror_w_bits",
        type=int,
        default=3,
        help="Fake-quant bit width for qerror logits-aware rank scoring.",
    )
    parser.add_argument(
        "--qerror_w_groupsize",
        type=int,
        default=64,
        help="Fake-quant groupsize for qerror logits-aware rank scoring.",
    )
    parser.add_argument(
        "--qerror_w_asym",
        action="store_true",
        help="Use asymmetric fake quantization in qerror logits-aware rank scoring.",
    )
    parser.add_argument(
        "--qerror_w_clip",
        action="store_true",
        help="Enable clipping-aware fake quantization in qerror logits-aware rank scoring.",
    )
    parser.add_argument(
        "--qerror_randomspin_enabled",
        action="store_true",
        help="Apply RandomSpin to the candidate low-rank bottleneck before qerror scoring.",
    )
    parser.add_argument(
        "--qerror_randomspin_seed",
        type=int,
        default=42,
        help="Seed for optional RandomSpin used during qerror logits-aware rank scoring.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.dim_map = {
        "q": 4096,
        "k": 4096,
        "v": 4096,
        "o": 4096,
        "gate": 11008,
        "up": 11008,
        "down": 4096,
    }
    default_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.dev_map = {
        "q": default_dev,
        "k": default_dev,
        "v": default_dev,
        "o": default_dev,
        "gate": default_dev,
        "up": default_dev,
        "down": default_dev,
    }
    main(args)
