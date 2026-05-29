import datetime
import json
import os
import re
from logging import Logger
from typing import Dict, Optional, Tuple

import datasets
import torch
import torch.distributed as dist
import transformers
from safetensors.torch import load_file
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator

from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main_split import prepare_model
from train_utils.modeling_llama_split import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import hadamard_matrix, random_hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

log: Logger = get_logger("spinquant")


# ============================================================================
# R_mid 初始化开关
# True  -> 使用显式分块 Hadamard R_mid：训练一个 128x128 块，再沿对角重复拼接
# False -> 保持原始的 full-rank random Hadamard R_mid 初始化方式
# ============================================================================
USE_EXPLICIT_BLOCK_HADAMARD_RMID = True
RMID_BLOCK_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 4
RMID_ORTHOGONALITY_CHECK_INTERVAL = 10
RMID_ORTHOGONALITY_ATOL = 1e-4
RMID_ORTHOGONALITY_RTOL = 1e-4


# ============================================================================
# R1 / R2 加载与输出文件名
# ============================================================================
PRETRAINED_R_PATH = "/data1/ljs/SpinQuant_split_spin_middle_activation/llama_rotation_split_AFM/5%_q_k_Layerwise_groupsize32_lr_1_5.bin"  # 为空字符串时从头训练 R1/R2；否则从该路径加载预训练好的 R1/R2。
FINAL_R_BASE_FILENAME = "5%_q_k_Layerwise_groupsize32_lr_1_5.bin"  # 仅在从头训练 R1/R2 时使用：训练结束后保存基础旋转矩阵的文件名，包含 R1 和每层的 R2。
FINAL_RMID_FILENAME = "layerwise_rmid_final.bin"  # 训练结束后保存最终 R_mid_q / R_mid_k 的文件名。
BEST_RMID_FILENAME = "layerwise_rmid_best.bin"  # 训练过程中按 loss 最优保存 R_mid_q / R_mid_k 的文件名。
METRICS_FILENAME = "training_metrics.jsonl"  # 训练指标日志文件名，按 JSONL 记录 step、epoch、loss、grad_norm 和 learning_rate。
RUN_CONFIG_FILENAME = "run_config.json"  # 本次训练的完整配置文件，记录所有超参数、输入模型路径和输出目录信息。

FINAL_R_BASE_FILENAME = "5%_q_k_Layerwise_groupsize32_lr_1_5.bin"  # 仅在从头训练 R1/R2 时使用：训练结束后保存基础旋转矩阵的文件名，包含 R1 和每层的 R2。
FINAL_RMID_FILENAME = "layerwise_rmid_final.bin"  # 训练结束后保存最终 R_mid_q / R_mid_k 的文件名。
BEST_RMID_FILENAME = "layerwise_rmid_best.bin"  # 训练过程中按 loss 最优保存 R_mid_q / R_mid_k 的文件名。
METRICS_FILENAME = "training_metrics.jsonl"  # 训练指标日志文件名，按 JSONL 记录 step、epoch、loss、grad_norm 和 learning_rate。
RUN_CONFIG_FILENAME = "run_config.json"  # 本次训练的完整配置文件，记录所有超参数、输入模型路径和输出目录信息。


# 明确定义一次输出文件名，避免上方注释/编码问题影响运行时常量。
FINAL_R_BASE_FILENAME, FINAL_RMID_FILENAME, BEST_RMID_FILENAME, METRICS_FILENAME, RUN_CONFIG_FILENAME = (
    "5%_q_k_Layerwise_groupsize32_lr_1_5.bin",
    "layerwise_rmid_final.bin",
    "layerwise_rmid_best.bin",
    "training_metrics.jsonl",
    "run_config.json",
)


def is_fsdp_enabled(training_args) -> bool:
    return training_args.fsdp != "" and training_args.fsdp != []


def load_sharded_safetensors(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    full_state_dict = {}
    for filename in os.listdir(ckpt_dir):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(ckpt_dir, filename)
            log.info(f"Loading shard: {filename}")
            shard_dict = load_file(file_path)
            full_state_dict.update(shard_dict)

    if not full_state_dict:
        raise ValueError(f"No .safetensors files found in {ckpt_dir}")

    return full_state_dict


def load_model_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if os.path.isdir(ckpt_path):
        return load_sharded_safetensors(ckpt_path)
    return load_file(ckpt_path)


def svd_llm_standardized_to_ptq_structure(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    rename_map = {
        "q_proj.linear2": "q_proj_2",
        "q_proj.linear1": "q_proj_1",
        "k_proj.linear2": "k_proj_2",
        "k_proj.linear1": "k_proj_1",
    }

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in rename_map.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
                break
        new_state_dict[new_key] = value
    return new_state_dict


def get_rank_and_flag(
    state_dict: Dict[str, torch.Tensor], key_prefix: str
) -> Tuple[Optional[int], bool]:
    lowrank_key = f"{key_prefix}_1.weight"
    full_key = f"{key_prefix}.weight"

    if lowrank_key in state_dict:
        return state_dict[lowrank_key].shape[0], True
    if full_key in state_dict:
        return None, False
    raise KeyError(f"{key_prefix} not found in state_dict")


def populate_lowrank_config(config, state_dict: Dict[str, torch.Tensor]) -> None:
    rank_attrs = ["q_rank", "k_rank", "v_rank", "o_rank", "gate_rank", "up_rank", "down_rank"]
    flag_attrs = [
        "q_lowrank",
        "k_lowrank",
        "v_lowrank",
        "o_lowrank",
        "gate_lowrank",
        "up_lowrank",
        "down_lowrank",
    ]

    for attr in rank_attrs:
        setattr(config, attr, [None] * config.num_hidden_layers)
    for attr in flag_attrs:
        setattr(config, attr, [False] * config.num_hidden_layers)

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        config.q_rank[layer_idx], config.q_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.self_attn.q_proj"
        )
        config.k_rank[layer_idx], config.k_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.self_attn.k_proj"
        )
        config.v_rank[layer_idx], config.v_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.self_attn.v_proj"
        )
        config.o_rank[layer_idx], config.o_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.self_attn.o_proj"
        )
        config.gate_rank[layer_idx], config.gate_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.mlp.gate_proj"
        )
        config.up_rank[layer_idx], config.up_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.mlp.up_proj"
        )
        config.down_rank[layer_idx], config.down_lowrank[layer_idx] = get_rank_and_flag(
            state_dict, f"{prefix}.mlp.down_proj"
        )


def get_cpu_state_dict(model, training_args) -> Dict[str, torch.Tensor]:
    if is_fsdp_enabled(training_args):
        return pt_fsdp_state_dict(model)
    return model.state_dict()


def filter_rotation_state(
    state_dict: Dict[str, torch.Tensor], include_rmid_only: bool
) -> Dict[str, torch.Tensor]:
    if include_rmid_only:
        return {
            key.replace(".weight", ""): value
            for key, value in state_dict.items()
            if "R_mid_q" in key or "R_mid_k" in key
        }

    return {
        key.replace(".weight", ""): value
        for key, value in state_dict.items()
        if "R1.weight" in key or "self_attn.R2" in key
    }


def check_rmid_orthogonality(
    model,
    training_args,
    atol: float,
    rtol: float,
) -> Dict[str, float]:
    cpu_state = get_cpu_state_dict(model, training_args)
    r_mid_dict = filter_rotation_state(cpu_state, include_rmid_only=True)

    max_error = 0.0
    worst_key = ""
    checked_count = 0

    for key, value in r_mid_dict.items():
        rotation = value.detach().to(torch.float32)
        identity = torch.eye(
            rotation.size(0),
            device=rotation.device,
            dtype=rotation.dtype,
        )
        gram = rotation @ rotation.t()
        deviation = gram - identity
        layer_max_error = deviation.abs().max().item()

        checked_count += 1
        if layer_max_error > max_error:
            max_error = layer_max_error
            worst_key = key

        if not torch.allclose(gram, identity, atol=atol, rtol=rtol):
            raise RuntimeError(
                f"R_mid orthogonality check failed for {key}: "
                f"max|R_mid R_mid^T - I| = {layer_max_error:.6e}, "
                f"atol={atol}, rtol={rtol}"
            )

    return {
        "checked_count": checked_count,
        "max_error": max_error,
        "worst_key": worst_key,
    }


def sanitize_path_component(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown_model"


def get_lowrank_model_name(lowrank_model_path: str) -> str:
    normalized = os.path.normpath(lowrank_model_path)
    basename = os.path.basename(normalized)
    stem, _ = os.path.splitext(basename)
    return sanitize_path_component(stem or basename)


def prepare_run_output_dir(model_args, training_args, ptq_args, local_rank: int) -> Tuple[str, str]:
    timestamp_holder = [
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if local_rank == 0 else None
    ]
    dist.broadcast_object_list(timestamp_holder, src=0)
    timestamp = timestamp_holder[0]
    lowrank_model_name = get_lowrank_model_name(ptq_args.svd_llm_ckpt)
    run_dir_name = f"{lowrank_model_name}_{timestamp}"
    run_output_dir = os.path.join(model_args.output_rotation_path, run_dir_name)

    os.makedirs(run_output_dir, exist_ok=True)
    model_args.output_rotation_path = run_output_dir
    training_args.output_dir = run_output_dir

    return run_output_dir, timestamp


def to_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if hasattr(obj, "to_dict"):
        return to_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return to_serializable(vars(obj))
    return str(obj)


def save_run_config(
    output_dir: str,
    timestamp: str,
    model_args,
    training_args,
    ptq_args,
    loaded_pretrained_base_rotations: bool,
    trainable_param_count: int,
) -> None:
    run_config = {
        "timestamp": timestamp,
        "run_output_dir": output_dir,
        "input_model": model_args.input_model,
        "low_rank_model_path": ptq_args.svd_llm_ckpt,
        "low_rank_model_name": get_lowrank_model_name(ptq_args.svd_llm_ckpt),
        "pretrained_r_path": PRETRAINED_R_PATH.strip() or None,
        "r1_r2_source": PRETRAINED_R_PATH.strip() if loaded_pretrained_base_rotations else "train_from_scratch",
        "loaded_pretrained_base_rotations": loaded_pretrained_base_rotations,
        "use_explicit_block_hadamard_rmid": USE_EXPLICIT_BLOCK_HADAMARD_RMID,
        "rmid_block_size": RMID_BLOCK_SIZE,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "trainable_parameter_count": trainable_param_count,
        "model_args": to_serializable(model_args),
        "training_args": to_serializable(training_args),
        "ptq_args": to_serializable(ptq_args),
    }

    config_path = os.path.join(output_dir, RUN_CONFIG_FILENAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    log.info(f"Run config saved to {config_path}")


class SaveBestRmidCallback(transformers.TrainerCallback):
    def __init__(self, output_dir: str, local_rank: int):
        self.best_loss = float("inf")
        self.output_dir = output_dir
        self.local_rank = local_rank

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.local_rank != 0 or logs is None or "loss" not in logs:
            return

        current_loss = logs["loss"]
        if current_loss >= self.best_loss:
            return

        self.best_loss = current_loss
        model = kwargs["model"]
        cpu_state = get_cpu_state_dict(model, args)
        r_mid_dict = filter_rotation_state(cpu_state, include_rmid_only=True)

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, BEST_RMID_FILENAME)
        torch.save(r_mid_dict, save_path)
        print(f"\n[Callback] Found lower loss {current_loss:.6f}; updated {BEST_RMID_FILENAME}")


class SaveMetricsCallback(transformers.TrainerCallback):
    def __init__(self, log_filepath: str, local_rank: int):
        self.log_filepath = log_filepath
        self.local_rank = local_rank
        if self.local_rank == 0:
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.local_rank != 0 or logs is None or "loss" not in logs:
            return

        log_entry = {
            "step": state.global_step,
            "epoch": logs.get("epoch"),
            "loss": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "learning_rate": logs.get("learning_rate"),
        }
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


class CheckRmidOrthogonalityCallback(transformers.TrainerCallback):
    def __init__(
        self,
        local_rank: int,
        interval: int = RMID_ORTHOGONALITY_CHECK_INTERVAL,
        atol: float = RMID_ORTHOGONALITY_ATOL,
        rtol: float = RMID_ORTHOGONALITY_RTOL,
    ):
        self.local_rank = local_rank
        self.interval = interval
        self.atol = atol
        self.rtol = rtol

    def _run_check(self, args, state, model, stage: str) -> None:
        result = check_rmid_orthogonality(
            model=model,
            training_args=args,
            atol=self.atol,
            rtol=self.rtol,
        )
        if self.local_rank == 0:
            log.info(
                f"[R_mid orthogonality] {stage}: checked {result['checked_count']} matrices; "
                f"worst={result['worst_key'] or 'N/A'}; "
                f"max|R_mid R_mid^T - I|={result['max_error']:.6e}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        if epoch is None:
            return

        completed_epoch = int(round(epoch))
        if abs(epoch - completed_epoch) > 1e-6:
            return

        if completed_epoch <= 0 or completed_epoch % self.interval != 0:
            return

        self._run_check(
            args=args,
            state=state,
            model=kwargs["model"],
            stage=f"epoch {completed_epoch}",
        )

    def on_train_end(self, args, state, control, **kwargs):
        self._run_check(
            args=args,
            state=state,
            model=kwargs["model"],
            stage="train end",
        )


class RotateModule(nn.Module):
    def __init__(self, init_weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(init_weight.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose: bool = False):
        return x @ self.weight if transpose else self.weight @ x


class BlockDiagonalRepeatRotateModule(nn.Module):
    """
    对外暴露完整的 [rank, rank] 块对角矩阵，
    但内部只训练一个共享的 [block_size, block_size] 参数块。
    """

    def __init__(self, block_init: torch.Tensor, full_size: int, block_size: int):
        super().__init__()
        if full_size % block_size != 0:
            raise ValueError(
                f"full_size={full_size} must be divisible by block_size={block_size}"
            )

        self.full_size = full_size
        self.block_size = block_size
        self.num_blocks = full_size // block_size
        self.block_weight = nn.Parameter(
            block_init.to(torch.float32).to(torch.device("cuda"))
        )

    @property
    def weight(self) -> torch.Tensor:
        eye = torch.eye(
            self.num_blocks,
            device=self.block_weight.device,
            dtype=self.block_weight.dtype,
        )
        return torch.kron(eye, self.block_weight)

    def forward(self, x, transpose: bool = False):
        weight = self.weight
        return x @ weight if transpose else weight @ x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = self.weight if keep_vars else self.weight.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        block_key = prefix + "block_weight"

        if block_key in state_dict:
            loaded = state_dict[block_key]
            if loaded.shape != self.block_weight.shape:
                error_msgs.append(
                    f"size mismatch for {block_key}: expected {tuple(self.block_weight.shape)}, got {tuple(loaded.shape)}"
                )
            else:
                self.block_weight.data.copy_(
                    loaded.to(self.block_weight.device, dtype=self.block_weight.dtype)
                )
            if strict and weight_key in missing_keys:
                missing_keys.remove(weight_key)
            return

        if weight_key not in state_dict:
            if strict:
                missing_keys.append(weight_key)
            return

        loaded = state_dict[weight_key]
        expected_shape = (self.full_size, self.full_size)
        if loaded.shape != expected_shape:
            error_msgs.append(
                f"size mismatch for {weight_key}: expected {expected_shape}, got {tuple(loaded.shape)}"
            )
            return

        first_block = loaded[: self.block_size, : self.block_size]
        ref = torch.kron(
            torch.eye(self.num_blocks, device=loaded.device, dtype=loaded.dtype),
            first_block,
        )
        if not torch.allclose(loaded, ref, atol=1e-5, rtol=1e-5):
            error_msgs.append(
                f"{weight_key} is not compatible with repeated block-diagonal structure."
            )
            return

        self.block_weight.data.copy_(
            first_block.to(self.block_weight.device, dtype=self.block_weight.dtype)
        )


def freeze_model_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def attach_base_rotations(model) -> None:
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // model.config.num_attention_heads

    model.R1 = RotateModule(random_hadamard_matrix(hidden_size, "cuda"))
    for layer in model.model.layers:
        layer.self_attn.R2 = RotateModule(random_hadamard_matrix(head_dim, "cuda"))


def load_pretrained_base_rotations(model, local_rank: int) -> bool:
    pretrained_r_path = PRETRAINED_R_PATH.strip()
    if not pretrained_r_path:
        if local_rank == 0:
            log.info("PRETRAINED_R_PATH is empty; R1/R2 will be trained from scratch.")
        return False

    if not os.path.exists(pretrained_r_path):
        raise FileNotFoundError(f"PRETRAINED_R_PATH does not exist: {pretrained_r_path}")

    if local_rank == 0:
        log.info(f"======== Loading pre-trained R1 and R2 from {pretrained_r_path} ========")
    r_state_dict = torch.load(pretrained_r_path, map_location="cpu")
    model.load_state_dict(r_state_dict, strict=False)
    return True


def freeze_base_rotations(model) -> None:
    model.R1.weight.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.R2.weight.requires_grad = False


def get_base_rotation_trainable_params(model) -> list[nn.Parameter]:
    trainable_params = [model.R1.weight]
    model.R1.weight.requires_grad = True

    for layer in model.model.layers:
        layer.self_attn.R2.weight.requires_grad = True
        trainable_params.append(layer.self_attn.R2.weight)

    return trainable_params


def build_rmid_module(rank: int) -> nn.Module:
    if USE_EXPLICIT_BLOCK_HADAMARD_RMID:
        if rank % RMID_BLOCK_SIZE != 0:
            raise ValueError(
                f"rank={rank} is not divisible by RMID_BLOCK_SIZE={RMID_BLOCK_SIZE}"
            )
        return BlockDiagonalRepeatRotateModule(
            hadamard_matrix(RMID_BLOCK_SIZE, "cuda"),
            full_size=rank,
            block_size=RMID_BLOCK_SIZE,
        )

    return RotateModule(random_hadamard_matrix(rank, "cuda"))


def get_rmid_trainable_param(module: nn.Module) -> nn.Parameter:
    if isinstance(module, BlockDiagonalRepeatRotateModule):
        module.block_weight.requires_grad = True
        return module.block_weight

    module.weight.requires_grad = True
    return module.weight


def attach_trainable_rmid(model, config) -> list[nn.Parameter]:
    trainable_params = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        if config.q_lowrank[layer_idx]:
            attn.R_mid_q = build_rmid_module(config.q_rank[layer_idx])
            trainable_params.append(get_rmid_trainable_param(attn.R_mid_q))

        if config.k_lowrank[layer_idx]:
            attn.R_mid_k = build_rmid_module(config.k_rank[layer_idx])
            trainable_params.append(get_rmid_trainable_param(attn.R_mid_k))

    return trainable_params


def build_model_and_config(model_args, training_args, ptq_args):
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model,
        token=model_args.access_token,
    )

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    state_dict = load_model_state(ptq_args.svd_llm_ckpt)
    state_dict = svd_llm_standardized_to_ptq_structure(state_dict)
    populate_lowrank_config(config, state_dict)

    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=None,
        state_dict=state_dict,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    return model, config


def build_train_dataset(tokenizer, model_max_length: int):
    raw_train = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    return CustomJsonDataset(
        raw_train,
        tokenizer,
        block_size=min(model_max_length, 2048),
    )


def build_trainer(
    model,
    tokenizer,
    training_args,
    train_data,
    optimizer,
    callbacks,
):
    trainer_cls = FSDPTrainer if is_fsdp_enabled(training_args) else Trainer
    return trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )


def save_final_artifacts(
    model,
    training_args,
    output_rotation_path: str,
    metrics_log_path: str,
    save_base_rotations: bool,
) -> None:
    cpu_state = get_cpu_state_dict(model, training_args)
    os.makedirs(output_rotation_path, exist_ok=True)

    r_mid_dict = filter_rotation_state(cpu_state, include_rmid_only=True)

    if save_base_rotations:
        r_base_dict = filter_rotation_state(cpu_state, include_rmid_only=False)
        torch.save(r_base_dict, os.path.join(output_rotation_path, FINAL_R_BASE_FILENAME))
    torch.save(r_mid_dict, os.path.join(output_rotation_path, FINAL_RMID_FILENAME))

    if save_base_rotations:
        log.info(f"Base R1/R2 and final R_mid saved. Best R_mid is at {BEST_RMID_FILENAME}")
    else:
        log.info(f"Pretrained base R1/R2 was reused, so only final R_mid was saved. Best R_mid is at {BEST_RMID_FILENAME}")
    log.info(f"Training metrics (loss, grad_norm) are saved at {metrics_log_path}")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    run_output_dir, timestamp = prepare_run_output_dir(
        model_args, training_args, ptq_args, local_rank
    )
    if local_rank == 0:
        log.info(f"Current run output directory: {run_output_dir}")

    log.info(f"the rank is {local_rank}")
    dist.barrier()

    model, config = build_model_and_config(model_args, training_args, ptq_args)
    model = prepare_model(ptq_args, model)
    model.model.embed_tokens.register_forward_hook(
        lambda module, input, output: output.requires_grad_(True)
    )

    freeze_model_parameters(model)
    attach_base_rotations(model)
    loaded_pretrained_base_rotations = load_pretrained_base_rotations(model, local_rank)

    trainable_params = []
    if loaded_pretrained_base_rotations:
        freeze_base_rotations(model)
    else:
        trainable_params.extend(get_base_rotation_trainable_params(model))

    trainable_params.extend(attach_trainable_rmid(model, config))
    training_args.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS

    if local_rank == 0:
        log.info(f"Model init completed. Trainable parameter group count: {len(trainable_params)}")
        log.info("Start to load tokenizer...")
        save_run_config(
            run_output_dir,
            timestamp,
            model_args,
            training_args,
            ptq_args,
            loaded_pretrained_base_rotations,
            trainable_param_count=len(trainable_params),
        )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        token=model_args.access_token,
    )

    model.config.use_cache = False
    model.seqlen = training_args.model_max_length
    train_data = build_train_dataset(tokenizer, training_args.model_max_length)

    optimizer = SGDG(trainable_params, lr=training_args.learning_rate, stiefel=True)
    metrics_log_path = os.path.join(model_args.output_rotation_path, METRICS_FILENAME)
    callbacks = [
        SaveBestRmidCallback(model_args.output_rotation_path, local_rank),
        SaveMetricsCallback(metrics_log_path, local_rank),
        CheckRmidOrthogonalityCallback(local_rank),
    ]
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_data=train_data,
        optimizer=optimizer,
        callbacks=callbacks,
    )

    dist.barrier()
    trainer.train()

    if local_rank == 0:
        save_final_artifacts(
            trainer.model,
            training_args,
            model_args.output_rotation_path,
            metrics_log_path,
            save_base_rotations=not loaded_pretrained_base_rotations,
        )

    dist.barrier()


if __name__ == "__main__":
    train()
