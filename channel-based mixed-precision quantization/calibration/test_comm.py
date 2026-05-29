from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import lm_eval
import torch
import transformers
from lm_eval.models.huggingface import HFLM
from transformers import LlamaForCausalLM, LlamaTokenizerFast


LLAMA2_7B_MODEL_PATH = (
    "/data/huggingface_model/models--meta-llama--Llama-2-7b-hf/snapshots/"
    "01c7f73d771dfac7d292323805ebc428287df4f9"
)

DEFAULT_MODEL_CONFIGS = [
    {
        "name": "llama2-7b",
        "state_dict_path": None,
        "model_path": LLAMA2_7B_MODEL_PATH,
    },
    {
        "name": "mixed",
        "state_dict_path": "/data2/wwy/SVD-LLM-calibration/mixed_20%_u@v.pt",
        "model_path": LLAMA2_7B_MODEL_PATH,
    },
    {
        "name": "mmlu",
        "state_dict_path": "/data2/wwy/SVD-LLM-calibration/mmlu_20%_u@v.pt",
        "model_path": LLAMA2_7B_MODEL_PATH,
    },
    {
        "name": "wiki",
        "state_dict_path": "/data2/wwy/SVD-LLM-calibration/wiki_20%_u@v.pt",
        "model_path": LLAMA2_7B_MODEL_PATH,
    },
]

DEFAULT_TASKS = (
    "arc_challenge",
    "social_iqa",
    "boolq",
    "openbookqa",
    "arc_easy",
    "winogrande",
    "hellaswag",
    "piqa",
)
DEFAULT_MMLU_TASKS = ("mmlu",)
DEFAULT_EVAL_SUITES = (
    {
        "name": "commonsense_0shot",
        "tasks": DEFAULT_TASKS,
        "num_fewshot": 0,
    },
    {
        "name": "mmlu_0shot",
        "tasks": DEFAULT_MMLU_TASKS,
        "num_fewshot": 0,
    },
    {
        "name": "mmlu_5shot",
        "tasks": DEFAULT_MMLU_TASKS,
        "num_fewshot": 5,
    },
)

DEFAULT_BATCH_SIZE = 8
DEFAULT_MODEL_MAX_LENGTH = 2048
DEFAULT_DTYPE = torch.bfloat16


def _resolve_device(model: torch.nn.Module | None = None, device: str | None = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"

    if model is not None:
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            pass

    return "cpu"


def _prepare_model_for_lm_eval(model: torch.nn.Module, tokenizer, device: str) -> None:
    model.to(device)
    model.eval()
    model.config.use_cache = False

    if getattr(tokenizer, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.use_cache = False
    generation_config.do_sample = False

    if getattr(generation_config, "temperature", None) is not None:
        generation_config.temperature = 1.0
    if getattr(generation_config, "top_p", None) is not None:
        generation_config.top_p = 1.0

    if getattr(generation_config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if getattr(generation_config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id


def load_model_from_config(
    config: Mapping[str, Any],
    *,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str | None = None,
):
    model_path = config["model_path"]
    state_dict_path = config.get("state_dict_path")

    if state_dict_path is None:
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=dtype,
        )
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu")
        config_obj = transformers.AutoConfig.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=None,
            state_dict=state_dict,
            config=config_obj,
            torch_dtype=dtype,
        )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    resolved_device = _resolve_device(model, device)
    _prepare_model_for_lm_eval(model, tokenizer, resolved_device)
    return model, tokenizer, resolved_device


def evaluate_tasks(
    model: torch.nn.Module,
    tokenizer,
    tasks: Sequence[str] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> Mapping[str, Any]:
    resolved_tasks = list(tasks or DEFAULT_TASKS)
    resolved_device = _resolve_device(model, device)
    _prepare_model_for_lm_eval(model, tokenizer, resolved_device)
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=resolved_device,
    )

    eval_kwargs = {
        "tasks": resolved_tasks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": resolved_device,
        "limit": limit,
    }

    with torch.no_grad():
        return lm_eval.simple_evaluate(hflm, **eval_kwargs)


def evaluate_suites(
    model: torch.nn.Module,
    tokenizer,
    suites: Sequence[Mapping[str, Any]] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    limit: int | None = None,
) -> dict[str, Mapping[str, Any]]:
    all_results: dict[str, Mapping[str, Any]] = {}
    for suite in suites or DEFAULT_EVAL_SUITES:
        suite_name = str(suite["name"])
        suite_tasks = suite.get("tasks")
        suite_num_fewshot = int(suite.get("num_fewshot", 0))
        all_results[suite_name] = evaluate_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks=suite_tasks,
            batch_size=batch_size,
            device=device,
            num_fewshot=suite_num_fewshot,
            limit=limit,
        )
    return all_results


def evaluate_default_suites(
    model: torch.nn.Module,
    tokenizer,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    limit: int | None = None,
) -> dict[str, Mapping[str, Any]]:
    return evaluate_suites(
        model=model,
        tokenizer=tokenizer,
        suites=DEFAULT_EVAL_SUITES,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )


def evaluate_model_config(
    config: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    num_fewshot: int = 0,
    limit: int | None = None,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Mapping[str, Any]:
    model, tokenizer, resolved_device = load_model_from_config(
        config,
        model_max_length=model_max_length,
        dtype=dtype,
        device=device,
    )
    return evaluate_tasks(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=batch_size,
        device=resolved_device,
        num_fewshot=num_fewshot,
        limit=limit,
    )


def evaluate_model_config_suites(
    config: Mapping[str, Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    limit: int | None = None,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> dict[str, Mapping[str, Any]]:
    model, tokenizer, resolved_device = load_model_from_config(
        config,
        model_max_length=model_max_length,
        dtype=dtype,
        device=device,
    )
    return evaluate_default_suites(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=resolved_device,
        limit=limit,
    )


def evaluate_model_configs(
    model_configs: Sequence[Mapping[str, Any]] | None = None,
    *,
    tasks: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    num_fewshot: int = 0,
    limit: int | None = None,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> dict[str, Mapping[str, Any]]:
    all_results: dict[str, Mapping[str, Any]] = {}
    for config in model_configs or DEFAULT_MODEL_CONFIGS:
        result = evaluate_model_config(
            config,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            num_fewshot=num_fewshot,
            limit=limit,
            model_max_length=model_max_length,
            dtype=dtype,
        )
        all_results[config["name"]] = result
    return all_results


def evaluate_model_configs_suites(
    model_configs: Sequence[Mapping[str, Any]] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    limit: int | None = None,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    all_results: dict[str, dict[str, Mapping[str, Any]]] = {}
    for config in model_configs or DEFAULT_MODEL_CONFIGS:
        result = evaluate_model_config_suites(
            config,
            batch_size=batch_size,
            device=device,
            limit=limit,
            model_max_length=model_max_length,
            dtype=dtype,
        )
        all_results[config["name"]] = result
    return all_results


def format_results(results: Mapping[str, Any]) -> str:
    task_results = results.get("results", results)
    if not task_results:
        return "No lm_eval results were produced."

    lines = []
    for task_name, metrics in task_results.items():
        if not isinstance(metrics, Mapping):
            lines.append(f"{task_name}: {metrics}")
            continue

        parts = []
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                parts.append(f"{metric_name}={metric_value:.6f}")
            else:
                parts.append(f"{metric_name}={metric_value}")
        lines.append(f"{task_name}: " + ", ".join(parts))
    return "\n".join(lines)


def format_summary(all_results: Mapping[str, Mapping[str, Any]]) -> str:
    if not all_results:
        return "No model results were produced."

    lines = []
    for model_name, results in all_results.items():
        lines.append(f"{model_name}:")
        lines.append(format_results(results))
    return "\n".join(lines)


def format_suite_results(all_results: Mapping[str, Mapping[str, Any]]) -> str:
    if not all_results:
        return "No suite results were produced."

    lines = []
    for suite_name, results in all_results.items():
        lines.append(f"{suite_name}:")
        lines.append(format_results(results))
    return "\n".join(lines)


def format_model_suite_summary(all_results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    if not all_results:
        return "No model suite results were produced."

    lines = []
    for model_name, suite_results in all_results.items():
        lines.append(f"{model_name}:")
        lines.append(format_suite_results(suite_results))
    return "\n".join(lines)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run lm_eval tasks for one or more LLaMA checkpoints.")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_max_length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    if args.tasks:
        all_results = evaluate_model_configs(
            tasks=args.tasks,
            batch_size=args.batch_size,
            device=args.device,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            model_max_length=args.model_max_length,
            dtype=dtype,
        )
        print(format_summary(all_results))
        return

    all_results = evaluate_model_configs_suites(
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        model_max_length=args.model_max_length,
        dtype=dtype,
    )
    print(format_model_suite_summary(all_results))


if __name__ == "__main__":
    main()
