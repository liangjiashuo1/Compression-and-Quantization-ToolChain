# coding=utf-8
"""
Build a task-aligned calibration_samples.json for low-rank PTQ.

This builder targets the current project task mix:
1. WikiText2
2. 8 commonsense tasks
3. MMLU

It can generate a plain stratified-random calibration set or apply
COLA-style activation clustering over a larger candidate pool.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import data_utils


def _allocate_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    return data_utils._allocate_counts(total, weights)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _choice_key_from_index(index: int) -> str:
    return chr(65 + int(index))


def _resolve_choice_index(answer_value: Any, num_choices: int) -> Optional[int]:
    if answer_value is None:
        return None
    if isinstance(answer_value, bool):
        return None
    if isinstance(answer_value, int):
        return answer_value if 0 <= answer_value < num_choices else None
    answer_str = str(answer_value).strip()
    if not answer_str:
        return None
    if answer_str.isdigit():
        digit = int(answer_str)
        if 0 <= digit < num_choices:
            return digit
        if 1 <= digit <= num_choices:
            return digit - 1
        return None
    if len(answer_str) == 1 and answer_str.upper().isalpha():
        idx = ord(answer_str.upper()) - ord("A")
        return idx if 0 <= idx < num_choices else None
    return None


def _pack_multiple_choice_sample(
    question: str,
    choices: Sequence[str],
    prefix: Optional[str] = None,
    answer_value: Any = None,
) -> Dict[str, Any]:
    lines = []
    if prefix:
        lines.append(prefix.strip())
    lines.append(f"Question:\n{question.strip()}")
    lines.append(
        "Choices:\n"
        + "\n".join(
            f"{_choice_key_from_index(idx)}. {str(choice).strip()}"
            for idx, choice in enumerate(choices)
        )
    )
    lines.append("Answer:")

    answer_idx = _resolve_choice_index(answer_value, len(choices))
    answer_key = _choice_key_from_index(answer_idx) if answer_idx is not None else None
    answer = str(choices[answer_idx]).strip() if answer_idx is not None else None
    if answer is not None and answer_key is not None:
        lines.append(f"{answer_key}. {answer}")
    elif answer is not None:
        lines.append(answer)

    return {
        "text": "\n\n".join(lines),
        "answer": answer,
        "answer_key": answer_key,
    }


def _dedupe_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for sample in samples:
        text = sample.get("text") if isinstance(sample, dict) else None
        if not isinstance(text, str):
            continue
        normalized = _normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(sample)
    return unique


def _filter_samples_by_token_length(
    samples: Sequence[Dict[str, Any]],
    tokenizer,
    min_length: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    filtered = []
    for sample in samples:
        text = sample.get("text") if isinstance(sample, dict) else None
        if not isinstance(text, str):
            continue
        token_count = len(tokenizer.encode(text))
        if token_count < min_length:
            continue
        if max_length > 0 and token_count > max_length:
            continue
        filtered.append(sample)
    return filtered


def _sample_records(
    records: Sequence[Dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    records = list(records)
    if target_count <= 0 or not records:
        return []
    if len(records) <= target_count:
        return records
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[idx] for idx in indices[:target_count]]


def _load_wikitext2_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    return [{"text": text} for text in data_utils._load_wikitext2_train_texts(cache_dir=cache_dir)]


def _load_arc_candidate_samples(config_name, cache_dir=None) -> List[Dict[str, Any]]:
    dataset = data_utils._load_hf_split(
        "ai2_arc",
        config_name=config_name,
        split_candidates=("train", "validation", "test"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        question = example.get("question", "")
        choices = example.get("choices", {})
        choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
        answer_value = example.get("answerKey")
        if question and choice_texts:
            samples.append(
                _pack_multiple_choice_sample(question, choice_texts, answer_value=answer_value)
            )
    return samples


def _load_hellaswag_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    try:
        dataset = data_utils._load_hf_split(
            "hellaswag",
            config_name="default",
            split_candidates=("train", "validation"),
            cache_dir=cache_dir,
        )
    except Exception:
        dataset = data_utils._load_cached_hellaswag_dataset(
            split_candidates=("train", "validation"),
            cache_dir=cache_dir,
        )
        if dataset is None:
            raise
    samples = []
    for example in dataset:
        question = " ".join(
            part.strip()
            for part in [
                example.get("activity_label", ""),
                example.get("ctx_a", ""),
                example.get("ctx_b", ""),
            ]
            if isinstance(part, str) and part.strip()
        )
        endings = example.get("endings", [])
        answer_value = example.get("label")
        if question and endings:
            samples.append(
                _pack_multiple_choice_sample(
                    question,
                    endings,
                    prefix="Choose the most plausible continuation.",
                    answer_value=answer_value,
                )
            )
    return samples


def _load_winogrande_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    dataset = data_utils._load_hf_split(
        "winogrande",
        config_name="winogrande_xl",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        sentence = example.get("sentence", "")
        options = [example.get("option1", ""), example.get("option2", "")]
        answer_value = example.get("answer")
        if sentence and all(options):
            samples.append(
                _pack_multiple_choice_sample(sentence, options, answer_value=answer_value)
            )
    return samples


def _load_piqa_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    dataset = data_utils._load_hf_split(
        "piqa",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        goal = example.get("goal", "")
        options = [example.get("sol1", ""), example.get("sol2", "")]
        answer_value = example.get("label")
        if goal and all(options):
            samples.append(
                _pack_multiple_choice_sample(
                    goal,
                    options,
                    prefix="Select the better solution.",
                    answer_value=answer_value,
                )
            )
    return samples


def _load_boolq_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    dataset = data_utils._load_hf_split(
        "boolq",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        question = example.get("question", "")
        passage = example.get("passage", "")
        answer_value = example.get("answer")
        if question and passage:
            answer_key = None
            answer = None
            if answer_value is not None:
                answer = "yes" if bool(answer_value) else "no"
                answer_key = answer
            samples.append(
                {
                    "text": "\n\n".join(
                        [
                            f"Passage:\n{passage.strip()}",
                            f"Question:\n{str(question).strip()}",
                            f"Answer: {answer}" if answer is not None else "Answer:",
                        ]
                    ),
                    "answer": answer,
                    "answer_key": answer_key,
                }
            )
    return samples


def _load_openbookqa_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    dataset = data_utils._load_hf_split(
        "openbookqa",
        config_name="main",
        split_candidates=("train", "validation", "test"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        question = example.get("question_stem", "")
        choices = example.get("choices", {})
        choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
        answer_value = example.get("answerKey")
        if question and choice_texts:
            samples.append(
                _pack_multiple_choice_sample(question, choice_texts, answer_value=answer_value)
            )
    return samples


def _load_social_iqa_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    dataset = data_utils._load_hf_split(
        "social_i_qa",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        context = example.get("context", "")
        question = example.get("question", "")
        choices = [
            example.get("answerA", ""),
            example.get("answerB", ""),
            example.get("answerC", ""),
        ]
        answer_value = example.get("label")
        if context and question and all(choices):
            samples.append(
                _pack_multiple_choice_sample(
                    question,
                    choices,
                    prefix=f"Context:\n{context.strip()}",
                    answer_value=answer_value,
                )
            )
    return samples


def _load_local_mmlu_candidate_samples() -> Optional[List[Dict[str, Any]]]:
    try:
        data_dir = data_utils.resolve_mmlu_data_dir(REPO_ROOT)
    except FileNotFoundError:
        return None

    for split in ("auxiliary_train", "dev"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue

        suffix = f"_{split}.csv"
        samples = []
        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith(suffix):
                continue
            import pandas as pd

            df = pd.read_csv(os.path.join(split_dir, filename), header=None)
            for idx in range(df.shape[0]):
                question = df.iloc[idx, 0]
                choices = [df.iloc[idx, j + 1] for j in range(df.shape[1] - 2)]
                answer_value = df.iloc[idx, df.shape[1] - 1]
                samples.append(
                    _pack_multiple_choice_sample(question, choices, answer_value=answer_value)
                )
        if samples:
            return samples
    return None


def _load_mmlu_candidate_samples(cache_dir=None) -> List[Dict[str, Any]]:
    local_samples = _load_local_mmlu_candidate_samples()
    if local_samples is not None:
        return local_samples

    dataset = data_utils._load_hf_split(
        "cais/mmlu",
        config_name="all",
        split_candidates=("auxiliary_train", "dev", "validation"),
        cache_dir=cache_dir,
    )
    samples = []
    for example in dataset:
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer_value = example.get("answer")
        if question and choices:
            samples.append(
                _pack_multiple_choice_sample(question, choices, answer_value=answer_value)
            )
    return samples


def _load_commonsense_task_candidate_samples(task_name, cache_dir=None) -> List[Dict[str, Any]]:
    loaders = {
        "arc_easy": lambda: _load_arc_candidate_samples("ARC-Easy", cache_dir=cache_dir),
        "arc_challenge": lambda: _load_arc_candidate_samples("ARC-Challenge", cache_dir=cache_dir),
        "hellaswag": lambda: _load_hellaswag_candidate_samples(cache_dir=cache_dir),
        "winogrande": lambda: _load_winogrande_candidate_samples(cache_dir=cache_dir),
        "piqa": lambda: _load_piqa_candidate_samples(cache_dir=cache_dir),
        "boolq": lambda: _load_boolq_candidate_samples(cache_dir=cache_dir),
        "openbookqa": lambda: _load_openbookqa_candidate_samples(cache_dir=cache_dir),
        "social_iqa": lambda: _load_social_iqa_candidate_samples(cache_dir=cache_dir),
    }
    if task_name not in loaders:
        raise ValueError(f"Unsupported commonsense calibration task: {task_name}")
    return loaders[task_name]()


def _build_candidate_samples(
    tokenizer,
    seed: int,
    sequence_length: int,
    candidate_multiplier: int,
    num_samples: int,
    wikitext2_ratio: float,
    commonsense_ratio: float,
    mmlu_ratio: float,
    commonsense_tasks: Sequence[str],
    cache_dir=None,
    min_tokens: int = 128,
    max_tokens: int = 4096,
):
    rng = random.Random(seed)
    total_candidates = max(num_samples, int(num_samples * candidate_multiplier))
    domain_counts = _allocate_counts(
        total_candidates,
        {
            "wikitext2": wikitext2_ratio,
            "commonsense": commonsense_ratio,
            "mmlu": mmlu_ratio,
        },
    )

    candidate_samples = []

    wiki_samples = _load_wikitext2_candidate_samples(cache_dir=cache_dir)
    wiki_samples = _dedupe_samples(wiki_samples)
    wiki_samples = _filter_samples_by_token_length(
        wiki_samples,
        tokenizer=tokenizer,
        min_length=min_tokens,
        max_length=max_tokens,
    )
    for sample in _sample_records(wiki_samples, domain_counts["wikitext2"], rng):
        candidate_samples.append(
            {
                **sample,
                "dataset_name": "wikitext2",
                "source_group": "wikitext2",
            }
        )

    commonsense_counts = _allocate_counts(
        domain_counts["commonsense"],
        {task_name: 1.0 for task_name in commonsense_tasks},
    )
    for task_name in commonsense_tasks:
        samples = _load_commonsense_task_candidate_samples(task_name, cache_dir=cache_dir)
        samples = _dedupe_samples(samples)
        samples = _filter_samples_by_token_length(
            samples,
            tokenizer=tokenizer,
            min_length=min_tokens,
            max_length=max_tokens,
        )
        for sample in _sample_records(samples, commonsense_counts.get(task_name, 0), rng):
            candidate_samples.append(
                {
                    **sample,
                    "dataset_name": task_name,
                    "source_group": "commonsense",
                }
            )

    mmlu_samples = _load_mmlu_candidate_samples(cache_dir=cache_dir)
    mmlu_samples = _dedupe_samples(mmlu_samples)
    mmlu_samples = _filter_samples_by_token_length(
        mmlu_samples,
        tokenizer=tokenizer,
        min_length=min_tokens,
        max_length=max_tokens,
    )
    for sample in _sample_records(mmlu_samples, domain_counts["mmlu"], rng):
        candidate_samples.append(
            {
                **sample,
                "dataset_name": "mmlu",
                "source_group": "mmlu",
            }
        )

    rng.shuffle(candidate_samples)
    return candidate_samples


def _select_samples_random(candidate_samples, num_samples, seed):
    rng = random.Random(seed)
    indices = list(range(len(candidate_samples)))
    rng.shuffle(indices)
    chosen = [candidate_samples[idx] for idx in indices[:num_samples]]
    for idx, sample in enumerate(chosen):
        sample["selection_index"] = idx
        sample["selection_method"] = "stratified_random"
    return chosen


def _select_samples_activation_clustering(
    candidate_samples,
    num_samples,
    model_name_or_path,
    tokenizer,
    device,
    batch_size,
):
    from transformers import AutoModelForCausalLM

    cola_root = os.path.join(REPO_ROOT, "cola")
    if cola_root not in sys.path:
        sys.path.insert(0, cola_root)

    try:
        from cola.sample_selection import select_samples
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import COLA sample_selection. Expected local package under "
            f"{cola_root} or an installed 'cola' package."
        ) from exc

    if not model_name_or_path:
        raise ValueError(
            "model_name_or_path is required when selection_method=activation_clustering"
        )

    dtype = torch.float16
    if torch.cuda.is_available():
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return _select_samples_activation_clustering_with_model(
        candidate_samples=candidate_samples,
        num_samples=num_samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )


def _select_samples_activation_clustering_with_model(
    candidate_samples,
    num_samples,
    model,
    tokenizer,
    device,
    batch_size,
):
    cola_root = os.path.join(REPO_ROOT, "cola")
    if cola_root not in sys.path:
        sys.path.insert(0, cola_root)

    try:
        from cola.sample_selection import select_samples
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import COLA sample_selection. Expected local package under "
            f"{cola_root} or an installed 'cola' package."
        ) from exc

    selected = select_samples(
        processed_samples=candidate_samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_clusters=num_samples,
        reduced_dim=64,
        activation_layers=None,
        batch_size=batch_size,
        random_state=42,
    )
    return selected


def _select_samples_activation_clustering_preserve_source_ratios(
    candidate_samples,
    num_samples,
    model_name_or_path,
    tokenizer,
    device,
    batch_size,
    source_weights: Dict[str, float],
    seed: int,
):
    from transformers import AutoModelForCausalLM

    if not model_name_or_path:
        raise ValueError(
            "model_name_or_path is required when selection_method=activation_clustering"
        )

    dtype = torch.float16
    if torch.cuda.is_available():
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    quotas = _allocate_counts(num_samples, source_weights)
    grouped_candidates: Dict[str, List[Dict]] = {}
    for sample in candidate_samples:
        source_group = sample.get("source_group", "unknown")
        grouped_candidates.setdefault(source_group, []).append(sample)

    selected_samples: List[Dict] = []
    rng = random.Random(seed)

    for source_group, quota in quotas.items():
        if quota <= 0:
            continue
        group_samples = grouped_candidates.get(source_group, [])
        if len(group_samples) < quota:
            raise ValueError(
                f"Source group '{source_group}' only has {len(group_samples)} candidates, "
                f"fewer than requested quota={quota}. Increase candidate_multiplier or relax filters."
            )
        chosen_group = _select_samples_activation_clustering_with_model(
            candidate_samples=group_samples,
            num_samples=quota,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
        selected_samples.extend(chosen_group)

    rng.shuffle(selected_samples)
    selected_samples = selected_samples[:num_samples]
    for idx, sample in enumerate(selected_samples):
        sample["selection_index"] = idx
        sample["selection_method"] = "activation_clustering_preserve_source_ratios"
    return selected_samples


def _save_outputs(
    output_dir: str,
    samples: List[Dict],
    metadata: Dict,
):
    os.makedirs(output_dir, exist_ok=True)
    samples_path = os.path.join(output_dir, "calibration_samples.json")
    metadata_path = os.path.join(output_dir, "calibration_metadata.json")

    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return samples_path, metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a task-aligned calibration_samples.json for PTQ."
    )
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--candidate_multiplier", type=float, default=4.0)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_method",
        type=str,
        choices=["random", "activation_clustering"],
        default="random",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--wikitext2_ratio", type=float, default=0.30)
    parser.add_argument("--commonsense_ratio", type=float, default=0.40)
    parser.add_argument("--mmlu_ratio", type=float, default=0.30)
    parser.add_argument(
        "--commonsense_tasks",
        nargs="+",
        default=list(data_utils.COMMONSENSE_CALIBRATION_TASKS),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_model_name = args.model_name_or_path
    if tokenizer_model_name is None:
        raise ValueError(
            "Please provide --model_name_or_path so the builder can tokenize and "
            "optionally extract activations."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model_name,
        use_fast=False,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    candidate_samples = _build_candidate_samples(
        tokenizer=tokenizer,
        seed=args.seed,
        sequence_length=args.sequence_length,
        candidate_multiplier=args.candidate_multiplier,
        num_samples=args.num_samples,
        wikitext2_ratio=args.wikitext2_ratio,
        commonsense_ratio=args.commonsense_ratio,
        mmlu_ratio=args.mmlu_ratio,
        commonsense_tasks=args.commonsense_tasks,
        cache_dir=args.cache_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    if len(candidate_samples) < args.num_samples:
        raise ValueError(
            f"Only built {len(candidate_samples)} candidate samples, fewer than "
            f"requested num_samples={args.num_samples}. Relax filters or increase sources."
        )

    if args.selection_method == "random":
        selected_samples = _select_samples_random(
            candidate_samples,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        selected_samples = _select_samples_activation_clustering_preserve_source_ratios(
            candidate_samples=candidate_samples,
            num_samples=args.num_samples,
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            source_weights={
                "wikitext2": args.wikitext2_ratio,
                "commonsense": args.commonsense_ratio,
                "mmlu": args.mmlu_ratio,
            },
            seed=args.seed,
        )

    metadata = {
        "num_samples": args.num_samples,
        "candidate_count": len(candidate_samples),
        "sequence_length": args.sequence_length,
        "selection_method": args.selection_method,
        "ratio_preserved_in_final_selection": args.selection_method == "activation_clustering",
        "seed": args.seed,
        "ratios": {
            "wikitext2": args.wikitext2_ratio,
            "commonsense": args.commonsense_ratio,
            "mmlu": args.mmlu_ratio,
        },
        "commonsense_tasks": list(args.commonsense_tasks),
        "source_counts": {
            "wikitext2": sum(sample.get("source_group") == "wikitext2" for sample in selected_samples),
            "commonsense": sum(sample.get("source_group") == "commonsense" for sample in selected_samples),
            "mmlu": sum(sample.get("source_group") == "mmlu" for sample in selected_samples),
        },
        "dataset_counts": {},
    }

    dataset_counts = {}
    for sample in selected_samples:
        dataset_name = sample.get("dataset_name", "unknown")
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
    metadata["dataset_counts"] = dataset_counts

    samples_path, metadata_path = _save_outputs(
        output_dir=args.output_dir,
        samples=selected_samples,
        metadata=metadata,
    )

    print(f"Saved calibration samples to: {samples_path}")
    print(f"Saved calibration metadata to: {metadata_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
