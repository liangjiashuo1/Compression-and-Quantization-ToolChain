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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import datasets
import torch
import transformers


def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


COMMONSENSE_CALIBRATION_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "boolq",
    "openbookqa",
    "social_iqa",
)


def resolve_mmlu_data_dir(base_dir=None):
    candidates = []

    env_path = os.environ.get("MMLU_DATA_DIR")
    if env_path:
        candidates.append(env_path)

    if base_dir is not None:
        candidates.extend(
            [
                os.path.join(base_dir, "mmlu"),
                os.path.join(base_dir, "MMLU"),
                os.path.join(base_dir, "data", "mmlu"),
                os.path.join(base_dir, "data", "MMLU"),
            ]
        )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend(
        [
            os.path.join(repo_root, "mmlu"),
            os.path.join(repo_root, "MMLU"),
            os.path.join(repo_root, "data", "mmlu"),
            os.path.join(repo_root, "data", "MMLU"),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "MMLU data directory not found. Set MMLU_DATA_DIR or place the local "
        "dataset under one of: {}".format(", ".join(os.path.abspath(p) for p in candidates))
    )


def _allocate_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total <= 0 or not weights:
        return {key: 0 for key in weights}

    normalized = {key: max(float(value), 0.0) for key, value in weights.items()}
    weight_sum = sum(normalized.values())
    if weight_sum <= 0:
        equal_weight = 1.0 / len(normalized)
        normalized = {key: equal_weight for key in normalized}
    else:
        normalized = {key: value / weight_sum for key, value in normalized.items()}

    raw_counts = {key: total * value for key, value in normalized.items()}
    counts = {key: int(raw_counts[key]) for key in normalized}
    assigned = sum(counts.values())
    remainders = sorted(
        ((raw_counts[key] - counts[key], key) for key in normalized),
        reverse=True,
    )
    for _, key in remainders[: total - assigned]:
        counts[key] += 1
    return counts


def _sample_texts(texts: Sequence[str], max_examples: int, rng: random.Random) -> List[str]:
    cleaned = [text for text in texts if isinstance(text, str) and text.strip()]
    if len(cleaned) <= max_examples:
        return cleaned
    indices = list(range(len(cleaned)))
    rng.shuffle(indices)
    return [cleaned[idx] for idx in indices[:max_examples]]


def _pack_multiple_choice(question: str, choices: Iterable[str], prefix: Optional[str] = None) -> str:
    lines = []
    if prefix:
        lines.append(prefix.strip())
    lines.append(f"Question:\n{question.strip()}")
    lines.append(
        "Choices:\n"
        + "\n".join(
            f"{chr(65 + idx)}. {str(choice).strip()}" for idx, choice in enumerate(choices)
        )
    )
    lines.append("Answer:")
    return "\n\n".join(lines)


def _load_hf_split(dataset_name, config_name=None, split_candidates=("train", "validation", "test"), cache_dir=None):
    last_error = None
    for split in split_candidates:
        try:
            if config_name is None:
                return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            return datasets.load_dataset(
                dataset_name, config_name, split=split, cache_dir=cache_dir
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to load dataset={dataset_name}, config={config_name}, "
        f"splits={split_candidates}. Last error: {last_error}"
    )


def _iter_existing_dirs(paths: Sequence[str]) -> Iterable[str]:
    for path in paths:
        if path and os.path.isdir(path):
            yield path


def _resolve_hf_cache_roots(cache_dir=None) -> List[str]:
    roots = []
    if cache_dir:
        roots.append(cache_dir)
    roots.extend(
        [
            os.environ.get("HF_DATASETS_CACHE"),
            os.environ.get("HF_HOME"),
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            "/data/huggingface_model",
            os.path.expanduser("~/.cache/huggingface/datasets"),
        ]
    )

    normalized = []
    for root in roots:
        if not root:
            continue
        if root.endswith("/datasets"):
            normalized.append(root)
        elif os.path.basename(root) == "huggingface":
            normalized.append(os.path.join(root, "datasets"))
        else:
            normalized.append(root)

    seen = set()
    deduped = []
    for root in normalized:
        abs_root = os.path.abspath(root)
        if abs_root in seen:
            continue
        seen.add(abs_root)
        deduped.append(abs_root)
    return deduped


def _load_cached_arrow_or_parquet_dataset(dataset_dir: str, split_candidates: Sequence[str]):
    candidate_files = []
    for current_root, _, files in os.walk(dataset_dir):
        for filename in files:
            lower = filename.lower()
            if not (lower.endswith(".arrow") or lower.endswith(".parquet")):
                continue
            for split in split_candidates:
                if split in lower:
                    candidate_files.append(os.path.join(current_root, filename))
                    break

    if not candidate_files:
        return None

    datasets_by_split = {}
    for split in split_candidates:
        split_files = [path for path in candidate_files if split in os.path.basename(path).lower()]
        if not split_files:
            continue
        split_path = split_files[0]
        datasets_by_split[split] = _read_arrow_or_parquet_records(split_path)

    for split in split_candidates:
        if split in datasets_by_split:
            return datasets_by_split[split]
    return None


def _read_arrow_or_parquet_records(path: str):
    try:
        import pyarrow as pa
        import pyarrow.ipc as pa_ipc
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            f"pyarrow is required to read cached dataset file directly: {path}. "
            f"Original error: {exc}"
        ) from exc

    lower_path = path.lower()
    if lower_path.endswith(".parquet"):
        table = pq.read_table(path)
        return table.to_pylist()

    with pa.memory_map(path, "r") as source:
        try:
            reader = pa_ipc.open_stream(source)
            table = reader.read_all()
        except Exception:
            source.seek(0)
            reader = pa_ipc.open_file(source)
            table = reader.read_all()
    return table.to_pylist()


def _load_cached_hellaswag_dataset(split_candidates=("train", "validation"), cache_dir=None):
    cache_roots = list(_iter_existing_dirs(_resolve_hf_cache_roots(cache_dir=cache_dir)))
    dataset_dirs = []
    for root in cache_roots:
        dataset_dirs.extend(
            [
                os.path.join(root, "hellaswag", "default"),
                os.path.join(root, "hellaswag"),
            ]
        )

    for dataset_dir in _iter_existing_dirs(dataset_dirs):
        loaded = _load_cached_arrow_or_parquet_dataset(dataset_dir, split_candidates)
        if loaded is not None:
            return loaded
    return None


def _load_wikitext2_train_texts(cache_dir=None):
    traindata = datasets.load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        cache_dir=cache_dir,
    )
    return [text for text in traindata["text"] if text and text.strip()]


def _load_arc_texts(config_name, cache_dir=None):
    dataset = _load_hf_split(
        "ai2_arc",
        config_name=config_name,
        split_candidates=("train", "validation", "test"),
        cache_dir=cache_dir,
    )
    texts = []
    for example in dataset:
        question = example.get("question", "")
        choices = example.get("choices", {})
        choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
        if question and choice_texts:
            texts.append(_pack_multiple_choice(question, choice_texts))
    return texts


def _load_hellaswag_texts(cache_dir=None):
    try:
        dataset = _load_hf_split(
            "hellaswag",
            config_name="default",
            split_candidates=("train", "validation"),
            cache_dir=cache_dir,
        )
    except Exception:
        dataset = _load_cached_hellaswag_dataset(
            split_candidates=("train", "validation"),
            cache_dir=cache_dir,
        )
        if dataset is None:
            raise
    texts = []
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
        if question and endings:
            texts.append(_pack_multiple_choice(question, endings, prefix="Choose the most plausible continuation."))
    return texts


def _load_winogrande_texts(cache_dir=None):
    dataset = _load_hf_split(
        "winogrande",
        config_name="winogrande_xl",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    texts = []
    for example in dataset:
        sentence = example.get("sentence", "")
        options = [example.get("option1", ""), example.get("option2", "")]
        if sentence and all(options):
            texts.append(_pack_multiple_choice(sentence, options))
    return texts


def _load_piqa_texts(cache_dir=None):
    dataset = _load_hf_split(
        "piqa",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    texts = []
    for example in dataset:
        goal = example.get("goal", "")
        options = [example.get("sol1", ""), example.get("sol2", "")]
        if goal and all(options):
            texts.append(_pack_multiple_choice(goal, options, prefix="Select the better solution."))
    return texts


def _load_boolq_texts(cache_dir=None):
    dataset = _load_hf_split(
        "boolq",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    texts = []
    for example in dataset:
        question = example.get("question", "")
        passage = example.get("passage", "")
        if question and passage:
            texts.append(
                "\n\n".join(
                    [
                        f"Passage:\n{passage.strip()}",
                        f"Question:\n{str(question).strip()}",
                        "Answer:",
                    ]
                )
            )
    return texts


def _load_openbookqa_texts(cache_dir=None):
    dataset = _load_hf_split(
        "openbookqa",
        config_name="main",
        split_candidates=("train", "validation", "test"),
        cache_dir=cache_dir,
    )
    texts = []
    for example in dataset:
        question = example.get("question_stem", "")
        choices = example.get("choices", {})
        choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
        if question and choice_texts:
            texts.append(_pack_multiple_choice(question, choice_texts))
    return texts


def _load_social_iqa_texts(cache_dir=None):
    dataset = _load_hf_split(
        "social_i_qa",
        split_candidates=("train", "validation"),
        cache_dir=cache_dir,
    )
    texts = []
    for example in dataset:
        context = example.get("context", "")
        question = example.get("question", "")
        choices = [example.get("answerA", ""), example.get("answerB", ""), example.get("answerC", "")]
        if context and question and all(choices):
            texts.append(
                _pack_multiple_choice(
                    question,
                    choices,
                    prefix=f"Context:\n{context.strip()}",
                )
            )
    return texts


def _pack_mmlu_example(question, choices):
    return _pack_multiple_choice(question, choices)


def _load_local_mmlu_texts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        data_dir = resolve_mmlu_data_dir(base_dir)
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
            import pandas as pd

            df = pd.read_csv(os.path.join(split_dir, filename), header=None)
            for idx in range(df.shape[0]):
                question = df.iloc[idx, 0]
                choices = [df.iloc[idx, j + 1] for j in range(df.shape[1] - 2)]
                texts.append(_pack_mmlu_example(question, choices))
        if texts:
            return texts
    return None


def _load_mmlu_texts(cache_dir=None):
    local_texts = _load_local_mmlu_texts()
    if local_texts is not None:
        return local_texts

    try:
        dataset = _load_hf_split(
            "cais/mmlu",
            config_name="all",
            split_candidates=("auxiliary_train", "dev", "validation"),
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load MMLU texts. Either set MMLU_DATA_DIR to a local dataset "
            "root containing split folders such as auxiliary_train/dev, or make sure "
            "Hugging Face can access 'cais/mmlu'. Original error: {}".format(exc)
        ) from exc
    texts = []
    for example in dataset:
        question = example.get("question", "")
        choices = example.get("choices", [])
        if question and choices:
            texts.append(_pack_mmlu_example(question, choices))
    return texts


def _load_commonsense_task_texts(task_name, cache_dir=None):
    loaders = {
        "arc_easy": lambda: _load_arc_texts("ARC-Easy", cache_dir=cache_dir),
        "arc_challenge": lambda: _load_arc_texts("ARC-Challenge", cache_dir=cache_dir),
        "hellaswag": lambda: _load_hellaswag_texts(cache_dir=cache_dir),
        "winogrande": lambda: _load_winogrande_texts(cache_dir=cache_dir),
        "piqa": lambda: _load_piqa_texts(cache_dir=cache_dir),
        "boolq": lambda: _load_boolq_texts(cache_dir=cache_dir),
        "openbookqa": lambda: _load_openbookqa_texts(cache_dir=cache_dir),
        "social_iqa": lambda: _load_social_iqa_texts(cache_dir=cache_dir),
    }
    if task_name not in loaders:
        raise ValueError(f"Unsupported commonsense calibration task: {task_name}")
    return loaders[task_name]()


def _sample_sequences_from_corpus(
    tot_text,
    tokenizer,
    nsamples,
    seqlen,
    rng,
):
    trainloader = []
    if not tot_text or not tot_text.strip():
        return trainloader

    while len(trainloader) < nsamples:
        i = rng.randint(0, max(0, len(tot_text) - seqlen - 1))
        j = min(len(tot_text), i + seqlen * 10)
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            if len(tot_text) <= seqlen * 10:
                trainenc = tokenizer(tot_text, return_tensors="pt", truncation=True, max_length=seqlen)
                if trainenc.input_ids.shape[1] == 0:
                    break
                inp = trainenc.input_ids
                if inp.shape[1] < seqlen:
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    pad = torch.full((inp.shape[0], seqlen - inp.shape[1]), pad_token_id, dtype=inp.dtype)
                    inp = torch.cat([inp, pad], dim=1)
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp[:, :seqlen], tar[:, :seqlen]))
                continue
            continue
        inp = trainenc.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_wikitext2_commonsense_mmlu_calibration_loader(
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    cache_dir=None,
    wikitext2_ratio=0.34,
    commonsense_ratio=0.33,
    mmlu_ratio=0.33,
    commonsense_tasks: Optional[Sequence[str]] = None,
):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if commonsense_tasks is None:
        commonsense_tasks = COMMONSENSE_CALIBRATION_TASKS

    rng = random.Random(seed)
    domain_counts = _allocate_counts(
        nsamples,
        {
            "wikitext2": wikitext2_ratio,
            "commonsense": commonsense_ratio,
            "mmlu": mmlu_ratio,
        },
    )

    trainloader = []

    if domain_counts["wikitext2"] > 0:
        wiki_texts = _sample_texts(
            _load_wikitext2_train_texts(cache_dir=cache_dir),
            max_examples=4096,
            rng=rng,
        )
        trainloader.extend(
            _sample_sequences_from_corpus(
                "\n\n".join(wiki_texts),
                tokenizer,
                domain_counts["wikitext2"],
                seqlen,
                random.Random(rng.randint(0, 2**31 - 1)),
            )
        )

    if domain_counts["commonsense"] > 0 and commonsense_tasks:
        task_counts = _allocate_counts(
            domain_counts["commonsense"],
            {task_name: 1.0 for task_name in commonsense_tasks},
        )
        for task_name, task_count in task_counts.items():
            if task_count <= 0:
                continue
            task_texts = _sample_texts(
                _load_commonsense_task_texts(task_name, cache_dir=cache_dir),
                max_examples=2048,
                rng=rng,
            )
            trainloader.extend(
                _sample_sequences_from_corpus(
                    "\n\n".join(task_texts),
                    tokenizer,
                    task_count,
                    seqlen,
                    random.Random(rng.randint(0, 2**31 - 1)),
                )
            )

    if domain_counts["mmlu"] > 0:
        mmlu_texts = _sample_texts(
            _load_mmlu_texts(cache_dir=cache_dir),
            max_examples=4096,
            rng=rng,
        )
        trainloader.extend(
            _sample_sequences_from_corpus(
                "\n\n".join(mmlu_texts),
                tokenizer,
                domain_counts["mmlu"],
                seqlen,
                random.Random(rng.randint(0, 2**31 - 1)),
            )
        )

    rng.shuffle(trainloader)
    return trainloader[:nsamples]


def resolve_cola_calibration_path(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("COLA_CALIBRATION_PATH")
    if env_path:
        candidates.append(env_path)

    env_output_dir = os.environ.get("COLA_OUTPUT_DIR")
    if env_output_dir:
        candidates.append(os.path.join(env_output_dir, "calibration_samples.json"))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend(
        [
            os.path.join(repo_root, "cola_output", "calibration_samples.json"),
            os.path.join(repo_root, "cola", "cola_output", "calibration_samples.json"),
        ]
    )

    checked = []
    for path in candidates:
        if not path:
            continue
        norm_path = os.path.abspath(path)
        checked.append(norm_path)
        if os.path.isfile(norm_path):
            return norm_path

    raise FileNotFoundError(
        "COLA calibration_samples.json not found. Set COLA_CALIBRATION_PATH or "
        "COLA_OUTPUT_DIR, or place the file at one of: {}".format(", ".join(checked))
    )


def _pad_or_trim_input_ids(inp: torch.Tensor, seqlen: int, pad_token_id: int) -> torch.Tensor:
    if inp.dim() == 1:
        inp = inp.unsqueeze(0)
    elif inp.dim() != 2:
        raise ValueError(f"Expected input_ids to be 1D or 2D, got shape {tuple(inp.shape)}")

    if inp.shape[1] > seqlen:
        return inp[:, :seqlen]
    if inp.shape[1] == seqlen:
        return inp

    pad_width = seqlen - inp.shape[1]
    pad = torch.full((inp.shape[0], pad_width), pad_token_id, dtype=inp.dtype)
    return torch.cat([inp, pad], dim=1)


def get_cola_calibration_loader(
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    calibration_path=None,
):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    resolved_path = resolve_cola_calibration_path(calibration_path)
    print(f"Using COLA calibration file: {resolved_path}")
    import json

    with open(resolved_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        raise ValueError(f"No calibration samples found in {resolved_path}")

    rng = random.Random(seed)
    ordered_indices = list(range(len(samples)))
    rng.shuffle(ordered_indices)
    ordered_samples = [samples[idx] for idx in ordered_indices]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    trainloader = []
    for sample_idx in range(nsamples):
        sample = ordered_samples[sample_idx % len(ordered_samples)]
        if "input_ids" in sample and sample["input_ids"] is not None:
            inp = torch.tensor(sample["input_ids"], dtype=torch.long)
        elif "text" in sample:
            inp = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seqlen,
            ).input_ids
        else:
            raise KeyError(
                "Each COLA calibration sample must contain either 'input_ids' or 'text'."
            )

        inp = _pad_or_trim_input_ids(inp, seqlen=seqlen, pad_token_id=pad_token_id)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_channel_selection_calibration_loader(
    calibration_source="wikitext2",
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    cache_dir=None,
    wikitext2_ratio=0.34,
    commonsense_ratio=0.33,
    mmlu_ratio=0.33,
    commonsense_tasks: Optional[Sequence[str]] = None,
    calibration_path=None,
):
    if calibration_source == "wikitext2":
        return get_wikitext2(
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            model=model,
            tokenizer=tokenizer,
            eval_mode=False,
        )

    if calibration_source == "wikitext2_commonsense_mmlu":
        return get_wikitext2_commonsense_mmlu_calibration_loader(
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            model=model,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            wikitext2_ratio=wikitext2_ratio,
            commonsense_ratio=commonsense_ratio,
            mmlu_ratio=mmlu_ratio,
            commonsense_tasks=commonsense_tasks,
        )

    if calibration_source == "cola_json":
        return get_cola_calibration_loader(
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            model=model,
            tokenizer=tokenizer,
            calibration_path=calibration_path,
        )

    raise ValueError(
        f"Unsupported channel-selection calibration source: {calibration_source}"
    )


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
