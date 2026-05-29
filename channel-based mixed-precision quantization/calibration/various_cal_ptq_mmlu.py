# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unified ASVD low-rank PTQ + MMLU/commonsense evaluation entrypoint.

This file replaces the duplicated ptq_mmlu_cal_{wiki,mmlu,mixed,cola}_svdllm
entrypoints. Use --calibration_source to select the calibration implementation
and --rot_before_ptq to apply the low-rank bottleneck rotation before PTQ.
"""

import argparse
import datetime
import importlib
import sys
from logging import Logger

import torch
import torch.distributed as dist

from ptq_split_mmlu_common import (
    build_tokenizer,
    load_lowrank_model,
    log_gpu_mem,
    reset_cuda_memory_stats,
    run_standard_evaluations,
)
from utils import utils
from utils.process_args import process_args_ptq


log: Logger = utils.get_logger("spinquant")

PTQ_MODEL_MODULES = {
    "wiki": "eval_utils.main",
    "mmlu": "eval_utils.main_mmlu_svdllm",
    "mixed": "eval_utils.main_mixed_svdllm",
    "cola": "eval_utils.main_cola_svdllm",
}


def parse_various_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--calibration_source",
        "--calibration-source",
        "--calib_source",
        "--calib-source",
        choices=sorted(PTQ_MODEL_MODULES),
        default="wiki",
        help="Calibration implementation to use for PTQ.",
    )
    parser.add_argument(
        "--rot_before_ptq",
        "--rot-before-ptq",
        action="store_true",
        help="Apply offline low-rank rotation before running PTQ.",
    )
    parser.add_argument(
        "--no_rot_before_ptq",
        "--no-rot-before-ptq",
        action="store_false",
        dest="rot_before_ptq",
        help="Disable offline low-rank rotation before PTQ.",
    )

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def load_ptq_model_fn(calibration_source):
    module_name = PTQ_MODEL_MODULES[calibration_source]
    module = importlib.import_module(module_name)
    return module.ptq_model


def maybe_apply_lowrank_rotation(model, enabled):
    if not enabled:
        return model
    from ptq_split_mmlu_common_rot_before_ptq import apply_lowrank_rotation_offline

    model = apply_lowrank_rotation_offline(model)
    log_gpu_mem("after offline_lowrank_rotation")
    return model


def train() -> None:
    various_args = parse_various_args()
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    ptq_model = load_ptq_model_fn(various_args.calibration_source)

    log.info("the rank is {}".format(local_rank))
    log.info(
        "PTQ calibration_source=%s, rot_before_ptq=%s",
        various_args.calibration_source,
        various_args.rot_before_ptq,
    )
    torch.distributed.barrier()

    model = load_lowrank_model(model_args, training_args, ptq_args)

    reset_cuda_memory_stats()
    model.cuda()
    log_gpu_mem("after model.cuda")

    model = maybe_apply_lowrank_rotation(model, various_args.rot_before_ptq)

    model = ptq_model(ptq_args, model, model_args)
    log_gpu_mem("after ptq_model")

    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")

    tokenizer = build_tokenizer(model_args, training_args)
    log.info("Complete tokenizer loading...")

    if local_rank == 0:
        run_standard_evaluations(model, tokenizer, ptq_args, __file__)

    dist.barrier()


if __name__ == "__main__":
    train()
