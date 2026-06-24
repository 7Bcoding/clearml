#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDK 侧按任务指定 Volcano vGPU (gpu-memory-factor=1024, 单位 GiB).

前提: agentk8sglue.vgpuHook.enabled=true

    python verify/vgpu_per_task_test.py --memory 2 --cores 30
    python verify/vgpu_per_task_test.py --memory 6 --cores 50
    python verify/vgpu_per_task_test.py --vgpu-number 2 --memory 2 --cores 30

Agent 无 SSH 密钥时用 --standalone (helm 单文件脚本) 或 HTTPS repo URL。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from clearml import Task

from vgpu_helpers import (
    add_remote_repo_args,
    apply_standalone_preflight,
    connect_vgpu,
    expected_memory_mib,
    GPU_MEMORY_FACTOR,
    pin_remote_packages,
    prepare_remote_repo,
    register_remote_requirements,
)

register_remote_requirements(__file__)

MEMORY_TOLERANCE = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default="volcano-queue")
    p.add_argument("--vgpu-number", type=int, default=1, help="volcano.sh/vgpu-number")
    p.add_argument(
        "--memory",
        type=int,
        default=2,
        help="volcano.sh/vgpu-memory in GiB (device-plugin gpu-memory-factor=1024)",
    )
    p.add_argument("--cores", type=int, default=30, help="volcano.sh/vgpu-cores, 0-100")
    p.add_argument(
        "--skip-torch",
        action="store_true",
        help="skip torch.cuda.device_count() check (nvidia-smi only)",
    )
    add_remote_repo_args(p)
    return p.parse_args()


def count_gpus_nvidia_smi() -> int:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("nvidia-smi -L failed:", result.stderr or result.stdout)
        return -1
    return sum(1 for line in (result.stdout or "").splitlines() if line.strip().startswith("GPU "))


def count_gpus_torch() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed, skip torch device count")
        return -1
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False")
        return 0
    return torch.cuda.device_count()


def query_gpu_memory_mib() -> list[float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "nvidia-smi query failed")

    values: list[float] = []
    for line in (result.stdout or "").strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        values.append(float(parts[1]))
    if not values:
        raise RuntimeError("no GPU memory.total rows in nvidia-smi output")
    return values


def memory_within_tolerance(actual_mib: float, expected_mib: float) -> bool:
    if expected_mib <= 0:
        return False
    return abs(actual_mib - expected_mib) <= expected_mib * MEMORY_TOLERANCE


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    apply_standalone_preflight(args)

    task = Task.init(
        project_name="volcano-vgpu",
        task_name="per-task-vgpu-test",
        task_type=Task.TaskTypes.testing,
        auto_connect_frameworks=False,
    )

    connect_vgpu(
        task,
        vgpu_number=args.vgpu_number,
        vgpu_memory=args.memory,
        vgpu_cores=args.cores,
    )

    pin_remote_packages(task, __file__)
    prepare_remote_repo(task, args)
    task.execute_remotely(queue_name=args.queue, exit_process=True)

    logger = task.get_logger()
    expected_mib = expected_memory_mib(args.memory, GPU_MEMORY_FACTOR)
    logger.report_text(
        "requested VGPU: number=%s memory_gib=%s cores=%s factor=%s"
        % (args.vgpu_number, args.memory, args.cores, GPU_MEMORY_FACTOR)
    )

    print("=" * 60)
    print("[1/3] nvidia-smi")
    print("=" * 60)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(result.stdout or result.stderr)

    print("=" * 60)
    print("[2/3] GPU 数量 (expect vgpu-number=%s)" % args.vgpu_number)
    print("=" * 60)
    smi_count = count_gpus_nvidia_smi()
    print("nvidia-smi -L GPU count:", smi_count)
    logger.report_scalar("gpu", "nvidia_smi_gpu_count", smi_count, iteration=0)

    torch_count = -1
    if not args.skip_torch:
        torch_count = count_gpus_torch()
        print("torch.cuda.device_count():", torch_count)
        logger.report_scalar("gpu", "torch_device_count", torch_count, iteration=0)

    if smi_count != args.vgpu_number:
        msg = "nvidia-smi GPU count %s != requested vgpu-number %s" % (smi_count, args.vgpu_number)
        print("FAIL:", msg)
        failures.append(msg)
    elif smi_count < 0:
        failures.append("could not determine GPU count from nvidia-smi -L")

    if not args.skip_torch and torch_count >= 0 and torch_count != args.vgpu_number:
        msg = "torch device count %s != requested vgpu-number %s" % (torch_count, args.vgpu_number)
        print("FAIL:", msg)
        failures.append(msg)

    list_result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    print(list_result.stdout or list_result.stderr)

    print("=" * 60)
    print("[3/3] 显存上限 (expect ~ %s MiB = %s GiB per GPU)" % (expected_mib, args.memory))
    print("=" * 60)
    try:
        mem_values = query_gpu_memory_mib()
        for index, mem_mb in enumerate(mem_values):
            logger.report_scalar("gpu", "memory_total_mb", mem_mb, iteration=index)
            ok = memory_within_tolerance(mem_mb, expected_mib)
            status = "OK" if ok else "WARN"
            print("GPU %s memory.total = %.0f MiB [%s]" % (index, mem_mb, status))
            if not ok:
                msg = "GPU %s memory %.0f MiB differs from expected %s MiB" % (index, mem_mb, expected_mib)
                print("WARNING:", msg)
                failures.append(msg)
    except Exception as exc:
        print("parse failed:", exc)
        failures.append(str(exc))

    if failures:
        logger.report_text("VALIDATION FAILED:\n" + "\n".join(failures))
        print("=" * 60)
        print("VALIDATION FAILED")
        for item in failures:
            print(" -", item)
        sys.exit(1)

    logger.report_text("VALIDATION PASSED")
    print("VALIDATION PASSED")
    print("DONE")


if __name__ == "__main__":
    main()
