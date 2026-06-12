#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volcano vGPU 全链路冒烟测试 (调度 + 显存隔离 + torch 分配 + ClearML 回传)

本地提交:
    python smoke_test_vgpu.py
    python smoke_test_vgpu.py --vgpu-memory 2 --vgpu-cores 30
    python smoke_test_vgpu.py --vgpu-number 2 --vgpu-memory 2 --vgpu-cores 30

仅验证平台、不测 torch 分配:
    python smoke_test_vgpu.py --skip-allocation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from pathlib import Path

from clearml import Task

from vgpu import connect_vgpu, expected_memory_mib, GPU_MEMORY_FACTOR

Task.add_requirements(str(Path(__file__).with_name("requirements-remote.txt")))

DEFAULT_QUEUE = "volcano-queue"
DEFAULT_VGPU_NUMBER = 1
DEFAULT_VGPU_MEMORY_GIB = 2
DEFAULT_VGPU_CORES = 30
MEMORY_TOLERANCE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volcano vGPU smoke test")
    parser.add_argument("--queue", default=DEFAULT_QUEUE, help="ClearML queue")
    parser.add_argument("--vgpu-number", type=int, default=DEFAULT_VGPU_NUMBER, help="volcano.sh/vgpu-number")
    parser.add_argument(
        "--vgpu-memory",
        type=int,
        default=DEFAULT_VGPU_MEMORY_GIB,
        help="volcano.sh/vgpu-memory in GiB (gpu-memory-factor=1024)",
    )
    parser.add_argument("--vgpu-cores", type=int, default=DEFAULT_VGPU_CORES, help="volcano.sh/vgpu-cores, 0-100")
    parser.add_argument(
        "--skip-allocation",
        action="store_true",
        help="skip torch incremental allocation / OOM test",
    )
    return parser.parse_args()


def count_gpus_nvidia_smi() -> int:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return -1
    return sum(1 for line in (result.stdout or "").splitlines() if line.strip().startswith("GPU "))


def main() -> None:
    args = parse_args()
    expected_mib = expected_memory_mib(args.vgpu_memory, GPU_MEMORY_FACTOR)

    task = Task.init(
        project_name="volcano-vgpu",
        task_name="smoke-test",
        task_type=Task.TaskTypes.testing,
    )

    connect_vgpu(
        task,
        vgpu_number=args.vgpu_number,
        vgpu_memory=args.vgpu_memory,
        vgpu_cores=args.vgpu_cores,
    )

    task.execute_remotely(queue_name=args.queue, exit_process=True)

    logger = task.get_logger()
    logger.report_text(
        "requested VGPU: number=%s memory_gib=%s cores=%s factor=%s"
        % (args.vgpu_number, args.vgpu_memory, args.vgpu_cores, GPU_MEMORY_FACTOR)
    )

    print("=" * 60)
    print("[1/4] nvidia-smi")
    print("=" * 60)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(result.stdout or result.stderr)

    print("=" * 60)
    print("[2/4] GPU 数量 (expect vgpu-number=%s)" % args.vgpu_number)
    print("=" * 60)
    smi_count = count_gpus_nvidia_smi()
    print("nvidia-smi -L GPU count:", smi_count)
    logger.report_scalar("gpu", "nvidia_smi_gpu_count", smi_count, iteration=0)
    list_result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    print(list_result.stdout or list_result.stderr)

    print("=" * 60)
    print("[3/4] 显存上限 (expect ~ %s MiB = %s GiB)" % (expected_mib, args.vgpu_memory))
    print("=" * 60)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(query.stdout)
    try:
        first_line = (query.stdout or "").strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        mem_total_mb = float(parts[1])
        logger.report_scalar("gpu", "memory_total_mb", mem_total_mb, iteration=0)
        print(
            "GPU 0 memory limit: %.0f MiB (expect ~ %s MiB = %s GiB)"
            % (mem_total_mb, expected_mib, args.vgpu_memory)
        )
        if abs(mem_total_mb - expected_mib) > expected_mib * MEMORY_TOLERANCE:
            print("WARNING: memory limit differs from request; check gpu-memory-factor / VGPU section")
    except Exception as exc:
        print(f"parse nvidia-smi failed: {exc}")

    print("=" * 60)
    print("[4/4] torch 分配测试")
    print("=" * 60)
    if args.skip_allocation:
        print("skipped (--skip-allocation)")
    else:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")

            torch_count = torch.cuda.device_count()
            print("torch.cuda.device_count():", torch_count)
            logger.report_scalar("gpu", "torch_device_count", torch_count, iteration=0)
            if torch_count != args.vgpu_number:
                print(
                    "WARNING: torch device count %s != requested vgpu-number %s"
                    % (torch_count, args.vgpu_number)
                )

            dev = torch.device("cuda:0")
            print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

            allocated = []
            step_mb = 512
            for i in range(64):
                try:
                    num_elements = step_mb * 1024 * 1024 // 4
                    allocated.append(torch.empty(num_elements, dtype=torch.float32, device=dev))
                    used_mb = torch.cuda.memory_allocated(dev) / 1024 / 1024
                    logger.report_scalar("gpu", "torch_allocated_mb", used_mb, iteration=i)
                    print(f"allocated {used_mb:.0f} MB")
                    time.sleep(0.2)
                except RuntimeError as exc:
                    used_mb = torch.cuda.memory_allocated(dev) / 1024 / 1024
                    print(f"OOM at {used_mb:.0f} MB (expected): {str(exc)[:200]}")
                    logger.report_scalar("gpu", "oom_at_mb", used_mb, iteration=0)
                    break
            allocated.clear()
            torch.cuda.empty_cache()
        except ImportError:
            print("torch not installed, skip allocation test")
            sys.exit(1)

    for i in range(5):
        logger.report_scalar("heartbeat", "value", i, iteration=i)
        time.sleep(1)

    print("ALL DONE")


if __name__ == "__main__":
    main()
