#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volcano vGPU 全链路冒烟测试 (调度 + 显存隔离 + ClearML 回传)

本地提交:
    python smoke_test_vgpu.py

仅验证平台, 不测 torch 分配时可注释掉 Task.add_requirements 整段。
"""
from __future__ import annotations

import subprocess
import sys
import time

from pathlib import Path

from clearml import Task

from vgpu import connect_vgpu, expected_memory_mib, GPU_MEMORY_FACTOR

Task.add_requirements(str(Path(__file__).with_name("requirements-remote.txt")))

DEFAULT_QUEUE = "volcano-queue"
DEFAULT_VGPU_MEMORY_GIB = 2
DEFAULT_VGPU_CORES = 30


def main() -> None:
    task = Task.init(
        project_name="volcano-vgpu",
        task_name="smoke-test",
        task_type=Task.TaskTypes.testing,
    )

    connect_vgpu(
        task,
        vgpu_number=1,
        vgpu_memory=DEFAULT_VGPU_MEMORY_GIB,
        vgpu_cores=DEFAULT_VGPU_CORES,
    )

    task.execute_remotely(queue_name=DEFAULT_QUEUE, exit_process=True)

    logger = task.get_logger()

    print("=" * 60)
    print("[1/3] nvidia-smi")
    print("=" * 60)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(result.stdout or result.stderr)

    print("=" * 60)
    print("[2/3] 显存上限")
    print("=" * 60)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(query.stdout)
    try:
        mem_total_mb = float(query.stdout.split(",")[1].strip().split()[0])
        logger.report_scalar("gpu", "memory_total_mb", mem_total_mb, iteration=0)
        expected = expected_memory_mib(DEFAULT_VGPU_MEMORY_GIB, GPU_MEMORY_FACTOR)
        print(
            "GPU memory limit: %.0f MiB (expect ~ %s MiB = %s GiB)"
            % (mem_total_mb, expected, DEFAULT_VGPU_MEMORY_GIB)
        )
    except Exception as exc:
        print(f"parse nvidia-smi failed: {exc}")

    print("=" * 60)
    print("[3/3] torch 分配测试")
    print("=" * 60)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")

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
