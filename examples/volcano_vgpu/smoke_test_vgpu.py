#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volcano vGPU 全链路冒烟测试 (调度 + 显存隔离 + 多卡 torch OOM + ClearML 回传)

本地提交:
    python smoke_test_vgpu.py --vgpu-memory 2 --vgpu-cores 30
    python smoke_test_vgpu.py --vgpu-number 2 --vgpu-memory 2 --vgpu-cores 30
    python smoke_test_vgpu.py --vgpu-number 2 --vgpu-memory 4 --vgpu-cores 50

仅验证平台、不测 torch 分配:
    python smoke_test_vgpu.py --vgpu-number 2 --skip-allocation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from clearml import Task

from vgpu import (
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

DEFAULT_QUEUE = "volcano-queue"
DEFAULT_VGPU_NUMBER = 1
DEFAULT_VGPU_MEMORY_GIB = 2
DEFAULT_VGPU_CORES = 30
MEMORY_TOLERANCE = 0.15
DEFAULT_ALLOC_STEP_MB = 512
MAX_ALLOC_STEPS = 64
# PyTorch 2.x + CUDA 12 on vGPU: non-tensor reserve is roughly fixed (~300–512 MiB), not proportional to quota.
CUDA_FRAMEWORK_RESERVE_MIB = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volcano vGPU smoke test (multi-GPU OOM)")
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
        help="skip per-GPU torch incremental allocation / OOM test",
    )
    parser.add_argument(
        "--alloc-step-mb",
        type=int,
        default=DEFAULT_ALLOC_STEP_MB,
        help="torch allocation chunk size per step (MiB)",
    )
    add_remote_repo_args(parser)
    return parser.parse_args()


def count_gpus_nvidia_smi() -> int:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return -1
    return sum(1 for line in (result.stdout or "").splitlines() if line.strip().startswith("GPU "))


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


def oom_within_tolerance(oom_mib: float, expected_mib: float, step_mb: int = DEFAULT_ALLOC_STEP_MB) -> bool:
    """Check torch OOM landed near the vGPU limit (not at full-card capacity).

    Model (works across quota sizes when ``step_mb`` is reasonable vs ``expected_mib``):
    - ``memory.total`` ([3/5]) already validates the scheduled quota.
    - Incremental ``torch.empty`` steps report the last *successful* tensor total before OOM.
    - CUDA/PyTorch holds a mostly fixed non-tensor reserve on first GPU use.
    - The next ``step_mb`` chunk fails when ``tensor + reserve > expected``.

    So a healthy OOM is roughly in ``[expected - step - reserve, expected]`` (tensor side).
    """
    if expected_mib <= 0 or step_mb <= 0:
        return False
    lower = max(0.0, expected_mib - step_mb - CUDA_FRAMEWORK_RESERVE_MIB)
    upper = expected_mib
    return lower <= oom_mib <= upper


def run_torch_oom_on_device(torch, device_index: int, step_mb: int, logger, tag: str) -> tuple[float, bool]:
    """Incrementally allocate on one GPU until OOM. Returns (oom_at_mb, did_oom)."""
    dev = torch.device("cuda:%s" % device_index)
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)

    print("GPU %s: %s" % (device_index, torch.cuda.get_device_name(device_index)))
    allocated = []
    oom_at_mb = 0.0
    did_oom = False

    for step in range(MAX_ALLOC_STEPS):
        try:
            num_elements = step_mb * 1024 * 1024 // 4
            allocated.append(torch.empty(num_elements, dtype=torch.float32, device=dev))
            used_mb = torch.cuda.memory_allocated(dev) / 1024 / 1024
            logger.report_scalar(tag, "torch_allocated_mb", used_mb, iteration=step)
            print("  GPU %s step %s: allocated %.0f MiB" % (device_index, step, used_mb))
            time.sleep(0.1)
        except RuntimeError as exc:
            oom_at_mb = torch.cuda.memory_allocated(dev) / 1024 / 1024
            did_oom = True
            print("  GPU %s OOM at %.0f MiB: %s" % (device_index, oom_at_mb, str(exc)[:160]))
            logger.report_scalar(tag, "oom_at_mb", oom_at_mb, iteration=0)
            break

    allocated.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)
    return oom_at_mb, did_oom


def main() -> None:
    args = parse_args()
    expected_mib = expected_memory_mib(args.vgpu_memory, GPU_MEMORY_FACTOR)
    failures: list[str] = []

    apply_standalone_preflight(args)

    task = Task.init(
        project_name="volcano-vgpu",
        task_name="smoke-test",
        task_type=Task.TaskTypes.testing,
        auto_connect_frameworks=False,
    )

    connect_vgpu(
        task,
        vgpu_number=args.vgpu_number,
        vgpu_memory=args.vgpu_memory,
        vgpu_cores=args.vgpu_cores,
    )

    pin_remote_packages(task, __file__)
    prepare_remote_repo(task, args)
    task.flush(wait_for_uploads=True)
    task.execute_remotely(queue_name=args.queue, exit_process=True)

    logger = task.get_logger()
    logger.report_text(
        "requested VGPU: number=%s memory_gib=%s cores=%s factor=%s"
        % (args.vgpu_number, args.vgpu_memory, args.vgpu_cores, GPU_MEMORY_FACTOR)
    )

    print("=" * 60)
    print("[1/5] nvidia-smi")
    print("=" * 60)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(result.stdout or result.stderr)

    print("=" * 60)
    print("[2/5] GPU 数量 (expect vgpu-number=%s)" % args.vgpu_number)
    print("=" * 60)
    smi_count = count_gpus_nvidia_smi()
    print("nvidia-smi -L GPU count:", smi_count)
    logger.report_scalar("gpu", "nvidia_smi_gpu_count", smi_count, iteration=0)
    list_result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    print(list_result.stdout or list_result.stderr)

    if smi_count != args.vgpu_number:
        failures.append(
            "nvidia-smi GPU count %s != requested vgpu-number %s" % (smi_count, args.vgpu_number)
        )

    print("=" * 60)
    print("[3/5] 每卡显存上限 (expect ~ %s MiB = %s GiB per GPU)" % (expected_mib, args.vgpu_memory))
    print("=" * 60)
    try:
        mem_values = query_gpu_memory_mib()
        if len(mem_values) < args.vgpu_number:
            failures.append(
                "nvidia-smi reports %s GPUs but vgpu-number=%s" % (len(mem_values), args.vgpu_number)
            )
        for index, mem_mb in enumerate(mem_values[: args.vgpu_number]):
            tag = "gpu%s" % index
            logger.report_scalar(tag, "memory_total_mb", mem_mb, iteration=0)
            ok = memory_within_tolerance(mem_mb, expected_mib)
            status = "OK" if ok else "FAIL"
            print("GPU %s memory.total = %.0f MiB [%s]" % (index, mem_mb, status))
            if not ok:
                failures.append(
                    "GPU %s memory %.0f MiB != expected ~%s MiB" % (index, mem_mb, expected_mib)
                )
    except Exception as exc:
        failures.append("nvidia-smi memory query failed: %s" % exc)

    print("=" * 60)
    print("[4/5] 每卡 torch OOM (step=%s MiB, expect OOM near %s MiB)" % (args.alloc_step_mb, expected_mib))
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
                failures.append(
                    "torch device count %s != requested vgpu-number %s" % (torch_count, args.vgpu_number)
                )

            print("torch %s" % torch.__version__)
            for gpu_index in range(args.vgpu_number):
                tag = "gpu%s" % gpu_index
                print("-" * 40)
                oom_at_mb, did_oom = run_torch_oom_on_device(
                    torch, gpu_index, args.alloc_step_mb, logger, tag
                )
                if not did_oom:
                    failures.append(
                        "GPU %s: no OOM after %s x %s MiB (limit may not be enforced)"
                        % (gpu_index, MAX_ALLOC_STEPS, args.alloc_step_mb)
                    )
                elif not oom_within_tolerance(oom_at_mb, expected_mib, args.alloc_step_mb):
                    failures.append(
                        "GPU %s: OOM at %.0f MiB, expected near %s MiB"
                        % (gpu_index, oom_at_mb, expected_mib)
                    )
                else:
                    print("GPU %s OOM OK at %.0f MiB (expected ~%s MiB)" % (gpu_index, oom_at_mb, expected_mib))

        except ImportError:
            print("torch not installed, skip allocation test")
            sys.exit(1)

    print("=" * 60)
    print("[5/5] heartbeat")
    print("=" * 60)
    for i in range(5):
        logger.report_scalar("heartbeat", "value", i, iteration=i)
        time.sleep(1)

    if failures:
        logger.report_text("VALIDATION FAILED:\n" + "\n".join(failures))
        print("=" * 60)
        print("VALIDATION FAILED")
        for item in failures:
            print(" -", item)
        sys.exit(1)

    logger.report_text("VALIDATION PASSED")
    print("VALIDATION PASSED")
    print("ALL DONE")


if __name__ == "__main__":
    main()
