#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 3：Volcano Job 整卡多机 NCCL 冒烟（Pod 内入口）

由 Volcano Job 拉起；不经过 ClearML k8s-glue。
- gang / MASTER_ADDR 由 Volcano Job + svc 插件提供
- NODE_RANK 来自 Volcano env 插件 VK_TASK_INDEX 或 VC_TASK_INDEX
- rank0 若设 CLEARML_TASK_ID，则向已有 Task 写 scalar（弱集成）

见 MULTINODE_schemes_zh.md §方案 3
"""
import argparse
import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist


def _apply_dist_env(args):
    socket_ifname = args.nccl_socket_ifname or os.environ.get("NCCL_SOCKET_IFNAME") or "eth0"
    nccl_debug = args.nccl_debug or os.environ.get("NCCL_DEBUG") or "INFO"
    nccl_debug_subsys = args.nccl_debug_subsys or os.environ.get("NCCL_DEBUG_SUBSYS") or "INIT,NET"
    nccl_ib_disable = args.nccl_ib_disable or os.environ.get("NCCL_IB_DISABLE") or "1"

    os.environ["NCCL_SOCKET_IFNAME"] = socket_ifname
    os.environ.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
    os.environ["NCCL_DEBUG"] = nccl_debug
    os.environ["NCCL_DEBUG_SUBSYS"] = nccl_debug_subsys
    os.environ["NCCL_IB_DISABLE"] = nccl_ib_disable


def _run_diag(label, cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError:
        print("%s: command not found: %s" % (label, cmd[0]))
        return
    except Exception as exc:
        print("%s: %s" % (label, exc))
        return

    output = (result.stdout or result.stderr or "").strip()
    if output:
        print("%s:\n%s" % (label, output))
    else:
        print("%s: <empty> rc=%s" % (label, result.returncode))


def _print_dist_diagnostics(rank, world_size, backend, master_addr, master_port):
    keys = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "GLOO_SOCKET_IFNAME",
    ]
    print(
        "distributed env: rank=%s world_size=%s backend=%s %s"
        % (rank, world_size, backend, " ".join("%s=%s" % (k, os.environ.get(k)) for k in keys))
    )
    _run_diag("hostname", ["hostname"])
    _run_diag("hostname -I", ["hostname", "-I"])
    _run_diag("getent hosts MASTER_ADDR", ["getent", "hosts", master_addr])
    _run_diag("ip -br addr", ["ip", "-br", "addr"])
    _run_diag("ip route", ["ip", "route"])


def _clearml_task(rank):
    task_id = os.environ.get("CLEARML_TASK_ID", "").strip()
    if rank != 0 or not task_id:
        return None, None
    try:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        task.mark_started(force=True)
        logger = task.get_logger()
        logger.report_text("Volcano Job rank0 attached to ClearML Task %s" % task_id)
        return task, logger
    except Exception as exc:
        print("WARN: ClearML logging disabled: %s" % exc)
        return None, None


def _mark_clearml_completed(task, message):
    if task is None:
        return
    try:
        task.flush(wait_for_uploads=True)
        task.mark_completed(force=True, status_message=message)
    except Exception as exc:
        print("WARN: ClearML mark_completed failed: %s" % exc)


def _mark_clearml_failed(task, exc):
    if task is None:
        return
    try:
        task.get_logger().report_text("FAILED: %s" % exc)
        task.flush(wait_for_uploads=True)
        task.mark_failed(force=True, status_reason="failed", status_message=str(exc))
    except Exception as mark_exc:
        print("WARN: ClearML mark_failed failed: %s" % mark_exc)


def main():
    parser = argparse.ArgumentParser(description="Volcano Job NCCL all_reduce smoke")
    parser.add_argument("--dist-timeout-sec", type=int, default=int(os.environ.get("DIST_TIMEOUT_SEC", "1800")))
    parser.add_argument("--nccl-socket-ifname", default="", help="Pod network interface for NCCL, usually eth0")
    parser.add_argument("--nccl-debug", default="", help="NCCL_DEBUG override")
    parser.add_argument("--nccl-debug-subsys", default="", help="NCCL_DEBUG_SUBSYS override")
    parser.add_argument("--nccl-ib-disable", default="", help="NCCL_IB_DISABLE override, default 1")
    parser.add_argument("--allow-cpu", action="store_true", help="allow gloo CPU fallback")
    args = parser.parse_args()

    _apply_dist_env(args)

    nnodes = int(os.environ.get("NNODES", "2"))
    rank_text = (
        os.environ.get("VK_TASK_INDEX")
        or os.environ.get("VC_TASK_INDEX")
        or os.environ.get("NODE_RANK")
        or os.environ.get("RANK")
    )
    if rank_text is None:
        raise RuntimeError("rank env missing; expected VK_TASK_INDEX/VC_TASK_INDEX from Volcano env plugin")
    node_rank = int(rank_text)
    master_addr = os.environ.get("MASTER_ADDR", "").strip()
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    require_cuda = (
        os.environ.get("REQUIRE_CUDA", "1").lower() not in ("0", "false", "no")
        and not args.allow_cpu
    )

    if not master_addr:
        raise RuntimeError("MASTER_ADDR empty; Volcano Job svc 插件或 env 未配置")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(node_rank)
    os.environ["WORLD_SIZE"] = str(nnodes)
    os.environ["LOCAL_RANK"] = "0"

    clearml_task, logger = _clearml_task(node_rank)
    process_group_ready = False
    try:
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; check runtimeClassName/CDI/vGPU injection")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            cuda_count = torch.cuda.device_count()
            if cuda_count != 1:
                print("WARN: this smoke script is one process per Pod and uses cuda:0 only; visible GPUs=%s" % cuda_count)
            torch.cuda.set_device(0)
            print("CUDA device: %s" % torch.cuda.get_device_name(0))
        _print_dist_diagnostics(node_rank, nnodes, backend, master_addr, master_port)

        dist.init_process_group(
            backend=backend,
            rank=node_rank,
            world_size=nnodes,
            timeout=timedelta(seconds=args.dist_timeout_sec),
        )
        process_group_ready = True
        device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")

        t = torch.ones(1, device=device) * (node_rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(nnodes * (nnodes + 1) / 2)
        ok = abs(float(t.item()) - expected) < 1e-3

        if logger is not None:
            logger.report_scalar("smoke", "allreduce_sum", float(t.item()), iteration=0)
            logger.report_single_value("allreduce_ok", 1.0 if ok else 0.0)
            logger.report_text(
                "Volcano Job smoke: sum=%s expected=%s ok=%s MASTER_ADDR=%s NODE_RANK=%s"
                % (t.item(), expected, ok, master_addr, node_rank)
            )

        print("rank=%s DONE ok=%s sum=%s MASTER_ADDR=%s" % (node_rank, ok, float(t.item()), master_addr))
        if not ok:
            raise RuntimeError("all_reduce mismatch")
        _mark_clearml_completed(clearml_task, "Volcano Job smoke completed")
    except Exception as exc:
        _mark_clearml_failed(clearml_task, exc)
        raise
    finally:
        if process_group_ready:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
