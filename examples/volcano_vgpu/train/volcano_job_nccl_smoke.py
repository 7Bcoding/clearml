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
import os

import torch
import torch.distributed as dist


def _logger():
    task_id = os.environ.get("CLEARML_TASK_ID", "").strip()
    if not task_id:
        return None
    try:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        return task.get_logger()
    except Exception as exc:
        print("WARN: ClearML logging disabled: %s" % exc)
        return None


def main():
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
    require_cuda = os.environ.get("REQUIRE_CUDA", "1").lower() not in ("0", "false", "no")

    if not master_addr:
        raise RuntimeError("MASTER_ADDR empty; Volcano Job svc 插件或 env 未配置")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check runtimeClassName/CDI/vGPU injection")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(node_rank)
    os.environ["WORLD_SIZE"] = str(nnodes)
    os.environ["LOCAL_RANK"] = "0"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(0)
        print("CUDA device: %s" % torch.cuda.get_device_name(0))
    dist.init_process_group(backend=backend, rank=node_rank, world_size=nnodes)
    device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")

    t = torch.ones(1, device=device) * (node_rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(nnodes * (nnodes + 1) / 2)
    ok = abs(float(t.item()) - expected) < 1e-3

    logger = _logger() if node_rank == 0 else None
    if logger is not None:
        logger.report_scalar("smoke", "allreduce_sum", float(t.item()), iteration=0)
        logger.report_single_value("allreduce_ok", 1.0 if ok else 0.0)
        logger.report_text(
            "Volcano Job smoke: sum=%s expected=%s ok=%s MASTER_ADDR=%s NODE_RANK=%s"
            % (t.item(), expected, ok, master_addr, node_rank)
        )

    dist.destroy_process_group()
    print("rank=%s DONE ok=%s sum=%s MASTER_ADDR=%s" % (node_rank, ok, float(t.item()), master_addr))
    if not ok:
        raise RuntimeError("all_reduce mismatch")


if __name__ == "__main__":
    main()
