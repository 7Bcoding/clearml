#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 1：整卡多机 —— ClearML launch_multi_node（无 Volcano gang）

- 本机提交 1 次；master Pod 内 launch_multi_node(N) 再入队 N-1 子 Task
- MASTER_ADDR 由 ClearML SDK 注入（需 Helm 配 CLEARML_MULTI_NODE_MASTER_DEF_ADDR）
- 资源按集群类型自适应（见 MULTINODE_schemes_zh.md §0.4）：
  · 原生 nvidia.com/gpu 集群：Agent vgpuHook 关，下方 VGPU 段被忽略（无害）
  · HAMi/vGPU 集群：Agent vgpuHook 开，按 VGPU 段注入每 Pod 一张卡
    （vgpu_cores=100 + 整卡显存 = 整卡；更小的值 = 切片）

前置与操作见 MULTINODE_schemes_zh.md §方案 1

用法:
  python train_launch_multinode_wholecard.py --num-nodes 2
  python train_launch_multinode_wholecard.py --num-nodes 2 --vgpu-cores 100 --vgpu-memory 24  # 整卡
  python train_launch_multinode_wholecard.py --num-nodes 2 --vgpu-cores 30 --vgpu-memory 4    # 切片
"""
import argparse
import os
from datetime import timedelta

from clearml import Task

DEFAULT_QUEUE = "multinode-full-gpu"
_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)


def _run_smoke(task: Task, logger, dist_timeout_sec: int, require_cuda: bool) -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(
        "distributed env: MASTER_ADDR=%s MASTER_PORT=%s NODE_RANK=%s RANK=%s WORLD_SIZE=%s"
        % (
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
            os.environ.get("NODE_RANK"),
            os.environ.get("RANK"),
            os.environ.get("WORLD_SIZE"),
        )
    )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check runtimeClassName/CDI/vGPU injection")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(0)
        print("CUDA device: %s" % torch.cuda.get_device_name(0))
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=dist_timeout_sec))
    device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")

    t = torch.ones(1, device=device) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) / 2)
    ok = abs(float(t.item()) - expected) < 1e-3

    if rank == 0 and logger is not None:
        logger.report_scalar("smoke", "allreduce_sum", float(t.item()), iteration=0)
        logger.report_single_value("allreduce_ok", 1.0 if ok else 0.0)
        logger.report_text(
            "launch_multi_node smoke: sum=%s expected=%s ok=%s MASTER_ADDR=%s RANK=%s WORLD_SIZE=%s"
            % (t.item(), expected, ok, os.environ.get("MASTER_ADDR"), rank, world_size)
        )
    dist.destroy_process_group()
    if not ok:
        raise RuntimeError("all_reduce mismatch")
    print("rank=%s DONE ok=%s" % (rank, ok))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=2, help="总节点数（含 rank0）")
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--wait", action="store_true", default=True, help="master 等待 worker Task 启动")
    parser.add_argument("--no-wait", action="store_false", dest="wait")
    parser.add_argument("--dist-timeout-sec", type=int, default=1800, help="PyTorch rendezvous 等待时间")
    parser.add_argument("--require-cuda", action="store_true", default=True, help="CUDA 不可用时直接失败")
    parser.add_argument("--allow-cpu", action="store_false", dest="require_cuda", help="允许退回 CPU/gloo，仅调试用")
    # vGPU 段（HAMi 集群由 vgpuHook 注入；原生 nvidia.com/gpu 集群忽略，无害）
    parser.add_argument("--vgpu-number", type=int, default=1, help="每 Pod 卡数（本脚本 1 进程/Pod，建议保持 1）")
    parser.add_argument("--vgpu-memory", type=int, default=24, help="每卡显存 GiB（整卡填整卡显存，如 4090=24）")
    parser.add_argument("--vgpu-cores", type=int, default=100, help="算力百分比（整卡=100，切片<100）")
    args = parser.parse_args()

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes 至少为 1")

    task = Task.init(project_name="volcano-vgpu", task_name="multinode-launch-wholecard")
    task.set_tags(["multinode", "wholecard", "launch-multi-node", "smoke"])

    task.connect(
        {"total_num_nodes": args.num_nodes, "queue": args.queue, "master_port": args.master_port},
        name="launch_multi_node",
    )

    # HAMi/vGPU 集群：vgpuHook 读此段，给每个 Pod（含 worker clone）注入 volcano.sh/vgpu-*
    # 原生 nvidia.com/gpu 集群（vgpuHook 关）会忽略此段，无害
    task.connect(
        {"vgpu_number": args.vgpu_number, "vgpu_memory": args.vgpu_memory, "vgpu_cores": args.vgpu_cores},
        name="VGPU",
    )

    if Task.running_locally():
        task.execute_remotely(queue_name=args.queue)

    # ===== 以下仅在集群 Pod 内执行 =====
    task.launch_multi_node(
        total_num_nodes=args.num_nodes,
        port=args.master_port,
        queue=args.queue,
        wait=args.wait,
        devices=1,
    )

    logger = task.get_logger() if int(os.environ.get("RANK", "0")) == 0 else None
    if logger is not None:
        logger.report_text(
            "node_rank=%s MASTER_ADDR=%s MASTER_PORT=%s RANK=%s WORLD_SIZE=%s"
            % (
                os.environ.get("NODE_RANK"),
                os.environ.get("MASTER_ADDR"),
                os.environ.get("MASTER_PORT"),
                os.environ.get("RANK"),
                os.environ.get("WORLD_SIZE"),
            )
        )
    _run_smoke(task, logger, args.dist_timeout_sec, args.require_cuda)


if __name__ == "__main__":
    main()
