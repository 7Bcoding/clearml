#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 1：整卡多机 —— ClearML launch_multi_node（无 Volcano gang）

- 本机提交 1 次；master Pod 内 launch_multi_node(N) 再入队 N-1 子 Task
- MASTER_ADDR 由 ClearML SDK 注入（需 Helm 配 CLEARML_MULTI_NODE_MASTER_DEF_ADDR）
- 整卡：走 multinode-full-gpu 队列，不要 connect(VGPU)

前置与操作见 MULTINODE_schemes_zh.md §方案 1

用法:
  python train_launch_multinode_wholecard.py --num-nodes 2
  python train_launch_multinode_wholecard.py --num-nodes 2 --queue multinode-full-gpu
"""
import argparse
import os

from clearml import Task

DEFAULT_QUEUE = "multinode-full-gpu"
_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)


def _run_smoke(task: Task, logger) -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
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
    args = parser.parse_args()

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes 至少为 1")

    task = Task.init(project_name="volcano-vgpu", task_name="multinode-launch-wholecard")
    task.set_tags(["multinode", "wholecard", "launch-multi-node", "smoke"])

    task.connect(
        {"total_num_nodes": args.num_nodes, "queue": args.queue, "master_port": args.master_port},
        name="launch_multi_node",
    )

    if Task.running_locally():
        task.execute_remotely(queue_name=args.queue)

    # ===== 以下仅在集群 Pod 内执行 =====
    task.launch_multi_node(
        total_num_nodes=args.num_nodes,
        port=args.master_port,
        queue=args.queue,
        wait=args.wait,
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
    _run_smoke(task, logger)


if __name__ == "__main__":
    main()
