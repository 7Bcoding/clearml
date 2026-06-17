#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 2：整卡多机 + Volcano PodGroup gang —— 本地一次性入队 N 个 ClearML Task

MASTER_ADDR 三种模式（写入 Multinode 段，由 train_multinode_podgroup.py 消费）:
  task-poll  rank0 写 Task 参数，worker 轮询（默认；需 values 配 MY_POD_IP）
  fixed      提交时指定 --master-addr <节点 IP>（配合 hostNetwork）
  service    提交时指定 --master-addr <Service DNS>（配合 k8s/service_multinode_master.example.yaml）

资源（HAMi/vGPU 集群）: --vgpu-number/memory/cores 写入每个 Task 的 VGPU 段，
  Agent vgpuHook 注入 volcano.sh/vgpu-*；须配套 *-vgpu* values / queue / podgroup
  （见 MULTINODE_schemes_zh.md §0.4）。原生 nvidia.com/gpu 集群（vgpuHook 关）会忽略该段。

前置与操作见 MULTINODE_schemes_zh.md §方案 2

用法:
  python submit_multinode_podgroup.py --num-nodes 2
  python submit_multinode_podgroup.py --num-nodes 2 --master-addr-mode fixed --master-addr 10.0.1.11
  python submit_multinode_podgroup.py --num-nodes 2 --master-addr-mode service \\
      --master-addr clearml-multinode-master.clearml.svc.cluster.local
"""
import argparse
import os
import uuid

from clearml import Task

DEFAULT_QUEUE = "multinode-full-gpu"
DEFAULT_PODGROUP = "clearml-gang-full-2"
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_SCRIPT = os.path.join(_HERE, "train_multinode_podgroup.py")
_REQS = os.path.join(_HERE, "requirements-remote.txt")

MASTER_ADDR_MODES = ("task-poll", "fixed", "service")


def _connect_multinode(
    task: Task,
    *,
    nnodes,
    node_rank,
    master_task_id,
    master_port,
    queue,
    master_addr_mode,
    master_addr,
    podgroup,
):
    task.connect(
        {
            "nnodes": nnodes,
            "node_rank": node_rank,
            "master_task_id": master_task_id,
            "master_port": master_port,
            "queue": queue,
            "master_addr_mode": master_addr_mode,
            "master_addr": master_addr or "",
            "podgroup": podgroup,
        },
        name="Multinode",
    )


def main():
    p = argparse.ArgumentParser(description="PodGroup gang: enqueue N ClearML tasks at once")
    p.add_argument("--num-nodes", type=int, default=2)
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--podgroup", default=DEFAULT_PODGROUP, help="须与 PodGroup CR 及 values 注解 group-name 一致")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument(
        "--master-addr-mode",
        choices=MASTER_ADDR_MODES,
        default="task-poll",
        help="MASTER_ADDR 解析方式",
    )
    p.add_argument(
        "--master-addr",
        default="",
        help="fixed: rank0 节点 IP; service: Headless Service DNS（见 k8s/service_multinode_master.example.yaml）",
    )
    p.add_argument("--project", default="volcano-vgpu")
    p.add_argument("--gang-id", default="")
    # vGPU 段（HAMi 集群由 vgpuHook 注入；原生 nvidia.com/gpu 集群忽略，无害）
    p.add_argument("--vgpu-number", type=int, default=1, help="每 Pod 卡数（1 进程/Pod，建议保持 1）")
    p.add_argument("--vgpu-memory", type=int, default=24, help="每卡显存 GiB（整卡填整卡显存，如 4090=24）")
    p.add_argument("--vgpu-cores", type=int, default=100, help="算力百分比（整卡=100，切片<100）")
    args = p.parse_args()

    if args.num_nodes < 2:
        raise SystemExit("PodGroup gang 冒烟至少 --num-nodes 2")
    if args.master_addr_mode in ("fixed", "service") and not args.master_addr.strip():
        raise SystemExit("--master-addr-mode %s 必须提供 --master-addr" % args.master_addr_mode)

    gang_id = args.gang_id or uuid.uuid4().hex[:8]
    base_name = "podgroup-gang-%s" % gang_id

    master = Task.create(
        project_name=args.project,
        task_name="%s-r0" % base_name,
        task_type=Task.TaskTypes.training,
        script=_TRAIN_SCRIPT,
        requirements_file=_REQS,
        add_task_init_call=False,
    )
    master.set_tags(
        ["multinode", "wholecard", "podgroup-gang", "smoke", "gang-%s" % gang_id, "rank-0", args.master_addr_mode]
    )

    clones = []
    for rank in range(1, args.num_nodes):
        node = Task.clone(source_task=master, name="%s-r%s" % (base_name, rank))
        node.set_tags(
            [
                "multinode",
                "wholecard",
                "podgroup-gang",
                "smoke",
                "gang-%s" % gang_id,
                "rank-%s" % rank,
                args.master_addr_mode,
            ]
        )
        clones.append(node)

    common = dict(
        nnodes=args.num_nodes,
        master_task_id=master.id,
        master_port=args.master_port,
        queue=args.queue,
        master_addr_mode=args.master_addr_mode,
        master_addr=args.master_addr.strip(),
        podgroup=args.podgroup,
    )
    _connect_multinode(master, node_rank=0, **common)
    for rank, node in enumerate(clones, start=1):
        _connect_multinode(node, node_rank=rank, **common)

    # VGPU 段：master 与各 clone 都要连（clone 时 master 尚无此段，故逐个连），
    # HAMi Agent 的 vgpuHook 按此段给每个 Pod 注入 volcano.sh/vgpu-*
    vgpu = {"vgpu_number": args.vgpu_number, "vgpu_memory": args.vgpu_memory, "vgpu_cores": args.vgpu_cores}
    for node in [master, *clones]:
        node.connect(vgpu, name="VGPU")

    print("=== PodGroup gang enqueue (gang_id=%s) ===" % gang_id)
    print("PodGroup: %s  minMember=%s  queue=%s" % (args.podgroup, args.num_nodes, args.queue))
    print("MASTER_ADDR mode: %s  addr=%s" % (args.master_addr_mode, args.master_addr or "(rank0 publishes)"))
    print("Master task (rank0): %s" % master.id)

    Task.enqueue(master, queue_name=args.queue)
    for node in clones:
        Task.enqueue(node, queue_name=args.queue)
        print("  worker task:", node.id)

    print("\nWatch: kubectl get podgroup %s -n clearml" % args.podgroup)
    print("       kubectl get pods -n clearml -o wide")
    print("WebUI: filter tag gang-%s ; rank0 scalar smoke/allreduce_sum" % gang_id)


if __name__ == "__main__":
    main()
