#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 2：整卡多机 + Volcano PodGroup gang（Pod 内 NCCL 冒烟）

由 submit_multinode_podgroup.py 本地一次性入队 N 个 Task 触发；不使用 VGPU 段。

MASTER_ADDR 三种模式（Multinode/master_addr_mode）:
  task-poll  rank0 用 MY_POD_IP 写入 master Task；worker 轮询（默认）
  fixed      所有 rank 直接用 Multinode/master_addr（配合 hostNetwork + 节点 IP）
  service    所有 rank 直接用 Multinode/master_addr（Service DNS，见 k8s/service_multinode_master.example.yaml）

也可被 env 覆盖: MULTINODE_MASTER_ADDR（values 注入，常用于 service 模式）
"""
import argparse
import os
import time

from clearml import Task

_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)

RUNTIME_MASTER_IP = "multinode_smoke/master_pod_ip"
POLL_SEC = 180


def _read_multinode(task: Task) -> dict:
    section = task.get_parameters_as_dict().get("Multinode") or {}
    out = {}
    for key in (
        "nnodes",
        "node_rank",
        "master_task_id",
        "master_port",
        "queue",
        "master_addr_mode",
        "master_addr",
        "podgroup",
    ):
        entry = section.get(key)
        if isinstance(entry, dict):
            out[key] = entry.get("value", entry)
        elif entry is not None:
            out[key] = entry
    return out


def _publish_master_ip(task: Task, ip: str) -> None:
    task.set_parameter(name=RUNTIME_MASTER_IP, value=ip, value_type=str, description="rank0 pod IP for task-poll mode")


def _poll_master_ip(master_task_id: str) -> str:
    master = Task.get_task(task_id=master_task_id)
    deadline = time.time() + POLL_SEC
    while time.time() < deadline:
        # 必须 reload：get_parameter 读的是 task 本地缓存 self.data，
        # 不刷新会一直读 Task.get_task 那一刻的旧快照，永远看不到 rank0 后写入的 IP
        master.reload()
        ip = master.get_parameter(name=RUNTIME_MASTER_IP, default=None)
        if ip:
            return str(ip)
        time.sleep(2)
    raise RuntimeError("timeout waiting master pod IP on task %s (mode=task-poll)" % master_task_id)


def _resolve_master_addr(task: Task, mn: dict, node_rank: int) -> str:
    mode = str(mn.get("master_addr_mode") or "task-poll").strip()
    configured = str(mn.get("master_addr") or "").strip()
    env_addr = os.environ.get("MULTINODE_MASTER_ADDR", "").strip()
    master_task_id = str(mn.get("master_task_id") or "")

    if mode == "fixed":
        addr = env_addr or configured
        if not addr:
            raise RuntimeError("mode=fixed 需要 Multinode/master_addr 或 env MULTINODE_MASTER_ADDR")
        return addr

    if mode == "service":
        addr = env_addr or configured
        if not addr:
            raise RuntimeError("mode=service 需要 Multinode/master_addr 或 env MULTINODE_MASTER_ADDR")
        return addr

    # task-poll
    if node_rank == 0:
        addr = os.environ.get("MY_POD_IP", "").strip()
        if not addr:
            raise RuntimeError(
                "mode=task-poll rank0 需要 env MY_POD_IP（values 配 status.podIP downward API，见 values-multinode-full-gpu.example.yaml）"
            )
        _publish_master_ip(task, addr)
        return addr

    if not master_task_id:
        raise RuntimeError("Multinode/master_task_id missing; use submit_multinode_podgroup.py")
    return _poll_master_ip(master_task_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", choices=("nccl", "gloo"))
    args = parser.parse_args()

    task = Task.init(project_name="volcano-vgpu", task_name="multinode-podgroup-wholecard")
    mn = _read_multinode(task)
    nnodes = int(mn.get("nnodes", 2))
    node_rank = int(mn.get("node_rank", 0))
    master_port = int(mn.get("master_port") or 29500)
    mode = str(mn.get("master_addr_mode") or "task-poll")

    import torch
    import torch.distributed as dist

    master_addr = _resolve_master_addr(task, mn, node_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(node_rank)
    os.environ["WORLD_SIZE"] = str(nnodes)
    os.environ["LOCAL_RANK"] = "0"

    logger = task.get_logger() if node_rank == 0 else None
    if logger is not None:
        logger.report_text(
            "PodGroup gang smoke: mode=%s MASTER_ADDR=%s node_rank=%s nnodes=%s podgroup=%s"
            % (mode, master_addr, node_rank, nnodes, mn.get("podgroup"))
        )

    backend = args.backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=node_rank, world_size=nnodes)
    device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")

    t = torch.ones(1, device=device) * (node_rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(nnodes * (nnodes + 1) / 2)
    ok = abs(float(t.item()) - expected) < 1e-3

    if node_rank == 0 and logger is not None:
        logger.report_scalar("smoke", "allreduce_sum", float(t.item()), iteration=0)
        logger.report_single_value("allreduce_ok", 1.0 if ok else 0.0)
        logger.report_text("all_reduce sum=%s expected=%s ok=%s backend=%s mode=%s" % (t.item(), expected, ok, backend, mode))
        if not ok:
            raise RuntimeError("all_reduce mismatch")

    dist.destroy_process_group()
    print("rank=%s DONE ok=%s sum=%s mode=%s" % (node_rank, ok, float(t.item()), mode))


if __name__ == "__main__":
    main()
