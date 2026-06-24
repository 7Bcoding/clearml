#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 3：Volcano Job 多机 DDP 训练模板（每 Pod 1 进程 / 1 张 vGPU）

由 submit/submit_volcano_job.py 或等价 Volcano Job 拉起。
平台负责注入：
  - NNODES
  - MASTER_ADDR / MASTER_PORT
  - VK_TASK_INDEX 或 VC_TASK_INDEX（节点 rank）
  - CLEARML_TASK_ID（可选，rank0 弱集成写指标）

示例：
  python submit/submit_volcano_job.py --num-nodes 2 --apply \
    --script volcano_job_ddp_train.py \
    --script-args "--epochs 5 --batch-size 128"
"""
import argparse
import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


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


def _node_rank() -> int:
    value = (
        os.environ.get("VK_TASK_INDEX")
        or os.environ.get("VC_TASK_INDEX")
        or os.environ.get("NODE_RANK")
        or os.environ.get("RANK")
    )
    if value is None:
        raise RuntimeError("rank env missing; expected VK_TASK_INDEX/VC_TASK_INDEX from Volcano env plugin")
    return int(value)


def _clearml_handles(rank: int):
    task_id = os.environ.get("CLEARML_TASK_ID", "").strip()
    if rank != 0 or not task_id:
        return None, None
    try:
        from clearml import OutputModel, Task

        task = Task.get_task(task_id=task_id)
        return task.get_logger(), OutputModel(task=task, framework="pytorch", name="volcano-job-ddp")
    except Exception as exc:
        print("WARN: ClearML logging disabled: %s" % exc)
        return None, None


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _make_dataset(samples: int, input_dim: int, num_classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, input_dim, generator=generator)
    teacher = torch.randn(input_dim, num_classes, generator=generator)
    logits = x @ teacher
    y = torch.argmax(logits, dim=1)
    return TensorDataset(x, y)


def main():
    parser = argparse.ArgumentParser(description="Volcano Job DDP training template")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128, help="per-rank batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="/tmp/volcano-job-ddp")
    parser.add_argument("--dist-timeout-sec", type=int, default=1800)
    parser.add_argument("--nccl-socket-ifname", default="", help="Pod network interface for NCCL, usually eth0")
    parser.add_argument("--nccl-debug", default="", help="NCCL_DEBUG override")
    parser.add_argument("--nccl-debug-subsys", default="", help="NCCL_DEBUG_SUBSYS override")
    parser.add_argument("--nccl-ib-disable", default="", help="NCCL_IB_DISABLE override, default 1")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", action="store_false", dest="require_cuda")
    args = parser.parse_args()

    _apply_dist_env(args)

    nnodes = int(os.environ.get("NNODES", "2"))
    rank = _node_rank()
    world_size = nnodes
    master_addr = os.environ.get("MASTER_ADDR", "").strip()
    master_port = os.environ.get("MASTER_PORT", "29500").strip()

    if not master_addr:
        raise RuntimeError("MASTER_ADDR empty; Volcano Job svc plugin or env is not configured")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check runtimeClassName/CDI/vGPU injection")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        cuda_count = torch.cuda.device_count()
        if cuda_count != 1:
            print("WARN: this training script is one process per Pod and uses cuda:0 only; visible GPUs=%s" % cuda_count)
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    print(
        "DDP start: rank=%s world_size=%s backend=%s MASTER_ADDR=%s MASTER_PORT=%s"
        % (rank, world_size, backend, master_addr, master_port)
    )
    if device.type == "cuda":
        print("CUDA device: %s" % torch.cuda.get_device_name(0))
    _print_dist_diagnostics(rank, world_size, backend, master_addr, master_port)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.dist_timeout_sec),
    )

    torch.manual_seed(args.seed + rank)
    dataset = _make_dataset(args.samples, args.input_dim, args.num_classes, args.seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyClassifier(args.input_dim, args.hidden, args.num_classes).to(device)
    if device.type == "cuda":
        model = DDP(model, device_ids=[0], output_device=0)
    else:
        model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    logger, output_model = _clearml_handles(rank)
    if logger is not None:
        logger.report_text(
            "Volcano Job DDP training: world_size=%s batch_size=%s samples=%s"
            % (world_size, args.batch_size, args.samples)
        )

    last_global_loss = 0.0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(target.numel())
            total_correct += int((output.argmax(dim=1) == target).sum().item())
            total_seen += int(target.numel())

        metrics = torch.tensor(
            [total_loss, float(total_correct), float(total_seen)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        global_loss = float(metrics[0].item() / max(metrics[2].item(), 1.0))
        global_acc = float(metrics[1].item() / max(metrics[2].item(), 1.0))
        last_global_loss = global_loss

        if rank == 0 and logger is not None:
            logger.report_scalar("train", "loss", global_loss, iteration=epoch)
            logger.report_scalar("train", "accuracy", global_acc, iteration=epoch)
        print(
            "rank=%s epoch=%s loss=%.6f acc=%.4f local_seen=%s"
            % (rank, epoch, global_loss, global_acc, total_seen)
        )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, "model.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        if output_model is not None:
            output_model.update_weights(ckpt_path)
        if logger is not None:
            logger.report_single_value("final_loss", last_global_loss)
            logger.report_text("checkpoint=%s" % ckpt_path)
        print("rank=0 checkpoint=%s" % ckpt_path)

    dist.destroy_process_group()
    print("rank=%s DONE" % rank)


if __name__ == "__main__":
    main()
