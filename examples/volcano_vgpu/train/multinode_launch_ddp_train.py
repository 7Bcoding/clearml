#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 1：ClearML launch_multi_node 多机 DDP 训练（无 Volcano gang）

本脚本是一 Pod 一进程 / 一张 vGPU 的保守模板：
  1. 本地提交 master Task
  2. master Pod 内调用 Task.launch_multi_node() 克隆并入队 worker Task
  3. ClearML SDK 注入 MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE
  4. 各 Pod 使用 torch.distributed + DDP 进行真实训练

用法：
  python train/multinode_launch_ddp_train.py --num-nodes 2 --queue multinode-full-gpu
  python train/multinode_launch_ddp_train.py --num-nodes 2 --epochs 5 --batch-size 128
  python train/multinode_launch_ddp_train.py --num-nodes 2 --vgpu-memory 24 --vgpu-cores 100

注意：
  - 这不是 gang 调度；worker 启动慢时请增大 --dist-timeout-sec。
  - 每个 Pod 只使用 1 张 GPU；多卡/每节点多进程建议走 torchrun 或 Volcano Job 路线。
"""
import argparse
import os
from datetime import timedelta

from clearml import OutputModel, Task

DEFAULT_QUEUE = "multinode-full-gpu"
_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)


def _run_training(args: argparse.Namespace, task: Task) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ.get("NODE_RANK", rank))

    print(
        "distributed env: MASTER_ADDR=%s MASTER_PORT=%s NODE_RANK=%s RANK=%s WORLD_SIZE=%s"
        % (
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
            node_rank,
            rank,
            world_size,
        )
    )

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check runtimeClassName/CDI/vGPU injection")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        print("CUDA device: %s" % torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.dist_timeout_sec),
    )

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

    generator = torch.Generator().manual_seed(args.seed)
    features = torch.randn(args.samples, args.input_dim, generator=generator)
    teacher = torch.randn(args.input_dim, args.num_classes, generator=generator)
    labels = torch.argmax(features @ teacher, dim=1)
    dataset = TensorDataset(features, labels)

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

    logger = None
    output_model = None
    if rank == 0:
        logger = task.get_logger()
        output_model = OutputModel(task=task, framework="pytorch", name="launch-multinode-ddp")
        logger.report_text(
            "launch_multi_node DDP training: world_size=%s batch_size=%s samples=%s"
            % (world_size, args.batch_size, args.samples)
        )

    last_global_loss = 0.0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_y.numel())
            total_correct += int((output.argmax(dim=1) == batch_y).sum().item())
            total_seen += int(batch_y.numel())

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
            task.upload_artifact("checkpoint", artifact_object=ckpt_path)
            logger.report_text("checkpoint=%s" % ckpt_path)
        print("rank=0 checkpoint=%s" % ckpt_path)

    dist.barrier()
    dist.destroy_process_group()
    print("rank=%s DONE" % rank)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClearML launch_multi_node DDP training")
    parser.add_argument("--num-nodes", type=int, default=2, help="总节点数（含 rank0 master）")
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--wait", action="store_true", default=True, help="master 等待 worker Task 启动")
    parser.add_argument("--no-wait", action="store_false", dest="wait")
    parser.add_argument("--hide-children", action="store_true", help="隐藏 launch_multi_node 创建的 worker Task")
    parser.add_argument("--dist-timeout-sec", type=int, default=1800, help="PyTorch rendezvous 等待时间")
    parser.add_argument("--require-cuda", action="store_true", default=True, help="CUDA 不可用时直接失败")
    parser.add_argument("--allow-cpu", action="store_false", dest="require_cuda", help="允许退回 CPU/gloo，仅调试用")

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
    parser.add_argument("--output-dir", default="/tmp/clearml-launch-ddp")

    # HAMi/vGPU 集群由 vgpuHook 读取此段；原生 nvidia.com/gpu 集群可忽略。
    parser.add_argument("--vgpu-number", type=int, default=1, help="每 Pod 卡数（本脚本建议保持 1）")
    parser.add_argument("--vgpu-memory", type=int, default=24, help="每卡显存 GiB（整卡填整卡显存，如 4090=24）")
    parser.add_argument("--vgpu-cores", type=int, default=100, help="算力百分比（整卡=100，切片<100）")
    task = Task.init(project_name="volcano-vgpu", task_name="multinode-launch-ddp-train")
    task.set_tags(["multinode", "launch-multi-node", "ddp", "train"])

    args = parser.parse_args()

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes 至少为 1")
    if args.vgpu_number != 1:
        raise SystemExit("本模板是一 Pod 一进程 / 一张卡；请保持 --vgpu-number 1")

    task.connect(
        {
            "total_num_nodes": args.num_nodes,
            "queue": args.queue,
            "master_port": args.master_port,
            "devices_per_node": 1,
            "wait": args.wait,
        },
        name="launch_multi_node",
    )
    task.connect(
        {"vgpu_number": args.vgpu_number, "vgpu_memory": args.vgpu_memory, "vgpu_cores": args.vgpu_cores},
        name="VGPU",
    )

    if Task.running_locally():
        task.execute_remotely(queue_name=args.queue)

    launch_conf = task.launch_multi_node(
        total_num_nodes=args.num_nodes,
        port=args.master_port,
        queue=args.queue,
        wait=args.wait,
        devices=1,
        hide_children=args.hide_children,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        task.get_logger().report_text("launch_multi_node config: %s" % launch_conf)

    _run_training(args, task)


if __name__ == "__main__":
    main()
