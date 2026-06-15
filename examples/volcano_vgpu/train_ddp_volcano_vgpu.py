#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 多卡 DDP 训练验证脚本

单 Pod 内通过 torch.multiprocessing.spawn 启动多进程 DDP (数据并行)。
每个进程绑定一张 vGPU; 显存上限仍由 per-task vGPU 配额独立限制。

本地提交 (--remote) 只需安装 clearml; torch 在集群任务 Pod 内由 agent pip 安装。
    python train_ddp_volcano_vgpu.py --remote --vgpu-number 2 --vgpu-memory 2 --vgpu-cores 30
    python train_ddp_volcano_vgpu.py --remote --epochs 10 --batch-size 64 --hidden 128

WebUI 复跑: Clone -> 修改 CONFIGURATION -> VGPU / Training -> Enqueue
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from clearml import Logger, OutputModel, Task

from vgpu import (
    add_remote_repo_args,
    apply_standalone_preflight,
    connect_vgpu,
    pin_remote_packages,
    prepare_remote_repo,
    register_remote_requirements,
)

register_remote_requirements(__file__)

DEFAULT_QUEUE = "volcano-queue"
DEFAULT_MASTER_PORT = 29500


def build_model(num_features: int = 784, hidden: int = 128, num_classes: int = 10) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClearML Volcano vGPU DDP training")
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=64, help="每张 GPU 的 batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--hidden", type=int, default=128, help="隐藏层宽度 (2GB vGPU 建议 <=128)")
    parser.add_argument("--num-samples", type=int, default=4096, help="合成数据集样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--min-runtime-sec",
        type=int,
        default=0,
        help="最短训练时长(秒); >0 时会循环到达此时长, 便于采集 ClearML 资源监控曲线",
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE, help="ClearML 队列")
    parser.add_argument("--vgpu-number", type=int, default=2, help="volcano.sh/vgpu-number / DDP world_size")
    parser.add_argument("--vgpu-memory", type=int, default=2, help="volcano.sh/vgpu-memory (GiB, factor=1024)")
    parser.add_argument("--vgpu-cores", type=int, default=30, help="volcano.sh/vgpu-cores (0-100)")
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT, help="DDP rendezvous 端口")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="本地提交后立即入队并退出; 训练在集群任务 Pod 内执行",
    )
    parser.add_argument(
        "--docker-image",
        default="",
        help="可选: 预装依赖的 GPU 镜像, 设置后会跳过 pip 安装 (需配合 set_packages)",
    )
    add_remote_repo_args(parser)
    return parser.parse_args()


def setup_ddp(rank: int, world_size: int, master_port: int) -> None:
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(
    model: Any,
    loader: Any,
    optim: Any,
    criterion: Any,
    device: Any,
    logger: Logger | None,
    epoch: int,
    global_step: int,
    rank: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optim.step()

        loss_value = float(loss.item())
        total_loss += loss_value
        if rank == 0 and logger is not None:
            logger.report_scalar("train", "loss", loss_value, iteration=global_step)
        global_step += 1

    avg_loss = total_loss / max(len(loader), 1)
    if rank == 0 and logger is not None:
        logger.report_scalar("train", "epoch_loss", avg_loss, iteration=epoch)
    return avg_loss, global_step


def ddp_worker(
    rank: int,
    world_size: int,
    hparams: dict[str, Any],
    vgpu: dict[str, Any],
    task_id: str,
    master_port: int,
    queue: str,
) -> None:
    import time

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    setup_ddp(rank, world_size, master_port)
    device = torch.device("cuda", rank)

    # Same dataset on every rank; DistributedSampler assigns disjoint shards.
    torch.manual_seed(int(hparams["seed"]))

    logger: Logger | None = None
    output_model: OutputModel | None = None
    if rank == 0:
        task = Task.get_task(task_id=task_id)
        logger = task.get_logger()
        output_model = OutputModel(task=task, framework="pytorch", name="ddp-baseline")
        logger.report_text(
            "DDP start: world_size=%s, torch=%s, vgpu=%s"
            % (world_size, torch.__version__, vgpu)
        )
        for idx in range(world_size):
            props = torch.cuda.get_device_properties(idx)
            msg = "rank=%s sees cuda:%s name=%s total_mem=%.1fGiB" % (
                rank,
                idx,
                props.name,
                props.total_memory / (1024**3),
            )
            print(msg)
            logger.report_text(msg)

    dist.barrier()

    num_samples = int(hparams.get("num_samples", 4096))
    num_features = 784
    num_classes = 10
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=int(hparams["batch_size"]),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(
        num_features=num_features,
        hidden=int(hparams["hidden"]),
        num_classes=num_classes,
    ).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(hparams["epochs"])
    min_runtime_sec = int(hparams.get("min_runtime_sec", 0))
    start_time = time.time()

    global_step = 0
    last_avg_loss = 0.0
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        last_avg_loss, global_step = train_one_epoch(
            model, loader, optim, criterion, device, logger, epoch, global_step, rank
        )
        print("rank=%s epoch=%s avg_loss=%.4f" % (rank, epoch, last_avg_loss))
        epoch += 1

        # rank 0 decides whether to continue; broadcast keeps all ranks collective-safe.
        keep_going = 1 if (epoch < epochs or time.time() - start_time < min_runtime_sec) else 0
        flag = torch.tensor([keep_going], device=device)
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            break

    dist.barrier()

    if rank == 0 and output_model is not None and logger is not None:
        ckpt_path = os.path.join(os.getcwd(), "model_ddp.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        output_model.update_weights(ckpt_path)
        output_model.update_design(
            config_dict={
                "arch": "mlp_ddp",
                "world_size": world_size,
                **hparams,
                **vgpu,
                "queue": queue,
            }
        )
        logger.report_single_value("final_epoch_loss", last_avg_loss)
        logger.report_text("DDP training finished.")
        print("DONE (rank 0)")

    cleanup_ddp()


def run_ddp_training(
    hparams: dict[str, Any],
    vgpu: dict[str, Any],
    task: Task,
    master_port: int,
    queue: str,
) -> None:
    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP training")

    world_size = int(vgpu.get("vgpu_number", hparams.get("vgpu_number", 1)))
    visible = torch.cuda.device_count()
    if visible < world_size:
        raise RuntimeError(
            "torch.cuda.device_count()=%s < world_size=%s; check vGPU scheduling"
            % (visible, world_size)
        )

    print("Launching DDP: world_size=%s, visible_gpus=%s, master_port=%s" % (world_size, visible, master_port))
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        ddp_worker,
        args=(world_size, hparams, vgpu, task.id, master_port, queue),
        nprocs=world_size,
        join=True,
    )


def main() -> None:
    args = parse_args()

    apply_standalone_preflight(args)

    task = Task.init(
        project_name="volcano-vgpu",
        task_name="train-ddp",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=False,
    )

    hparams = task.connect(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden": args.hidden,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "min_runtime_sec": args.min_runtime_sec,
            "world_size": args.vgpu_number,
            "master_port": args.master_port,
        },
        name="Training",
    )

    vgpu = connect_vgpu(
        task,
        vgpu_number=args.vgpu_number,
        vgpu_memory=args.vgpu_memory,
        vgpu_cores=args.vgpu_cores,
    )

    if args.docker_image:
        task.set_base_docker(args.docker_image)
        task.set_packages("")
    elif args.remote:
        pin_remote_packages(task, __file__)

    if args.remote:
        prepare_remote_repo(task, args)
        task.execute_remotely(queue_name=args.queue, exit_process=True)

    run_ddp_training(
        hparams=hparams,
        vgpu=vgpu,
        task=task,
        master_port=int(hparams["master_port"]),
        queue=args.queue,
    )


if __name__ == "__main__":
    main()
