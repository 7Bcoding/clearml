#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 多卡 DDP 长跑训练 (CNN, 默认 >=5 分钟)

相比 train_ddp_volcano_vgpu.py 的轻量 MLP, 这里用一个可配置的 VGG 风格 CNN +
合成图像数据集, 单步计算量更大, 配合基于时长的训练循环, 可在 2x2GB vGPU 上稳定
跑满 5 分钟以上, 便于采集 ClearML 的 GPU / 主机资源监控曲线。

本地提交 (--remote) 只需安装 clearml; torch 在集群任务 Pod 内由 agent pip 安装。
    python train_ddp_cnn_volcano_vgpu.py --remote --vgpu-number 2 --vgpu-memory 2 --vgpu-cores 30
    python train_ddp_cnn_volcano_vgpu.py --remote --target-minutes 10 --batch-size 128 --width 64

显存预算 (每卡 2GB, 约 1.5GiB 可用张量):
    width=64  batch=128  image=32  -> 安全
    OOM 时优先降 --batch-size 或 --width

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


def build_model(width: int = 64, in_channels: int = 3, num_classes: int = 10) -> Any:
    """VGG 风格 CNN: 4 个卷积 stage + 全局池化 + 分类头, 通道数由 width 控制。"""
    import torch.nn as nn

    def conv_block(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        conv_block(in_channels, width),
        conv_block(width, width * 2),
        conv_block(width * 2, width * 4),
        conv_block(width * 4, width * 8),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(width * 8, num_classes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClearML Volcano vGPU long-running DDP CNN training")
    parser.add_argument("--epochs", type=int, default=10000, help="最大 epoch 数 (通常由 --target-minutes 提前结束)")
    parser.add_argument("--target-minutes", type=float, default=5.0, help="目标训练时长(分钟); 跑满即停, <=0 关闭")
    parser.add_argument("--batch-size", type=int, default=128, help="每张 GPU 的 batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--width", type=int, default=64, help="CNN 基础通道数 (2GB vGPU 建议 <=64)")
    parser.add_argument("--image-size", type=int, default=32, help="合成图像边长")
    parser.add_argument("--in-channels", type=int, default=3, help="图像通道数")
    parser.add_argument("--num-classes", type=int, default=10, help="分类类别数")
    parser.add_argument("--num-samples", type=int, default=20000, help="合成数据集样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--queue", default=DEFAULT_QUEUE, help="ClearML 队列")
    parser.add_argument("--vgpu-number", type=int, default=2, help="volcano.sh/vgpu-number / DDP world_size")
    parser.add_argument("--vgpu-memory", type=int, default=2, help="volcano.sh/vgpu-memory (GiB, factor=1024)")
    parser.add_argument("--vgpu-cores", type=int, default=50, help="volcano.sh/vgpu-cores (0-100)")
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
) -> tuple[float, float, int]:
    import time

    import torch

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = 0
    epoch_start = time.time()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optim.step()

        loss_value = float(loss.item())
        total_loss += loss_value
        num_batches += 1
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += int((preds == batch_y).sum().item())
            total_count += int(batch_y.numel())

        if rank == 0 and logger is not None:
            logger.report_scalar("train", "loss", loss_value, iteration=global_step)
        global_step += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_count, 1)
    if rank == 0 and logger is not None:
        elapsed = max(time.time() - epoch_start, 1e-6)
        images_per_sec = total_count / elapsed
        logger.report_scalar("train", "epoch_loss", avg_loss, iteration=epoch)
        logger.report_scalar("train", "accuracy", accuracy, iteration=epoch)
        logger.report_scalar("perf", "images_per_sec", images_per_sec, iteration=epoch)
    return avg_loss, accuracy, global_step


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
        output_model = OutputModel(task=task, framework="pytorch", name="ddp-cnn")
        logger.report_text(
            "DDP CNN start: world_size=%s, torch=%s, vgpu=%s"
            % (world_size, torch.__version__, vgpu)
        )
        for idx in range(world_size):
            props = torch.cuda.get_device_properties(idx)
            msg = "rank=%s sees cuda:%s name=%s total_mem=%.1fGiB" % (
                rank, idx, props.name, props.total_memory / (1024**3),
            )
            print(msg)
            logger.report_text(msg)

    dist.barrier()

    num_samples = int(hparams["num_samples"])
    image_size = int(hparams["image_size"])
    in_channels = int(hparams["in_channels"])
    num_classes = int(hparams["num_classes"])

    x = torch.randn(num_samples, in_channels, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=int(hparams["batch_size"]),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    model = build_model(
        width=int(hparams["width"]),
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        msg = "model params=%.2fM, dataset=%s images (%sx%sx%s), global_batch=%s" % (
            n_params / 1e6, num_samples, in_channels, image_size, image_size,
            int(hparams["batch_size"]) * world_size,
        )
        print(msg)
        if logger is not None:
            logger.report_text(msg)
            logger.report_single_value("model_params_million", round(n_params / 1e6, 3))

    epochs = int(hparams["epochs"])
    target_seconds = float(hparams["target_minutes"]) * 60.0
    start_time = time.time()

    global_step = 0
    last_loss = 0.0
    last_acc = 0.0
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        last_loss, last_acc, global_step = train_one_epoch(
            model, loader, optim, criterion, device, logger, epoch, global_step, rank
        )
        elapsed = time.time() - start_time
        print("rank=%s epoch=%s loss=%.4f acc=%.4f elapsed=%.0fs" % (rank, epoch, last_loss, last_acc, elapsed))
        epoch += 1

        # rank 0 decides whether to continue; broadcast keeps all ranks collective-safe.
        keep_going = 1 if (epoch < epochs and elapsed < target_seconds) else 0
        if target_seconds <= 0 and epoch >= epochs:
            keep_going = 0
        flag = torch.tensor([keep_going], device=device)
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            break

    dist.barrier()

    if rank == 0 and output_model is not None and logger is not None:
        ckpt_path = os.path.join(os.getcwd(), "model_ddp_cnn.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        output_model.update_weights(ckpt_path)
        output_model.update_design(
            config_dict={
                "arch": "vgg_style_cnn_ddp",
                "world_size": world_size,
                "epochs_run": epoch,
                **hparams,
                **vgpu,
                "queue": queue,
            }
        )
        logger.report_single_value("final_loss", last_loss)
        logger.report_single_value("final_accuracy", last_acc)
        logger.report_single_value("epochs_run", epoch)
        logger.report_text("DDP CNN training finished after %s epochs." % epoch)
        print("DONE (rank 0): epochs_run=%s" % epoch)

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

    world_size = int(vgpu.get("vgpu_number", hparams.get("world_size", 1)))
    visible = torch.cuda.device_count()
    if visible < world_size:
        raise RuntimeError(
            "torch.cuda.device_count()=%s < world_size=%s; check vGPU scheduling"
            % (visible, world_size)
        )

    print("Launching DDP CNN: world_size=%s, visible_gpus=%s, master_port=%s" % (world_size, visible, master_port))
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
        task_name="train-ddp-cnn",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=False,
    )

    hparams = task.connect(
        {
            "epochs": args.epochs,
            "target_minutes": args.target_minutes,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "width": args.width,
            "image_size": args.image_size,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "num_samples": args.num_samples,
            "seed": args.seed,
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
