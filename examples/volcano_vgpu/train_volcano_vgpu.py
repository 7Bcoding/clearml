#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 标准远程训练模板

适用环境: 自建 ClearML Server + K8s Glue Agent + Volcano/HAMi vGPU

本地提交 (开发机需能访问 ClearML API, 且已配置 ~/clearml.conf):
    python train_volcano_vgpu.py --remote --vgpu-memory 2 --vgpu-cores 30
    python train_volcano_vgpu.py --remote --epochs 20 --lr 3e-4

平台需启用 agent vgpuHook; SDK 通过 Task.connect(..., name="VGPU") 按任务指定显存/算力。
WebUI 复跑: Clone -> 修改 CONFIGURATION -> VGPU -> Enqueue
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path

from clearml import Logger, OutputModel, Task

from vgpu import add_remote_repo_args, apply_standalone_preflight, connect_vgpu, prepare_remote_repo

# 必须在 Task.init 之前; pin cu124 兼容版本, 避免 agent 自动安装 torch 2.12
Task.add_requirements(str(Path(__file__).with_name("requirements-remote.txt")))

DEFAULT_QUEUE = "volcano-queue"


def build_model(num_features: int = 784, hidden: int = 256, num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClearML Volcano vGPU training template")
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--queue", default=DEFAULT_QUEUE, help="ClearML 队列")
    parser.add_argument("--vgpu-number", type=int, default=1, help="volcano.sh/vgpu-number")
    parser.add_argument("--vgpu-memory", type=int, default=2, help="volcano.sh/vgpu-memory (GiB, factor=1024)")
    parser.add_argument("--vgpu-cores", type=int, default=50, help="volcano.sh/vgpu-cores (0-100)")
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    logger: Logger,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optim.step()

        total_loss += float(loss.item())
        logger.report_scalar("train", "loss", float(loss.item()), iteration=global_step)
        global_step += 1

    avg_loss = total_loss / max(len(loader), 1)
    logger.report_scalar("train", "epoch_loss", avg_loss, iteration=epoch)
    return avg_loss, global_step


def main() -> None:
    args = parse_args()

    apply_standalone_preflight(args)

    task = Task.init(
        project_name="volcano-vgpu",
        task_name="train-template",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={"pytorch": True},
    )

    hparams = task.connect(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden": args.hidden,
            "seed": args.seed,
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

    if args.remote:
        prepare_remote_repo(task, args)
        task.execute_remotely(queue_name=args.queue, exit_process=True)

    # ── 以下仅在集群任务 Pod 内执行 ──
    logger = task.get_logger()
    output_model = OutputModel(task=task, framework="pytorch", name="baseline")

    torch.manual_seed(int(hparams["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.report_text(f"device={device}, torch={torch.__version__}")
    print(f"device={device}, torch={torch.__version__}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 示例用合成数据; 实际项目替换为 Dataset / DataLoader
    num_samples = 2048
    num_features = 784
    num_classes = 10
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=int(hparams["batch_size"]),
        shuffle=True,
    )

    model = build_model(
        num_features=num_features,
        hidden=int(hparams["hidden"]),
        num_classes=num_classes,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(int(hparams["epochs"])):
        avg_loss, global_step = train_one_epoch(
            model, loader, optim, criterion, device, logger, epoch, global_step
        )
        print(f"epoch={epoch} avg_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(os.getcwd(), "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    output_model.update_weights(ckpt_path)
    output_model.update_design(
        config_dict={
            "arch": "mlp",
            **hparams,
            **vgpu,
            "queue": args.queue,
        }
    )

    logger.report_single_value("final_epoch_loss", avg_loss)
    logger.report_text("Training finished.")
    print("DONE")


if __name__ == "__main__":
    main()
