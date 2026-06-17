#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 多卡 DDP 长跑训练 (CNN, 默认 >=5 分钟, 官方写法)

VGG 风格 CNN + 合成图像数据集 + 基于时长的训练循环, 在 2x2GB vGPU 上稳定跑满
5 分钟以上, 便于采集 ClearML 的 GPU / 主机资源监控曲线。

平台相关只有「依赖声明 + VGPU connect」两处, 其余是标准 ClearML + 标准 DDP。
DDP 用 mp.spawn 必须放在 main() (if __name__ == "__main__") 里; torch 在 worker 内导入。

提交:
    python train_ddp_cnn_volcano_vgpu.py
    python train_ddp_cnn_volcano_vgpu.py --target-minutes 10 --batch-size 128 --width 64
显存预算 (每卡 2GB, 约 1.5GiB 可用): width=64 batch=128 image=32 安全; OOM 降 batch/width。
"""
import argparse
import os

from clearml import Task

MASTER_PORT = 29500
QUEUE = "volcano-queue"


def build_model(width, in_channels, num_classes):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(in_channels, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, width * 8),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 8, num_classes),
    )


def ddp_worker(rank, world_size, hp, task_id):
    import time

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(int(hp["seed"]))

    logger = None
    output_model = None
    if rank == 0:
        from clearml import OutputModel
        task = Task.get_task(task_id=task_id)
        logger = task.get_logger()
        output_model = OutputModel(task=task, framework="pytorch", name="ddp-cnn")
        logger.report_text("DDP CNN start: world_size=%s, torch=%s" % (world_size, torch.__version__))

    dist.barrier()

    img = int(hp["image_size"])
    ch = int(hp["in_channels"])
    ncls = int(hp["num_classes"])
    x = torch.randn(int(hp["num_samples"]), ch, img, img)
    y = torch.randint(0, ncls, (int(hp["num_samples"]),))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=int(hp["batch_size"]), sampler=sampler, pin_memory=True, drop_last=True
    )

    model = build_model(int(hp["width"]), ch, ncls).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optim = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    criterion = nn.CrossEntropyLoss()

    if rank == 0 and logger is not None:
        n_params = sum(p.numel() for p in model.parameters())
        logger.report_text("model params=%.2fM, global_batch=%s" % (n_params / 1e6, int(hp["batch_size"]) * world_size))
        logger.report_single_value("model_params_million", round(n_params / 1e6, 3))

    epochs = int(hp["epochs"])
    target_sec = float(hp["target_minutes"]) * 60.0
    start = time.time()
    step = 0
    last_loss = 0.0
    last_acc = 0.0
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        correct = 0
        count = 0
        nb = 0
        ep_start = time.time()
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optim.step()
            running += float(loss.item())
            nb += 1
            with torch.no_grad():
                correct += int((logits.argmax(1) == by).sum().item())
                count += int(by.numel())
            if rank == 0 and logger is not None:
                logger.report_scalar("train", "loss", float(loss.item()), iteration=step)
            step += 1
        last_loss = running / max(nb, 1)
        last_acc = correct / max(count, 1)
        elapsed = time.time() - start
        if rank == 0 and logger is not None:
            logger.report_scalar("train", "epoch_loss", last_loss, iteration=epoch)
            logger.report_scalar("train", "accuracy", last_acc, iteration=epoch)
            logger.report_scalar("perf", "images_per_sec", count / max(time.time() - ep_start, 1e-6), iteration=epoch)
        print("rank=%s epoch=%s loss=%.4f acc=%.4f elapsed=%.0fs" % (rank, epoch, last_loss, last_acc, elapsed))
        epoch += 1

        # rank0 决定是否继续并 broadcast, 保证各 rank 集合通信同步 (否则会 hang)
        keep = 1 if (epoch < epochs and elapsed < target_sec) else 0
        if target_sec <= 0 and epoch >= epochs:
            keep = 0
        flag = torch.tensor([keep], device=device)
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            break

    dist.barrier()
    if rank == 0 and logger is not None and output_model is not None:
        ckpt = "model_ddp_cnn.pt"
        torch.save(model.module.state_dict(), ckpt)
        output_model.update_weights(ckpt)
        task.upload_artifact("checkpoint", artifact_object=ckpt)
        logger.report_single_value("final_loss", last_loss)
        logger.report_single_value("final_accuracy", last_acc)
        logger.report_single_value("epochs_run", epoch)
        print("DONE (rank 0): epochs_run=%s" % epoch)
    dist.destroy_process_group()


def main():
    # 【平台相关 1/2】依赖文件 (官方 API, 必须在 Task.init 之前)
    reqs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file=reqs)

    task = Task.init(project_name="volcano-vgpu", task_name="train-ddp-cnn")
    task.set_tags(["ddp", "cnn", "long-run", "example"])

    # ------------------------------------------------------------------
    # 超参定义 (三选一, 详见 USAGE_zh.md §4)
    # ------------------------------------------------------------------
    # [A] argparse —— 默认
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000, help="最大 epoch (通常由 --target-minutes 提前结束)")
    parser.add_argument("--target-minutes", type=float, default=5.0, help="目标训练时长(分钟); <=0 关闭")
    parser.add_argument("--batch-size", type=int, default=128, help="每张卡的 batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=64, help="CNN 基础通道数 (2GB vGPU 建议 <=64)")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # [B] 纯 dict: 取消注释, 并注释掉上面的 argparse 块
    # hp = task.connect({"epochs": 10000, "target_minutes": 5.0, "batch_size": 128, "lr": 1e-3,
    #                    "weight_decay": 5e-4, "width": 64, "image_size": 32, "in_channels": 3,
    #                    "num_classes": 10, "num_samples": 20000, "seed": 42}, name="Training")
    # args = argparse.Namespace(**hp)   # 之后照常用 args.epochs ...; vars(args) 仍可传给 worker

    # [C] YAML: 取消注释, 并注释掉 argparse 块; 参考 config.example.yaml
    # CNN 用到的 width/image_size 等 config.example.yaml 没有, 用 dict 默认值兜底再被 yaml 覆盖
    # cfg = task.connect_configuration(os.path.join(os.path.dirname(__file__), "config.example.yaml"), name="General")
    # hp = {"target_minutes": 5.0, "width": 64, "image_size": 32, "in_channels": 3, "num_classes": 10,
    #       **cfg["training"], **cfg.get("model", {}), **cfg.get("data", {})}
    # args = argparse.Namespace(**hp)

    # 【平台相关 2/2】★ 申请 vGPU —— 在这里自定义资源 (DDP: vgpu_number 即 world_size) ★
    vgpu = task.connect({"vgpu_number": 2, "vgpu_memory": 2, "vgpu_cores": 50}, name="VGPU")

    task.execute_remotely(queue_name=QUEUE)  # 本地调试可注释此行

    # ===== 以下只在集群 Pod 内执行 =====
    import torch
    import torch.multiprocessing as mp

    world_size = int(vgpu["vgpu_number"])
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("need %s GPUs, got %s" % (world_size, torch.cuda.device_count()))

    print("Launching DDP CNN: world_size=%s, visible=%s" % (world_size, torch.cuda.device_count()))
    mp.set_start_method("spawn", force=True)
    mp.spawn(ddp_worker, args=(world_size, vars(args), task.id), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
