#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 多卡 DDP 训练 (MLP, 官方写法)

单 Pod 内用 torch.multiprocessing.spawn 起多进程 DDP, 每进程绑定一张 vGPU。
平台相关只有「依赖声明 + VGPU connect」两处, 其余是标准 ClearML + 标准 DDP。

注意: DDP 用 mp.spawn 必须放在 if __name__ == "__main__" 保护的 main() 里
(子进程会重新 import 本模块); torch 在 worker 内导入, 故本机提交无需 torch。

提交:
    python train/singlepod_ddp_smoke.py
    python train/singlepod_ddp_smoke.py --epochs 10 --batch-size 64 --hidden 128
要点: --batch-size 是每卡 batch (全局=batch×vgpu_number); 仅 rank0 上报 ClearML。
"""
import argparse
import os

from clearml import Task

MASTER_PORT = 29500
QUEUE = "volcano-queue"


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
        output_model = OutputModel(task=task, framework="pytorch", name="ddp-baseline")
        logger.report_text("DDP start: world_size=%s, torch=%s" % (world_size, torch.__version__))

    dist.barrier()

    # 同一份合成数据, DistributedSampler 切分到各 rank (换成你的真实 Dataset)
    x = torch.randn(int(hp["num_samples"]), 784)
    y = torch.randint(0, 10, (int(hp["num_samples"]),))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=int(hp["batch_size"]), sampler=sampler, pin_memory=True)

    model = nn.Sequential(nn.Linear(784, int(hp["hidden"])), nn.ReLU(), nn.Linear(int(hp["hidden"]), 10)).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optim = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    criterion = nn.CrossEntropyLoss()

    epochs = int(hp["epochs"])
    min_runtime = int(hp["min_runtime_sec"])
    start = time.time()
    step = 0
    last = 0.0
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optim.step()
            running += float(loss.item())
            if rank == 0 and logger is not None:
                logger.report_scalar("train", "loss", float(loss.item()), iteration=step)
            step += 1
        last = running / max(len(loader), 1)
        if rank == 0 and logger is not None:
            logger.report_scalar("train", "epoch_loss", last, iteration=epoch)
        print("rank=%s epoch=%s avg_loss=%.4f" % (rank, epoch, last))
        epoch += 1

        # rank0 决定是否继续并 broadcast, 保证各 rank 集合通信同步 (否则会 hang)
        keep = 1 if (epoch < epochs or time.time() - start < min_runtime) else 0
        flag = torch.tensor([keep], device=device)
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            break

    dist.barrier()
    if rank == 0 and logger is not None and output_model is not None:
        ckpt = "model_ddp.pt"
        torch.save(model.module.state_dict(), ckpt)
        output_model.update_weights(ckpt)
        task.upload_artifact("checkpoint", artifact_object=ckpt)
        logger.report_single_value("final_epoch_loss", last)
        print("DONE (rank 0)")
    dist.destroy_process_group()


def main():
    # 【平台相关 1/2】依赖文件 (官方 API, 必须在 Task.init 之前)
    reqs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file=reqs)

    task = Task.init(project_name="volcano-vgpu", task_name="train-ddp")
    task.set_tags(["ddp", "mlp", "example"])
    if Task.running_locally():
        # Keep the remote environment small and deterministic; cloned tasks can
        # otherwise retain a huge auto-frozen local environment.
        task.set_packages(reqs)

    # ------------------------------------------------------------------
    # 超参定义 (三选一, 详见 USAGE_zh.md §4)
    # DDP: 超参在 main() 里解析, 经 vars(args) 传给 worker; rank0 只负责上报 ClearML
    # ------------------------------------------------------------------
    # [A] argparse —— 默认
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64, help="每张卡的 batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-runtime-sec", type=int, default=0, help=">0 时循环至此时长, 便于采集监控")
    args = parser.parse_args()

    # [B] 纯 dict: 取消注释, 并注释掉上面的 argparse 块
    # hp = task.connect({"epochs": 5, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4,
    #                    "hidden": 128, "num_samples": 4096, "seed": 42, "min_runtime_sec": 0}, name="Training")
    # args = argparse.Namespace(**hp)   # 之后照常用 args.epochs ...; vars(args) 仍可传给 worker

    # [C] YAML: 取消注释, 并注释掉 argparse 块; 参考 config.example.yaml
    # cfg = task.connect_configuration(os.path.join(os.path.dirname(__file__), "config.example.yaml"), name="General")
    # hp = {**cfg["training"], **cfg.get("data", {}), "min_runtime_sec": 0}
    # args = argparse.Namespace(**hp)

    # 【平台相关 2/2】★ 申请 vGPU —— 在这里自定义资源 (DDP: vgpu_number 即 world_size) ★
    vgpu = task.connect({"vgpu_number": 2, "vgpu_memory": 2, "vgpu_cores": 30}, name="VGPU")

    task.execute_remotely(queue_name=QUEUE)  # 本地调试可注释此行

    # ===== 以下只在集群 Pod 内执行 =====
    import torch
    import torch.multiprocessing as mp

    world_size = int(vgpu["vgpu_number"])
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("need %s GPUs, got %s" % (world_size, torch.cuda.device_count()))

    print("Launching DDP: world_size=%s, visible=%s" % (world_size, torch.cuda.device_count()))
    mp.set_start_method("spawn", force=True)
    mp.spawn(ddp_worker, args=(world_size, vars(args), task.id), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
