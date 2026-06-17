#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多机多卡 DDP 训练模板（torchrun 入口，官方写法）

⚠️ 平台前置：需要平台侧先支持「一个 ClearML Task → 多 Pod Job」与专用整卡队列。
   就绪前请用单 Pod 的 train_template.py / train_ddp_*.py。
   平台改造见 PLATFORM_scheme3b_volcano_job_agent_zh.md；操作见 MULTINODE_schemes_zh.md。

与单 Pod DDP 的差异（见 USAGE_zh.md §8.1）：
  - 进程由 **torchrun**（平台在各节点拉起）启动，不再用 mp.spawn；
  - MASTER_ADDR / MASTER_PORT / RANK / LOCAL_RANK / WORLD_SIZE 由平台注入（torchrun
    或 Volcano Job svc 插件），脚本只读 env，不自己设 127.0.0.1；
  - 整卡多机队列 vgpuHook=false，用 nvidia.com/gpu，**不要** connect(VGPU)；
    资源诉求写在 Cluster 段（nnodes / nproc_per_node）供平台读取。
  - ClearML 仍只在 **global rank 0** 写日志 / 存模型，且 execute_remotely 只调用一次。

提交（平台就绪后）:
    python train_multinode_ddp_template.py --epochs 10 --batch-size 64
"""
import argparse
import os

from clearml import OutputModel, Task

_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)

task = Task.init(project_name="volcano-vgpu", task_name="train-multinode-ddp")
task.set_tags(["multinode", "ddp", "template"])

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64, help="每卡 batch；全局=batch×WORLD_SIZE")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--queue", default="multinode-full-gpu", help="整卡多机队列，由平台约定")
args = parser.parse_args()

# 资源诉求写入 Cluster 段（平台读取以决定 Pod 数 / 每 Pod 卡数）；整卡多机不写 VGPU 段
task.connect({"nnodes": 2, "nproc_per_node": 2}, name="Cluster")

task.execute_remotely(queue_name=args.queue)  # 本地调试可注释此行

# ========== 以下在训练 Pod 内，由 torchrun 为每个 rank 执行 ==========
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# 平台/torchrun 注入；不在脚本里硬编码 MASTER_ADDR
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

logger = None
output_model = None
if rank == 0:
    tid = os.environ.get("CLEARML_TASK_ID") or Task.current_task().id
    task = Task.get_task(task_id=tid)
    logger = task.get_logger()
    output_model = OutputModel(task=task, framework="pytorch", name="multinode-ddp")
    logger.report_text(
        "multinode DDP start: world_size=%s MASTER_ADDR=%s torch=%s"
        % (world_size, os.environ.get("MASTER_ADDR"), torch.__version__)
    )

torch.manual_seed(42 + rank)
# —— 用你的真实 Dataset 替换以下合成数据（建议配合 clearml.Dataset，见 USAGE_zh.md §5.7）——
dataset = TensorDataset(torch.randn(4096, 784), torch.randint(0, 10, (4096,)))
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)

model = DDP(
    nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device),
    device_ids=[local_rank], output_device=local_rank,
)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

loss = None
for epoch in range(args.epochs):
    sampler.set_epoch(epoch)
    model.train()
    for bx, by in loader:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(bx), by)
        loss.backward()
        optim.step()
    if rank == 0 and logger is not None:
        logger.report_scalar("train", "loss", float(loss.item()), iteration=epoch)
    print("rank=%s epoch=%s done" % (rank, epoch))

if rank == 0 and output_model is not None:
    torch.save(model.module.state_dict(), "model.pt")
    output_model.update_weights("model.pt")
    if loss is not None:
        logger.report_single_value("final_loss", float(loss.item()))

dist.destroy_process_group()
print("rank=%s DONE" % rank)
