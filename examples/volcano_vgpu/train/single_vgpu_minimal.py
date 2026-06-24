#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 单卡训练模板 (官方写法, 推荐起点)

平台相关只有 2 处 (见 USAGE_zh.md):
  1) Task.force_requirements_env_freeze(...)   Pod 依赖
  2) task.connect({...}, name="VGPU")          申请 vGPU

超参默认用 argparse (ClearML 自动捕获到 WebUI Args 段); 其他写法见脚本内注释与 USAGE_zh.md §4。
ClearML 训练集成: 标签 / scalar 曲线 / artifact 上传 (见 USAGE_zh.md §5)。

提交: python train/single_vgpu_minimal.py
本地调试: 注释 execute_remotely 一行
WebUI 复跑: Clone -> CONFIGURATION (Args / VGPU) -> Enqueue
"""
import argparse
import os

from clearml import Task

_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)
# These examples are self-contained; store the entry script in the Task so
# offline workers do not need to git clone/fetch GitHub.
Task.force_store_standalone_script(True)

task = Task.init(project_name="volcano-vgpu", task_name="train-template")
task.set_tags(["template", "single-gpu"])  # WebUI 筛选用, 可选
if Task.running_locally():
    # Keep the remote environment small and deterministic; cloned tasks can
    # otherwise retain a huge auto-frozen local environment.
    task.set_packages(_REQS)

# ------------------------------------------------------------------
# 超参定义 (三选一, 详见 USAGE_zh.md §4)
# ------------------------------------------------------------------
# [A] argparse —— 默认, 适合 CLI 驱动训练; Task.init 后 parse 即自动进 WebUI Args 段
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# [B] 纯 dict (适合 Notebook / 无 CLI): 取消注释, 并注释掉上面的 argparse 块
# hp = task.connect({"epochs": 5, "batch_size": 128, "lr": 1e-3, "hidden": 256, "seed": 42}, name="Training")
# args = argparse.Namespace(**hp)   # 之后照常用 args.epochs / args.batch_size ...

# [C] YAML 配置文件 (适合嵌套/大配置): 取消注释, 并注释掉 argparse 块; 参考 config.example.yaml
# cfg = task.connect_configuration(os.path.join(os.path.dirname(__file__), "config.example.yaml"), name="General")
# args = argparse.Namespace(**cfg["training"])   # config.example.yaml 的 key 已是下划线形式

# 【平台】申请 vGPU (vgpu_memory 单位 GiB, factor=1024)
task.connect({"vgpu_number": 1, "vgpu_memory": 2, "vgpu_cores": 30}, name="VGPU")

task.execute_remotely(queue_name="volcano-queue")  # 本地调试可注释

# ============================================================
# 以下只在集群 Pod 内执行
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = task.get_logger()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.report_text("device=%s, torch=%s" % (device, torch.__version__))

x = torch.randn(2048, 784)
y = torch.randint(0, 10, (2048,))
loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

model = nn.Sequential(nn.Linear(784, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 10)).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

step = 0
for epoch in range(args.epochs):
    model.train()
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(bx), by)
        loss.backward()
        optim.step()
        logger.report_scalar("train", "loss", float(loss.item()), iteration=step)  # step 级曲线
        step += 1
    logger.report_scalar("train", "epoch_loss", float(loss.item()), iteration=epoch)  # epoch 级曲线
    print("epoch=%s done" % epoch)

ckpt = "model.pt"
torch.save(model.state_dict(), ckpt)
task.upload_artifact("checkpoint", artifact_object=ckpt)  # ARTIFACTS 页可下载
logger.report_single_value("final_loss", float(loss.item()))  # 实验列表可排序的最终指标
print("DONE")
