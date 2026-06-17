#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearML + Volcano vGPU 单卡训练示例 (官方写法, 较完整)

相比 train_template.py, 这里演示更多 ClearML 训练集成:
  OutputModel / artifact 上传 / 标签 / 最终指标 / (可选) InputModel 预训练权重
超参默认 argparse; 其他写法见脚本内注释与 USAGE_zh.md §4。

提交:
    python train_volcano_vgpu.py
    python train_volcano_vgpu.py --epochs 20 --lr 3e-4
WebUI 复跑: Clone -> 改 CONFIGURATION (Args / VGPU) -> Enqueue
"""
import argparse
import os

from clearml import OutputModel, Task

_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)

task = Task.init(project_name="volcano-vgpu", task_name="train-single-gpu")
task.set_tags(["single-gpu", "mlp", "example"])  # WebUI 筛选用

# ------------------------------------------------------------------
# 超参定义 (三选一, 详见 USAGE_zh.md §4)
# ------------------------------------------------------------------
# [A] argparse —— 默认
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# [B] 纯 dict: 取消注释, 并注释掉上面的 argparse 块
# hp = task.connect({"epochs": 5, "batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4,
#                    "hidden": 256, "seed": 42}, name="Training")
# args = argparse.Namespace(**hp)   # 之后照常用 args.epochs / args.batch_size ...

# [C] YAML: 取消注释, 并注释掉 argparse 块; 参考 config.example.yaml
# cfg = task.connect_configuration(os.path.join(os.path.dirname(__file__), "config.example.yaml"), name="General")
# args = argparse.Namespace(**cfg["training"])   # config.example.yaml 的 key 已是下划线形式

# 【平台】申请 vGPU (vgpu_memory 单位 GiB)
task.connect({"vgpu_number": 1, "vgpu_memory": 2, "vgpu_cores": 50}, name="VGPU")

# 可选: 使用预装 torch 的镜像, 省去 Pod 内 pip (见 USAGE_zh.md §5)
# task.set_base_docker("your-registry/pytorch:2.5.1-cuda12.4-cudnn9-runtime")

task.execute_remotely(queue_name="volcano-queue")  # 本地调试可注释此行

# ============================================================
# 以下只在集群 Pod 内执行
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = task.get_logger()
output_model = OutputModel(task=task, framework="pytorch", name="baseline")

# 可选: 从 ClearML Model Registry 加载预训练权重 (见 USAGE_zh.md §5)
# from clearml import InputModel
# input_model = InputModel(model_id="YOUR_MODEL_ID")  # 或 name="project/model-name"
# pretrained = input_model.get_weights()
# logger.report_text("loaded pretrained from %s" % input_model.id)

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.report_text("device=%s, torch=%s" % (device, torch.__version__))
print("device=%s, torch=%s" % (device, torch.__version__))
if device.type == "cuda":
    print("GPU: %s" % torch.cuda.get_device_name(0))

# —— 用你的真实 Dataset/DataLoader 替换以下合成数据 ——
x = torch.randn(2048, 784)
y = torch.randint(0, 10, (2048,))
loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

model = nn.Sequential(nn.Linear(784, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 10)).to(device)
# if pretrained: model.load_state_dict(torch.load(pretrained, map_location=device))
optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

step = 0
avg_loss = 0.0
for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(bx), by)
        loss.backward()
        optim.step()
        running += float(loss.item())
        logger.report_scalar("train", "loss", float(loss.item()), iteration=step)  # step 级
        step += 1
    avg_loss = running / max(len(loader), 1)
    logger.report_scalar("train", "epoch_loss", avg_loss, iteration=epoch)  # epoch 级
    print("epoch=%s avg_loss=%.4f" % (epoch, avg_loss))

ckpt_path = os.path.join(os.getcwd(), "model.pt")
torch.save(model.state_dict(), ckpt_path)
output_model.update_weights(ckpt_path)  # MODELS 页, 可发布到 Model Registry
task.upload_artifact("checkpoint", artifact_object=ckpt_path)  # ARTIFACTS 页可下载
logger.report_single_value("final_epoch_loss", avg_loss)  # 实验列表可排序
logger.report_text("Training finished.")
print("DONE")
