#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI 表单提交 —— 单机 vGPU 训练（Pipeline + NEW RUN）

将 train/single_vgpu_full.py 注册为 Pipeline 的一步；算法工程师在 WebUI 填表启动，无需改 YAML。

前置（平台一次性）:
  1. 至少跑通一次模板 Task（或 Publish）:
       python train/single_vgpu_full.py   # 本地 exit 于 execute_remotely 亦可
     项目 volcano-vgpu / 任务名 train-single-gpu
  2. services 队列有 Agent（或本脚本加 --local）

注册 Pipeline（平台工程师）:
  python submit/register_pipeline_single.py
  python submit/register_pipeline_single.py --local          # 无 services Agent 时在本机跑 controller
  python submit/register_pipeline_single.py --services-queue services

算法工程师日常使用:
  WebUI → Pipelines → 项目 volcano-vgpu → 「Submit single-gpu training」→ + NEW RUN
  填 epochs / lr / vGPU 等 → Run

参数覆盖路径与 HPO 相同: CONFIGURATION 段名/键名，如 Args/--epochs、VGPU/vgpu_memory
"""
import argparse
import sys

from clearml import Task
from clearml.automation import PipelineController

BASE_PROJECT = "volcano-vgpu"
BASE_TASK_NAME = "train-single-gpu"
PIPELINE_NAME = "Submit single-gpu training"
PIPELINE_VERSION = "1.0.0"
DEFAULT_TRAIN_QUEUE = "volcano-queue"
DEFAULT_SERVICES_QUEUE = "services"


def _warn_if_base_task_missing() -> None:
    t = Task.get_task(project_name=BASE_PROJECT, task_name=BASE_TASK_NAME, allow_archived=False)
    if t is None:
        print(
            "WARNING: 未找到模板 Task %s/%s。请先执行: python train/single_vgpu_full.py"
            % (BASE_PROJECT, BASE_TASK_NAME),
            file=sys.stderr,
        )


def build_pipeline() -> PipelineController:
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=BASE_PROJECT,
        version=PIPELINE_VERSION,
        add_pipeline_tags=True,
    )

    pipe.add_parameter("epochs", 5, "训练 epoch 数", param_type="int")
    pipe.add_parameter("lr", 1e-3, "学习率", param_type="float")
    pipe.add_parameter("batch_size", 128, "batch size", param_type="int")
    pipe.add_parameter("weight_decay", 1e-4, "AdamW weight decay", param_type="float")
    pipe.add_parameter("hidden", 256, "MLP hidden 维度", param_type="int")
    pipe.add_parameter("seed", 42, "随机种子", param_type="int")
    pipe.add_parameter("vgpu_number", 1, "每 Pod vGPU 卡数", param_type="int")
    pipe.add_parameter("vgpu_memory", 4, "vGPU 显存 (GiB, factor=1024)", param_type="int")
    pipe.add_parameter("vgpu_cores", 30, "vGPU 算力 (%)", param_type="int")
    pipe.add_parameter("queue", DEFAULT_TRAIN_QUEUE, "训练任务入队队列名")

    pipe.set_default_execution_queue(DEFAULT_TRAIN_QUEUE)

    pipe.add_step(
        name="train",
        base_task_project=BASE_PROJECT,
        base_task_name=BASE_TASK_NAME,
        execution_queue="${pipeline.queue}",
        parameter_override={
            "Args/--epochs": "${pipeline.epochs}",
            "Args/--lr": "${pipeline.lr}",
            "Args/--batch-size": "${pipeline.batch_size}",
            "Args/--weight-decay": "${pipeline.weight_decay}",
            "Args/--hidden": "${pipeline.hidden}",
            "Args/--seed": "${pipeline.seed}",
            "VGPU/vgpu_number": "${pipeline.vgpu_number}",
            "VGPU/vgpu_memory": "${pipeline.vgpu_memory}",
            "VGPU/vgpu_cores": "${pipeline.vgpu_cores}",
        },
    )
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="注册/启动单机 vGPU 训练 Pipeline")
    parser.add_argument(
        "--local",
        action="store_true",
        help="在本机运行 pipeline controller（不依赖 services 队列 Agent）",
    )
    parser.add_argument(
        "--local-steps",
        action="store_true",
        help="与 --local 合用：训练 step 也在本机子进程执行（仅调试）",
    )
    parser.add_argument(
        "--services-queue",
        default=DEFAULT_SERVICES_QUEUE,
        help="pipeline controller 入队队列 (默认: services)",
    )
    args = parser.parse_args()

    _warn_if_base_task_missing()
    pipe = build_pipeline()

    if args.local or args.local_steps:
        print("Starting pipeline locally (controller on this machine)...")
        pipe.start_locally(run_pipeline_steps_locally=args.local_steps)
    else:
        print(
            "Enqueueing pipeline controller to queue=%s (training steps -> pipeline.queue)"
            % args.services_queue
        )
        pipe.start(queue=args.services_queue)

    print("Pipeline registered/started: %s / %s v%s" % (BASE_PROJECT, PIPELINE_NAME, PIPELINE_VERSION))
    print("WebUI: Pipelines -> %s -> '%s' -> + NEW RUN" % (BASE_PROJECT, PIPELINE_NAME))


if __name__ == "__main__":
    main()
