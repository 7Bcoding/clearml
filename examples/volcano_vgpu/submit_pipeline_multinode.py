#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI 表单提交 —— 多机 vGPU 训练（Pipeline + NEW RUN）

将 train_launch_multinode_wholecard.py 注册为 Pipeline 的一步；表单覆盖 VGPU 与 launch_multi_node 段。

前置（平台一次性）:
  1. 跑通 multinode Agent + 队列 multinode-full-gpu（见 MULTINODE_schemes_zh.md）
  2. 至少跑通一次模板 Task:
       python train_launch_multinode_wholecard.py --num-nodes 2 ...
     项目 volcano-vgpu / 任务名 multinode-launch-wholecard
  3. services 队列有 Agent（或 --local）

注册:
  python submit_pipeline_multinode.py
  python submit_pipeline_multinode.py --local

算法工程师:
  WebUI → Pipelines → 「Submit multinode-vgpu training」→ + NEW RUN
  填 num_nodes / vGPU 显存算力 / 队列 → Run
"""
import argparse
import sys

from clearml import Task
from clearml.automation import PipelineController

BASE_PROJECT = "volcano-vgpu"
BASE_TASK_NAME = "multinode-launch-wholecard"
PIPELINE_NAME = "Submit multinode-vgpu training"
PIPELINE_VERSION = "1.0.0"
DEFAULT_TRAIN_QUEUE = "multinode-full-gpu"
DEFAULT_SERVICES_QUEUE = "services"


def _warn_if_base_task_missing() -> None:
    t = Task.get_task(project_name=BASE_PROJECT, task_name=BASE_TASK_NAME, allow_archived=False)
    if t is None:
        print(
            "WARNING: 未找到模板 Task %s/%s。请先执行: python train_launch_multinode_wholecard.py --num-nodes 2"
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

    pipe.add_parameter("num_nodes", 2, "总节点数（含 rank0 master）", param_type="int")
    pipe.add_parameter("master_port", 29500, "NCCL MASTER_PORT", param_type="int")
    pipe.add_parameter("vgpu_number", 1, "每 Pod vGPU 卡数", param_type="int")
    pipe.add_parameter("vgpu_memory", 4, "vGPU 显存 (GiB)", param_type="int")
    pipe.add_parameter("vgpu_cores", 30, "vGPU 算力 (%)", param_type="int")
    pipe.add_parameter("queue", DEFAULT_TRAIN_QUEUE, "master/worker 训练队列")

    pipe.set_default_execution_queue(DEFAULT_TRAIN_QUEUE)

    pipe.add_step(
        name="multinode_train",
        base_task_project=BASE_PROJECT,
        base_task_name=BASE_TASK_NAME,
        execution_queue="${pipeline.queue}",
        parameter_override={
            "launch_multi_node/total_num_nodes": "${pipeline.num_nodes}",
            "launch_multi_node/queue": "${pipeline.queue}",
            "launch_multi_node/master_port": "${pipeline.master_port}",
            "VGPU/vgpu_number": "${pipeline.vgpu_number}",
            "VGPU/vgpu_memory": "${pipeline.vgpu_memory}",
            "VGPU/vgpu_cores": "${pipeline.vgpu_cores}",
        },
    )
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="注册/启动多机 vGPU 训练 Pipeline")
    parser.add_argument(
        "--local",
        action="store_true",
        help="在本机运行 pipeline controller",
    )
    parser.add_argument(
        "--local-steps",
        action="store_true",
        help="训练 step 也在本机子进程执行（仅调试）",
    )
    parser.add_argument(
        "--services-queue",
        default=DEFAULT_SERVICES_QUEUE,
        help="pipeline controller 队列 (默认: services)",
    )
    args = parser.parse_args()

    _warn_if_base_task_missing()
    pipe = build_pipeline()

    if args.local or args.local_steps:
        print("Starting multinode pipeline locally...")
        pipe.start_locally(run_pipeline_steps_locally=args.local_steps)
    else:
        print("Enqueueing pipeline controller to queue=%s" % args.services_queue)
        pipe.start(queue=args.services_queue)

    print("Pipeline: %s / %s v%s" % (BASE_PROJECT, PIPELINE_NAME, PIPELINE_VERSION))
    print("WebUI: Pipelines -> %s -> '%s' -> + NEW RUN" % (BASE_PROJECT, PIPELINE_NAME))


if __name__ == "__main__":
    main()
