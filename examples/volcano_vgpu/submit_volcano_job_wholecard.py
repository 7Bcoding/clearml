#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 3：提交 Volcano Job 整卡多机 gang 冒烟

流程:
  1. （可选）创建 ClearML Task，供 rank0 写日志
  2. 创建 ConfigMap（挂载 train_volcano_job_smoke.py）
  3. kubectl apply Volcano Job

前置见 MULTINODE_schemes_zh.md §方案 3

用法:
  python submit_volcano_job_wholecard.py --num-nodes 2 --dry-run
  python submit_volcano_job_wholecard.py --num-nodes 2 --apply
  python submit_volcano_job_wholecard.py --num-nodes 2 --apply --no-clearml-task
"""
import argparse
import os
import subprocess
import sys
import textwrap
import uuid

from clearml import Task

DEFAULT_QUEUE = "multinode-full-gpu"
DEFAULT_NAMESPACE = "clearml"
_HERE = os.path.dirname(os.path.abspath(__file__))
_SMOKE_SCRIPT = os.path.join(_HERE, "train_volcano_job_smoke.py")
_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"


def _job_manifest(*, job_name, namespace, queue, num_nodes, master_port, clearml_task_id, configmap_name):
    master_host = "%s-worker-0" % job_name
    task_id = clearml_task_id or ""
    return textwrap.dedent(
        """\
        apiVersion: batch.volcano.sh/v1alpha1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
          labels:
            app: clearml-volcano-job-smoke
            clearml-scheme: "3"
        spec:
          minAvailable: {num_nodes}
          schedulerName: volcano
          queue: {queue}
          plugins:
            env: []
            svc: []
          tasks:
            - replicas: {num_nodes}
              name: worker
              template:
                metadata:
                  labels:
                    app: clearml-volcano-job-smoke
                spec:
                  restartPolicy: Never
                  schedulerName: volcano
                  runtimeClassName: nvidia
                  nodeSelector:
                    gpu.present: "true"
                  volumes:
                    - name: smoke-script
                      configMap:
                        name: {configmap_name}
                  containers:
                    - name: train
                      image: nvidia/cuda:12.4.1-runtime-ubuntu22.04
                      command: ["/bin/bash", "-lc"]
                      args:
                        - |
                          set -e
                          pip install -q clearml==2.1.8 torch==2.5.1 \\
                            --extra-index-url {torch_index}
                          python /app/train_volcano_job_smoke.py
                      env:
                        - name: NNODES
                          value: "{num_nodes}"
                        - name: MASTER_ADDR
                          value: "{master_host}"
                        - name: MASTER_PORT
                          value: "{master_port}"
                        - name: CLEARML_TASK_ID
                          value: "{task_id}"
                        - name: NCCL_DEBUG
                          value: "INFO"
                        - name: NCCL_IB_DISABLE
                          value: "1"
                        - name: NCCL_SOCKET_IFNAME
                          value: "eth0"
                      resources:
                        limits:
                          nvidia.com/gpu: "1"
                        requests:
                          cpu: "8"
                          memory: 32Gi
                          nvidia.com/gpu: "1"
                      volumeMounts:
                        - name: smoke-script
                          mountPath: /app
        """
    ).format(
        job_name=job_name,
        namespace=namespace,
        queue=queue,
        num_nodes=num_nodes,
        master_port=master_port,
        clearml_task_id=task_id,
        configmap_name=configmap_name,
        master_host=master_host,
        torch_index=_TORCH_INDEX,
    )


def _kubectl_apply_yaml(yaml_text: str) -> None:
    subprocess.check_call(["kubectl", "apply", "-f", "-"], input=yaml_text, text=True)


def main():
    p = argparse.ArgumentParser(description="Submit Volcano Job whole-card gang smoke (scheme 3)")
    p.add_argument("--num-nodes", type=int, default=2)
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--job-name", default="", help="默认 clearml-vjob-<id>")
    p.add_argument("--project", default="volcano-vgpu")
    p.add_argument("--dry-run", action="store_true", help="只打印 YAML / kubectl 提示，不执行")
    p.add_argument("--apply", action="store_true", help="执行 kubectl apply")
    p.add_argument("--no-clearml-task", action="store_true", help="不创建 ClearML Task（无 WebUI 指标）")
    args = p.parse_args()

    if args.num_nodes < 2:
        raise SystemExit("--num-nodes 至少为 2")
    if args.apply and args.dry_run:
        raise SystemExit("--apply 与 --dry-run 不能同时使用")

    run_id = uuid.uuid4().hex[:8]
    job_name = args.job_name or ("clearml-vjob-%s" % run_id)
    configmap_name = "%s-script" % job_name

    clearml_task_id = ""
    if not args.no_clearml_task:
        t = Task.create(
            project_name=args.project,
            task_name="volcano-job-smoke-%s" % run_id,
            task_type=Task.TaskTypes.training,
            script=_SMOKE_SCRIPT,
            add_task_init_call=False,
        )
        t.set_tags(["multinode", "wholecard", "volcano-job", "scheme-3", "smoke", "job-%s" % job_name])
        clearml_task_id = t.id
        print("ClearML Task (rank0 日志): %s" % clearml_task_id)

    manifest = _job_manifest(
        job_name=job_name,
        namespace=args.namespace,
        queue=args.queue,
        num_nodes=args.num_nodes,
        master_port=args.master_port,
        clearml_task_id=clearml_task_id,
        configmap_name=configmap_name,
    )

    print("\n=== Volcano Job manifest (%s) ===" % job_name)
    print(manifest)

    cm_cmd = [
        "kubectl",
        "create",
        "configmap",
        configmap_name,
        "-n",
        args.namespace,
        "--from-file=train_volcano_job_smoke.py=%s" % _SMOKE_SCRIPT,
        "--dry-run=client",
        "-o",
        "yaml",
    ]

    if args.dry_run:
        print("\n[dry-run] ConfigMap 命令:")
        print(" ", " ".join(cm_cmd), "| kubectl apply -f -")
        print("\n[dry-run] Job manifest 见上方")
        return

    if not args.apply:
        print("\n手动执行:")
        print(" ", " ".join(cm_cmd), "| kubectl apply -f -")
        print("  kubectl apply -f - <<'EOF'")
        print(manifest, end="")
        print("EOF")
        print("\n或: python %s --num-nodes %s --apply" % (os.path.basename(__file__), args.num_nodes))
        return

    cm_yaml = subprocess.check_output(cm_cmd, text=True)
    print("\n=== Applying ConfigMap %s ===" % configmap_name)
    _kubectl_apply_yaml(cm_yaml)
    print("\n=== Applying Volcano Job %s ===" % job_name)
    _kubectl_apply_yaml(manifest)

    print("\nWatch:")
    print("  kubectl get job %s -n %s" % (job_name, args.namespace))
    print("  kubectl get pods -n %s -l volcano.sh/job-name=%s -o wide" % (args.namespace, job_name))
    if clearml_task_id:
        print("WebUI Task: %s  (rank0 scalar smoke/allreduce_sum)" % clearml_task_id)


if __name__ == "__main__":
    main()
