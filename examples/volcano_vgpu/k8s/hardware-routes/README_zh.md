# 4090 HAMi、A100 HAMi、A100 MIG 硬件路线配置

本目录把 ClearML + Volcano + volcano-vgpu-device-plugin 的 GPU 入口按
“硬件型号 + 隔离模式 + 是否多机”拆开。这样做的核心目的是避免同一个
ClearML queue 被错误的 agent 模板消费，例如 A100 MIG 任务被 4090 HAMi
agent 拉走。

## 新增节点操作文档

- [新增 4090 节点部署配置流程](ADD_4090_NODE_zh.md)
- [新增 A100 节点部署配置流程](ADD_A100_NODE_zh.md)

## 路线表

| 路线 | ClearML queue | Volcano mode | 节点选择器 |
|---|---|---|---|
| 4090 HAMi-core | `gpu-4090-hami` | `hami-core` | `gpu.product=4090`, `gpu.mode=hami-core` |
| 4090 HAMi-core 多机 | `gpu-4090-hami-multinode` | `hami-core` | `gpu.product=4090`, `gpu.mode=hami-core` |
| A100 HAMi-core | `gpu-a100-hami` | `hami-core` | `gpu.product=a100`, `gpu.mode=hami-core` |
| A100 HAMi-core 多机 | `gpu-a100-hami-multinode` | `hami-core` | `gpu.product=a100`, `gpu.mode=hami-core` |
| A100 Dynamic MIG | `gpu-a100-mig` | `mig` | `gpu.product=a100`, `gpu.mode=mig` |
| A100 Dynamic MIG 多机 | `gpu-a100-mig-multinode` | `mig` | `gpu.product=a100`, `gpu.mode=mig` |

建议 ClearML queue 名和 Volcano Queue 名保持一致。

## 文件说明

- `queues.yaml`: 六条路线对应的 Volcano Queue。
- `volcano-vgpu-device-config.yaml`: device plugin 全局配置，包含 `gpuMemoryFactor: 1024` 和 A100 MIG geometry。
- `volcano-vgpu-node-config.routes.example.yaml`: 每个节点的 `operatingmode` 示例。这里的节点名必须替换成真实节点名后再应用。
- `agent-values-*.yaml`: 每条路线一个 ClearML agent Helm values。
- `canary-*.yaml`: 直接跑 Kubernetes Pod 的冒烟测试。

## 节点标签和污点

4090 HAMi-core 节点:

```bash
kubectl label node <4090-node> \
  gpu.present=true gpu.product=4090 gpu.mode=hami-core gpu.vendor=nvidia \
  --overwrite

kubectl taint node <4090-node> gpu.clearml/4090=true:NoSchedule --overwrite
```

A100 HAMi-core 节点:

```bash
kubectl label node <a100-hami-node> \
  gpu.present=true gpu.product=a100 gpu.mode=hami-core gpu.vendor=nvidia \
  --overwrite

kubectl taint node <a100-hami-node> gpu.clearml/a100=true:NoSchedule --overwrite
```

A100 MIG 节点:

```bash
kubectl label node <a100-mig-node> \
  gpu.present=true gpu.product=a100 gpu.mode=mig gpu.vendor=nvidia \
  --overwrite

kubectl taint node <a100-mig-node> gpu.clearml/a100=true:NoSchedule --overwrite
```

## A100 MIG 前置动作

A100 MIG 节点必须先在宿主机开启 MIG，再把 node config 切到 `operatingmode=mig`:

```bash
sudo nvidia-smi -i 0 -mig 1
sudo reboot
```

开启或关闭 MIG 会 reset GPU，会影响该 GPU 上所有正在跑的进程，包括宿主机上的 `docker run --gpus ...` 训练。

## 应用顺序

先备份线上配置:

```bash
mkdir -p ~/clearml-hardware-routes-backup
kubectl -n kube-system get cm volcano-vgpu-device-config -o yaml > ~/clearml-hardware-routes-backup/volcano-vgpu-device-config.yaml
kubectl -n kube-system get cm volcano-vgpu-node-config -o yaml > ~/clearml-hardware-routes-backup/volcano-vgpu-node-config.yaml
kubectl get queue -o yaml > ~/clearml-hardware-routes-backup/queues.yaml
```

根据实际 `allocatable` 调整 `queues.yaml` 里的 `capability`:

```bash
kubectl get nodes -o json | jq -r '.items[] |
  "\(.metadata.name) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"])"'
```

应用队列:

```bash
kubectl apply -f examples/volcano_vgpu/k8s/hardware-routes/queues.yaml
```

替换 `volcano-vgpu-node-config.routes.example.yaml` 里的节点名，然后应用 device plugin 配置并重启相关组件:

```bash
kubectl apply -f examples/volcano_vgpu/k8s/hardware-routes/volcano-vgpu-device-config.yaml
kubectl apply -f examples/volcano_vgpu/k8s/hardware-routes/volcano-vgpu-node-config.routes.example.yaml
kubectl -n kube-system rollout restart ds/volcano-device-plugin
kubectl -n volcano-system rollout restart deploy/volcano-scheduler
```

在 ClearML WebUI 或 API 中创建这些 queue:

```text
gpu-4090-hami
gpu-4090-hami-multinode
gpu-a100-hami
gpu-a100-hami-multinode
gpu-a100-mig
gpu-a100-mig-multinode
```

分别安装 agent release:

```bash
HELM_REPO=/path/to/clearml-helm-charts
ROUTES=examples/volcano_vgpu/k8s/hardware-routes

helm upgrade --install clearml-agent-4090-hami "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-4090-hami.yaml"

helm upgrade --install clearml-agent-4090-hami-multinode "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-4090-hami-multinode.yaml"

helm upgrade --install clearml-agent-a100-hami "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-hami.yaml"

helm upgrade --install clearml-agent-a100-hami-multinode "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-hami-multinode.yaml"

helm upgrade --install clearml-agent-a100-mig "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-mig.yaml"

helm upgrade --install clearml-agent-a100-mig-multinode "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-mig-multinode.yaml"
```

## 冒烟测试

先跑直接 Kubernetes Pod，再提交 ClearML 任务:

```bash
kubectl apply -f examples/volcano_vgpu/k8s/hardware-routes/canary-4090-hami.yaml
kubectl logs canary-4090-hami
kubectl delete pod canary-4090-hami

kubectl apply -f examples/volcano_vgpu/k8s/hardware-routes/canary-a100-hami.yaml
kubectl logs canary-a100-hami
kubectl delete pod canary-a100-hami

kubectl apply -f examples/volcano_vgpu/k8s/hardware-routes/canary-a100-mig.yaml
kubectl logs canary-a100-mig
kubectl delete pod canary-a100-mig
```

## 任务侧约定

4090 HAMi-core 和 A100 HAMi-core 任务需要写 `vgpu_cores`:

```python
task.connect({"vgpu_number": 1, "vgpu_memory": 4, "vgpu_cores": 30}, name="VGPU")
task.execute_remotely(queue_name="gpu-4090-hami")
```

A100 MIG 任务不要依赖 `vgpu_cores`，算力隔离由 MIG profile 决定:

```python
task.connect({"vgpu_number": 1, "vgpu_memory": 10}, name="VGPU")
task.execute_remotely(queue_name="gpu-a100-mig")
```

## 宿主机 docker run 注意事项

HAMi/Volcano 不会为宿主机直接运行的 `docker run --gpus ...` 训练预留资源。如果同一张物理 GPU 同时跑非 K8s Docker 训练和 K8s 训练，Volcano 可能认为资源仍然可用，导致 OOM、性能抖动或互相抢卡。

生产上建议二选一:

```text
1. GPU 节点交给 K8s/ClearML 专用，不跑宿主机 docker run 训练。
2. 如果必须混用，在 volcano-vgpu-node-config 的 filterdevices 中排除给 docker run 使用的 GPU。
```
