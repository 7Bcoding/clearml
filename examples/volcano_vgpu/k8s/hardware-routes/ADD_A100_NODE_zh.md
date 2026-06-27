# 新增 A100 节点部署配置流程

本文档用于把一台新的 A100 机器加入当前
ClearML + K3s + Volcano + volcano-vgpu-device-plugin 集群。A100 有两条可落地路线:

```text
路线 A: A100 HAMi-core
  隔离方式: 软件 vGPU
  ClearML queue: gpu-a100-hami / gpu-a100-hami-multinode
  Kubernetes label: gpu.product=a100, gpu.mode=hami-core

路线 B: A100 Dynamic MIG
  隔离方式: NVIDIA MIG 硬件切片
  ClearML queue: gpu-a100-mig / gpu-a100-mig-multinode
  Kubernetes label: gpu.product=a100, gpu.mode=mig
```

如果你希望快速纳入调度并复用现有 4090 HAMi 经验，先选 HAMi-core。
如果你希望更强隔离、更稳定 QoS，选 MIG。两条路线可以在不同 A100 节点池并存。

## 0. 影响范围

新增 A100 节点本身不会影响已有训练。以下操作一般只影响新节点或后续调度:

- 新机器安装 NVIDIA driver、toolkit、k3s-agent。
- 给新节点打 label / taint。
- 新增 Volcano Queue。
- 新增 ClearML agent release。

会影响 A100 本机 GPU 进程的操作:

- `sudo nvidia-smi -i <gpu> -mig 1`
- `sudo nvidia-smi -i <gpu> -mig 0`
- 创建/销毁 MIG geometry。
- `sudo reboot`

这些操作会影响该 GPU 上正在运行的进程，包括宿主机上的
`docker run --gpus ...` 训练。开启 MIG 前必须确认 A100 上没有需要保留的
GPU 训练。

## 1. 选择路线

### 1.1 A100 HAMi-core

适合:

```text
内部可信训练
需要灵活显存和算力百分比
想最小改动纳入 ClearML 调度
暂时不想规划 MIG profile
```

任务侧示例:

```python
task.connect({"vgpu_number": 1, "vgpu_memory": 20, "vgpu_cores": 50}, name="VGPU")
task.execute_remotely(queue_name="gpu-a100-hami")
```

### 1.2 A100 MIG

适合:

```text
多租户隔离
稳定 QoS
推理服务或长期训练
希望硬件级显存/SM 隔离
```

任务侧示例:

```python
task.connect({"vgpu_number": 1, "vgpu_memory": 10}, name="VGPU")
task.execute_remotely(queue_name="gpu-a100-mig")
```

MIG 任务不要依赖 `vgpu_cores`，算力由 MIG profile 决定。

## 2. 控制节点准备

在 K3s server/control-plane 上执行:

```bash
export K3S_SERVER_IP=10.10.36.6
export NEW_NODE=gpu-a100-01
export A100_MODE=hami-core   # 可选: hami-core 或 mig
export CLEARML_NS=clearml
export PLUGIN_NS=kube-system
export VOLCANO_NS=volcano-system
export CLEARML_REPO=/path/to/clearml
export HELM_REPO=/path/to/clearml-helm-charts
export ROUTES=$CLEARML_REPO/examples/volcano_vgpu/k8s/hardware-routes
```

检查当前集群:

```bash
kubectl get nodes -o wide
kubectl get queue
kubectl get runtimeclass
kubectl get pods -A | egrep -i 'clearml|volcano|vgpu|hami|nvidia'
```

备份现有配置:

```bash
mkdir -p ~/clearml-add-a100-backup

kubectl -n $PLUGIN_NS get cm volcano-vgpu-device-config -o yaml \
  > ~/clearml-add-a100-backup/volcano-vgpu-device-config.yaml 2>/dev/null || true

kubectl -n $PLUGIN_NS get cm volcano-vgpu-node-config -o yaml \
  > ~/clearml-add-a100-backup/volcano-vgpu-node-config.yaml 2>/dev/null || true

kubectl get queue -o yaml > ~/clearml-add-a100-backup/queues.yaml
```

取 K3s node token:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

后文用 `<NODE_TOKEN>` 表示这个值。

## 3. A100 裸机准备

在新 A100 机器上执行。

设置唯一 hostname:

```bash
sudo hostnamectl set-hostname gpu-a100-01
hostname
```

基础工具:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg jq pciutils lsb-release
```

确认硬件可见:

```bash
lspci | grep -i nvidia
```

安装 NVIDIA driver。A100 建议使用与你现有 CUDA 12.x 镜像兼容的 datacenter
driver，生产上最好和现有 GPU 节点统一版本。Ubuntu 示例:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-550
sudo reboot
```

重启后验证:

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total,mig.mode.current,mig.mode.pending --format=csv
```

## 4. 安装 NVIDIA Container Toolkit / Runtime

在新 A100 节点执行:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
```

配置 CDI:

```bash
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi
sudo mkdir -p /var/run/cdi
sudo rm -f /var/run/cdi/nvidia*.yaml /var/run/cdi/nvidia*.json
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
sudo nvidia-ctk --debug cdi list
```

根据 `cdi list` 输出设置 `default-kind`。如果输出是
`management.nvidia.com/gpu=GPU-...`:

```bash
sudo nvidia-ctk config --in-place \
  --set nvidia-container-runtime.modes.cdi.default-kind=management.nvidia.com/gpu
```

如果输出是 `nvidia.com/gpu=GPU-...`:

```bash
sudo nvidia-ctk config --in-place \
  --set nvidia-container-runtime.modes.cdi.default-kind=nvidia.com/gpu
```

检查:

```bash
which nvidia-container-runtime
sudo grep -R "mode\|default-kind" /etc/nvidia-container-runtime 2>/dev/null
sudo nvidia-ctk --debug cdi list
```

## 5. 如果选择 A100 MIG，先启用 MIG

如果选择 HAMi-core，跳过本节。

确认没有训练占用 GPU:

```bash
nvidia-smi
docker ps
```

开启 MIG:

```bash
sudo nvidia-smi -i 0 -mig 1
```

如果提示 pending enable、GPU reset 不支持、或被其他客户端占用，按 NVIDIA
MIG 文档要求停止占用 GPU 的服务或直接重启:

```bash
sudo reboot
```

重启后确认:

```bash
nvidia-smi --query-gpu=index,name,mig.mode.current,mig.mode.pending --format=csv
nvidia-smi -L
nvidia-smi mig -lgip
```

如果之后重配 MIG geometry，需要重新生成 CDI:

```bash
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
sudo nvidia-ctk --debug cdi list
```

## 6. 加入 K3s 集群

根据你选择的路线设置 `NODE_MODE`。

HAMi-core:

```bash
export NODE_MODE=hami-core
```

MIG:

```bash
export NODE_MODE=mig
```

在新 A100 节点执行:

```bash
curl -sfL https://get.k3s.io | \
  K3S_URL=https://10.10.36.6:6443 \
  K3S_TOKEN='<NODE_TOKEN>' \
  K3S_NODE_NAME='gpu-a100-01' \
  INSTALL_K3S_EXEC="agent \
    --node-label gpu.present=true \
    --node-label gpu.product=a100 \
    --node-label gpu.mode=${NODE_MODE} \
    --node-label gpu.vendor=nvidia \
    --node-taint gpu.clearml/a100=true:NoSchedule" \
  sh -
```

检查服务:

```bash
sudo systemctl status k3s-agent --no-pager
sudo journalctl -u k3s-agent -n 120 --no-pager
```

确认 K3s 识别 NVIDIA runtime:

```bash
sudo grep -i nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

如果没有任何 `nvidia` runtime 相关内容:

```bash
sudo systemctl restart k3s-agent
sudo grep -i nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

## 7. 控制节点确认新节点

在控制节点执行:

```bash
kubectl get nodes -o wide
kubectl describe node gpu-a100-01 | egrep -i 'gpu.present|gpu.product|gpu.mode|Taints'
```

如果注册时 label/taint 没写对，用 `kubectl` 修正。

A100 HAMi-core:

```bash
kubectl label node gpu-a100-01 \
  gpu.present=true gpu.product=a100 gpu.mode=hami-core gpu.vendor=nvidia \
  --overwrite
```

A100 MIG:

```bash
kubectl label node gpu-a100-01 \
  gpu.present=true gpu.product=a100 gpu.mode=mig gpu.vendor=nvidia \
  --overwrite
```

共同 taint:

```bash
kubectl taint node gpu-a100-01 gpu.clearml/a100=true:NoSchedule --overwrite
```

确认 RuntimeClass:

```bash
kubectl get runtimeclass nvidia
```

如果没有:

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF
```

## 8. Runtime CDI canary

先只验证 K3s runtime:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: runtime-canary-a100
  namespace: default
spec:
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-a100-01
  tolerations:
    - key: gpu.clearml/a100
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["/bin/bash", "-lc"]
      args:
        - |
          nvidia-smi -L
          ldconfig -p | grep -E 'libcuda|libnvidia-ml' || true
EOF

kubectl logs runtime-canary-a100
kubectl describe pod runtime-canary-a100
kubectl delete pod runtime-canary-a100
```

## 9. 配置 volcano-vgpu-device-plugin 节点模式

不要直接覆盖线上 `volcano-vgpu-node-config`，除非你的文件已经包含所有现有节点。
推荐编辑 live ConfigMap:

```bash
kubectl -n kube-system edit cm volcano-vgpu-node-config
```

A100 HAMi-core 节点条目:

```json
{
  "name": "gpu-a100-01",
  "operatingmode": "hami-core",
  "devicememoryscaling": 1.0,
  "devicecorescaling": 1.0,
  "devicesplitcount": 10,
  "migstrategy": "none",
  "filterdevices": {
    "uuid": [],
    "index": []
  }
}
```

A100 MIG 节点条目:

```json
{
  "name": "gpu-a100-01",
  "operatingmode": "mig",
  "devicememoryscaling": 1.0,
  "devicecorescaling": 1.0,
  "devicesplitcount": 7,
  "migstrategy": "mixed",
  "filterdevices": {
    "uuid": [],
    "index": []
  }
}
```

确认 device plugin 全局配置包含:

```yaml
gpuMemoryFactor: 1024
knownMigGeometries:
```

如果未配置，可参考:

```bash
kubectl apply -f $ROUTES/volcano-vgpu-device-config.yaml
```

然后重启 device plugin:

```bash
kubectl -n kube-system rollout restart ds/volcano-device-plugin
kubectl -n kube-system rollout status ds/volcano-device-plugin
```

检查新节点 allocatable:

```bash
kubectl describe node gpu-a100-01 | grep -A12 -B3 volcano.sh
kubectl get node gpu-a100-01 -o json | jq '.status.allocatable'
```

预期看到:

```text
volcano.sh/vgpu-number
volcano.sh/vgpu-memory
volcano.sh/vgpu-cores
```

MIG 路线下，训练任务通常只请求 `vgpu-number` 和 `vgpu-memory`，不依赖
`vgpu-cores`。

## 10. 应用或确认队列

如果已应用过 `hardware-routes/queues.yaml`，只需要检查对应队列。

A100 HAMi-core:

```bash
kubectl get queue gpu-a100-hami gpu-a100-hami-multinode
kubectl describe queue gpu-a100-hami
```

A100 MIG:

```bash
kubectl get queue gpu-a100-mig gpu-a100-mig-multinode
kubectl describe queue gpu-a100-mig
```

如果还没有应用:

```bash
kubectl apply -f $ROUTES/queues.yaml
```

按真实资源调整 `queues.yaml` 里的 capability。查看真实资源:

```bash
kubectl get nodes -l gpu.product=a100 -o json | jq -r '.items[] |
  "\(.metadata.name) mode=\(.metadata.labels["gpu.mode"]) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"])"'
```

## 11. 安装或扩容 ClearML Agent

如果这是第一台 A100 route 节点，按路线安装对应 agent release。

A100 HAMi-core:

```bash
helm upgrade --install clearml-agent-a100-hami "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-hami.yaml"

helm upgrade --install clearml-agent-a100-hami-multinode "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-hami-multinode.yaml"
```

A100 MIG:

```bash
helm upgrade --install clearml-agent-a100-mig "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-mig.yaml"

helm upgrade --install clearml-agent-a100-mig-multinode "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-a100-mig-multinode.yaml"
```

如果 route agent 已存在，新节点通常不需要新增 agent；同一个 agent 会继续
监听 ClearML queue，并由 Kubernetes/Volcano 把任务调度到符合 label 的节点。

确认:

```bash
kubectl -n clearml get deploy | grep a100
kubectl -n clearml logs deploy/clearml-agent-a100-hami --tail=120 2>/dev/null || true
kubectl -n clearml logs deploy/clearml-agent-a100-mig --tail=120 2>/dev/null || true
```

ClearML WebUI 中应存在你选择路线的 queue:

```text
gpu-a100-hami
gpu-a100-hami-multinode
gpu-a100-mig
gpu-a100-mig-multinode
```

## 12. A100 HAMi-core canary

如果选择 HAMi-core，执行:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: hami-canary-a100
  namespace: default
  annotations:
    volcano.sh/vgpu-mode: "hami-core"
spec:
  schedulerName: volcano
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-a100-01
    gpu.product: a100
    gpu.mode: hami-core
  tolerations:
    - key: gpu.clearml/a100
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["/bin/bash", "-lc"]
      args:
        - |
          nvidia-smi -L
          nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
      resources:
        limits:
          volcano.sh/vgpu-number: "1"
          volcano.sh/vgpu-memory: "20"
          volcano.sh/vgpu-cores: "50"
        requests:
          cpu: "8"
          memory: 32Gi
          volcano.sh/vgpu-number: "1"
          volcano.sh/vgpu-memory: "20"
          volcano.sh/vgpu-cores: "50"
EOF

kubectl get pod hami-canary-a100 -o wide
kubectl logs hami-canary-a100
kubectl describe pod hami-canary-a100
kubectl delete pod hami-canary-a100
```

成功特征:

- Pod 运行在 `gpu-a100-01`。
- 容器内 `nvidia-smi` 正常。
- `memory.total` 接近请求的 `20 GiB`。

## 13. A100 MIG canary

如果选择 MIG，执行:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: mig-canary-a100
  namespace: default
  annotations:
    volcano.sh/vgpu-mode: "mig"
spec:
  schedulerName: volcano
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-a100-01
    gpu.product: a100
    gpu.mode: mig
  tolerations:
    - key: gpu.clearml/a100
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["/bin/bash", "-lc"]
      args:
        - |
          nvidia-smi -L
          nvidia-smi
      resources:
        limits:
          volcano.sh/vgpu-number: "1"
          volcano.sh/vgpu-memory: "10"
        requests:
          cpu: "8"
          memory: 32Gi
          volcano.sh/vgpu-number: "1"
          volcano.sh/vgpu-memory: "10"
EOF

kubectl get pod mig-canary-a100 -o wide
kubectl logs mig-canary-a100
kubectl describe pod mig-canary-a100
kubectl delete pod mig-canary-a100
```

成功特征:

- Pod 运行在 `gpu-a100-01`。
- 容器内只看到分配到的 MIG 设备或 MIG-backed CUDA 设备。
- 没有 `unresolvable CDI devices` 事件。

## 14. ClearML 任务验证

A100 HAMi-core 单机验证:

```bash
cd $CLEARML_REPO/examples/volcano_vgpu

python verify/vgpu_per_task_test.py \
  --queue gpu-a100-hami \
  --memory 20 \
  --cores 50
```

A100 HAMi-core 多机:

```bash
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue gpu-a100-hami-multinode \
  --vgpu-number 1 \
  --vgpu-memory 80 \
  --vgpu-cores 100
```

A100 MIG 单机可用最小 Python 模板提交:

```python
from clearml import Task

Task.force_store_standalone_script(True)

task = Task.init(
    project_name="volcano-vgpu",
    task_name="a100-mig-smoke",
    task_type=Task.TaskTypes.testing,
)

task.connect({"vgpu_number": 1, "vgpu_memory": 10}, name="VGPU")
task.execute_remotely(queue_name="gpu-a100-mig", exit_process=True)

import subprocess
print(subprocess.check_output(["nvidia-smi", "-L"], text=True))
```

A100 MIG 多机 smoke 如果脚本参数当前仍要求 `--vgpu-cores`，可以临时给
`100`，但平台语义上不依赖它:

```bash
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue gpu-a100-mig-multinode \
  --vgpu-number 1 \
  --vgpu-memory 10 \
  --vgpu-cores 100
```

检查:

```bash
kubectl -n clearml get pods -o wide --sort-by=.metadata.creationTimestamp
kubectl -n clearml get events --sort-by=.lastTimestamp | tail -120
```

## 15. 宿主机 docker run 混用风险

如果 A100 节点上还有宿主机 `docker run --gpus ...` 训练，Volcano/HAMi 不会
可靠感知这些非 K8s 进程占用的 GPU 资源。可能出现:

```text
K8s 认为 GPU 还有资源
实际 GPU 已被 docker run 占用
新 Pod 调度成功但训练 OOM 或严重抖动
```

生产建议:

```text
1. A100 节点交给 K8s/ClearML 专用。
2. 如果必须混用，在 volcano-vgpu-node-config 的 filterdevices 中排除给 docker run 使用的 GPU。
```

## 16. 回滚

先停止调度到该节点:

```bash
kubectl cordon gpu-a100-01
```

移除 route label:

```bash
kubectl label node gpu-a100-01 gpu.mode- --overwrite
```

从 `volcano-vgpu-node-config` 中删除该节点条目后:

```bash
kubectl -n kube-system rollout restart ds/volcano-device-plugin
```

如果是 MIG 路线，需要关闭 MIG 时，必须确认 GPU 空闲:

```bash
sudo nvidia-smi -i 0 -mig 0
sudo reboot
```

如需完全移除 K3s agent，在新节点执行:

```bash
sudo /usr/local/bin/k3s-agent-uninstall.sh
```

然后在控制节点清理 Node 对象:

```bash
kubectl delete node gpu-a100-01
```

## 17. 参考

- K3s agent 加入集群: https://docs.k3s.io/quick-start
- K3s agent label/taint 与 NVIDIA runtime: https://docs.k3s.io/cli/agent
- K3s NVIDIA runtime 自动发现: https://docs.k3s.io/advanced
- NVIDIA Container Toolkit 安装: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA CDI: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
- NVIDIA MIG: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html
