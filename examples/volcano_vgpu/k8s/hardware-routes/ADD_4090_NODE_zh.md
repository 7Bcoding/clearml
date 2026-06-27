# 新增 4090 节点部署配置流程

本文档用于把一台新的 RTX 4090 机器加入当前
ClearML + K3s + Volcano + volcano-vgpu-device-plugin 集群，并作为
`hami-core` 软件 vGPU 节点使用。

目标路线:

```text
GPU: RTX 4090
隔离模式: HAMi-core / volcano.sh/vgpu-mode=hami-core
ClearML queue: gpu-4090-hami
ClearML multinode queue: gpu-4090-hami-multinode
Kubernetes label: gpu.product=4090, gpu.mode=hami-core
Kubernetes taint: gpu.clearml/4090=true:NoSchedule
```

## 0. 影响范围

新增节点本身不会影响已有训练。以下操作一般只影响新节点或后续调度:

- 新机器安装 NVIDIA driver、toolkit、k3s-agent。
- 给新节点打 label / taint。
- 新增 Volcano Queue。
- 新增 ClearML agent release。

需要谨慎的操作:

- `kubectl -n kube-system rollout restart ds/volcano-device-plugin` 会短暂影响新 GPU 资源注册和后续调度，一般不会杀已运行训练 Pod。
- `kubectl -n volcano-system rollout restart deploy/volcano-scheduler` 会短暂影响新任务调度，不会主动杀已运行 Pod。
- 不要在已有生产节点上重启 Docker daemon；本文只对新 4090 节点操作。

## 1. 控制节点准备

在 K3s server/control-plane 上执行:

```bash
export K3S_SERVER_IP=10.10.36.6
export NEW_NODE=gpu-4090-01
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
mkdir -p ~/clearml-add-4090-backup

kubectl -n $PLUGIN_NS get cm volcano-vgpu-device-config -o yaml \
  > ~/clearml-add-4090-backup/volcano-vgpu-device-config.yaml 2>/dev/null || true

kubectl -n $PLUGIN_NS get cm volcano-vgpu-node-config -o yaml \
  > ~/clearml-add-4090-backup/volcano-vgpu-node-config.yaml 2>/dev/null || true

kubectl get queue -o yaml > ~/clearml-add-4090-backup/queues.yaml
```

取 K3s node token:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

后文用 `<NODE_TOKEN>` 表示这个值。

## 2. 4090 裸机准备

在新 4090 机器上执行。

设置唯一 hostname:

```bash
sudo hostnamectl set-hostname gpu-4090-01
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

安装 NVIDIA driver。生产建议使用与你现有 4090 节点一致的 driver 版本。
Ubuntu 示例:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-550
sudo reboot
```

重启后验证:

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
```

预期能看到 RTX 4090，单卡显存约 24GiB。

## 3. 安装 NVIDIA Container Toolkit / Runtime

在新 4090 节点执行。这里使用 NVIDIA 官方 apt 仓库。

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

配置 NVIDIA runtime 为 CDI 模式:

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

## 4. 加入 K3s 集群

K3s 会在启动时自动探测 `nvidia` runtime。官方文档也说明了
`--node-label` 和 `--node-taint` 只在注册时生效；后续修改请用 `kubectl`。

在新 4090 节点执行:

```bash
curl -sfL https://get.k3s.io | \
  K3S_URL=https://10.10.36.6:6443 \
  K3S_TOKEN='<NODE_TOKEN>' \
  K3S_NODE_NAME='gpu-4090-01' \
  INSTALL_K3S_EXEC='agent \
    --node-label gpu.present=true \
    --node-label gpu.product=4090 \
    --node-label gpu.mode=hami-core \
    --node-label gpu.vendor=nvidia \
    --node-taint gpu.clearml/4090=true:NoSchedule' \
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

## 5. 控制节点确认新节点

在控制节点执行:

```bash
kubectl get nodes -o wide
kubectl describe node gpu-4090-01 | egrep -i 'gpu.present|gpu.product|gpu.mode|Taints'
```

如果注册时 label/taint 没写对，用 `kubectl` 修正:

```bash
kubectl label node gpu-4090-01 \
  gpu.present=true gpu.product=4090 gpu.mode=hami-core gpu.vendor=nvidia \
  --overwrite

kubectl taint node gpu-4090-01 gpu.clearml/4090=true:NoSchedule --overwrite
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

## 6. Runtime CDI canary

先只验证 K3s runtime，不验证 HAMi/Volcano:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: runtime-canary-4090
  namespace: default
spec:
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-4090-01
  tolerations:
    - key: gpu.clearml/4090
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

kubectl logs runtime-canary-4090
kubectl describe pod runtime-canary-4090
kubectl delete pod runtime-canary-4090
```

这一步成功后，再进入 HAMi/Volcano 配置。

## 7. 配置 volcano-vgpu-device-plugin 节点模式

4090 节点使用 `hami-core`。不要直接覆盖线上
`volcano-vgpu-node-config`，除非你的文件已经包含所有现有节点。

推荐做法是编辑 live ConfigMap:

```bash
kubectl -n kube-system edit cm volcano-vgpu-node-config
```

给新节点增加或确认类似条目:

```json
{
  "name": "gpu-4090-01",
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

如果你的 ConfigMap 是 YAML 结构而不是 JSON 结构，也保持同样语义:

```yaml
name: gpu-4090-01
operatingmode: hami-core
devicesplitcount: 10
migstrategy: none
```

重启 device plugin，让新节点资源重新注册:

```bash
kubectl -n kube-system rollout restart ds/volcano-device-plugin
kubectl -n kube-system rollout status ds/volcano-device-plugin
```

检查新节点 allocatable:

```bash
kubectl describe node gpu-4090-01 | grep -A12 -B3 volcano.sh
kubectl get node gpu-4090-01 -o json | jq '.status.allocatable'
```

预期看到:

```text
volcano.sh/vgpu-number
volcano.sh/vgpu-memory
volcano.sh/vgpu-cores
```

## 8. 应用或确认队列

如果已应用过 `hardware-routes/queues.yaml`，只需要检查:

```bash
kubectl get queue gpu-4090-hami gpu-4090-hami-multinode
kubectl describe queue gpu-4090-hami
kubectl describe queue gpu-4090-hami-multinode
```

如果还没有应用:

```bash
kubectl apply -f $ROUTES/queues.yaml
```

按真实资源调整 `queues.yaml` 里的 capability。查看真实资源:

```bash
kubectl get nodes -l gpu.product=4090,gpu.mode=hami-core -o json | jq -r '.items[] |
  "\(.metadata.name) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"])"'
```

## 9. 安装或扩容 ClearML Agent

如果这是第一台 4090 route 节点，安装两个 agent release:

```bash
helm upgrade --install clearml-agent-4090-hami "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-4090-hami.yaml"

helm upgrade --install clearml-agent-4090-hami-multinode "$HELM_REPO/charts/clearml-agent" \
  -n clearml -f "$ROUTES/agent-values-4090-hami-multinode.yaml"
```

如果 route agent 已存在，新节点通常不需要新增 agent；同一个 agent 会继续
监听 ClearML queue，并由 Kubernetes/Volcano 把任务调度到符合 label 的节点。

确认:

```bash
kubectl -n clearml get deploy | grep 4090
kubectl -n clearml logs deploy/clearml-agent-4090-hami --tail=120
```

ClearML WebUI 中应存在:

```text
gpu-4090-hami
gpu-4090-hami-multinode
```

## 10. HAMi canary

为了确认新节点可用，建议临时 pin 到该节点:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: hami-canary-4090
  namespace: default
  annotations:
    volcano.sh/vgpu-mode: "hami-core"
spec:
  schedulerName: volcano
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-4090-01
    gpu.product: "4090"
    gpu.mode: hami-core
  tolerations:
    - key: gpu.clearml/4090
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
          volcano.sh/vgpu-memory: "4"
          volcano.sh/vgpu-cores: "30"
        requests:
          cpu: "4"
          memory: 16Gi
          volcano.sh/vgpu-number: "1"
          volcano.sh/vgpu-memory: "4"
          volcano.sh/vgpu-cores: "30"
EOF

kubectl get pod hami-canary-4090 -o wide
kubectl logs hami-canary-4090
kubectl describe pod hami-canary-4090
kubectl delete pod hami-canary-4090
```

成功特征:

- Pod 运行在 `gpu-4090-01`。
- 容器内 `nvidia-smi` 正常。
- `memory.total` 接近请求的 `4 GiB`。

## 11. ClearML 任务验证

在开发机或控制机提交单机验证:

```bash
cd $CLEARML_REPO/examples/volcano_vgpu

python verify/vgpu_per_task_test.py \
  --queue gpu-4090-hami \
  --memory 4 \
  --cores 30
```

多机 smoke 示例:

```bash
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue gpu-4090-hami-multinode \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

检查:

```bash
kubectl -n clearml get pods -o wide --sort-by=.metadata.creationTimestamp
kubectl -n clearml get events --sort-by=.lastTimestamp | tail -120
```

## 12. 回滚

如果新节点验证失败，先停止把任务调度到该节点:

```bash
kubectl cordon gpu-4090-01
```

移除 ClearML route label 或改成维护状态:

```bash
kubectl label node gpu-4090-01 gpu.mode- --overwrite
```

从 `volcano-vgpu-node-config` 中删除该节点条目后:

```bash
kubectl -n kube-system rollout restart ds/volcano-device-plugin
```

如需完全移除 K3s agent，在新节点执行:

```bash
sudo /usr/local/bin/k3s-agent-uninstall.sh
```

然后在控制节点清理 Node 对象:

```bash
kubectl delete node gpu-4090-01
```

## 13. 参考

- K3s agent 加入集群: https://docs.k3s.io/quick-start
- K3s agent label/taint 与 NVIDIA runtime: https://docs.k3s.io/cli/agent
- K3s NVIDIA runtime 自动发现: https://docs.k3s.io/advanced
- NVIDIA Container Toolkit 安装: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA CDI: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
