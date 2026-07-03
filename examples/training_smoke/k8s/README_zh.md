# LLM 微调 ClearML Agent 安装示例

本目录给出 `ClearML + Volcano + volcano-vgpu-device-plugin/HAMi-core` 下的 LLM 微调队列安装示例。

文件说明：

```text
pvc-ai-data.example.yaml
  共享模型/数据/输出目录 PVC 示例，面向未来 NAS/NFS/CSI。

local-pv-ai-data.example.yaml
  host local PV 示例，适合冒烟测试先使用单台 GPU 机器本地盘。

volcano-queue-llm-finetune-vgpu.yaml
  Volcano Queue 示例，资源语义使用 volcano.sh/vgpu-*。

clearml-agent-llm-finetune-vgpu-values.yaml
  ClearML Agent Helm values，监听 llm-finetune-vgpu 队列。

clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml
  host local PV 场景的 Helm overlay，用 nodeSelector 把训练 Pod 固定到本地 PV 所在节点。
```

## 安装顺序

1. 创建或确认已有 ClearML Agent 凭据 Secret：

```bash
kubectl -n clearml get secret clearml-agent-creds
```

如果没有，需要先按你们现有方式创建 `clearml-agent-creds`。

2. 创建数据 PVC。

如果先用 host local PV，推荐用：

```bash
# 先在目标 GPU 节点上创建目录：
# sudo mkdir -p /data/ai-local/{models,datasets,output,cache,clearml}
# sudo chmod -R 0777 /data/ai-local

# 修改 local-pv-ai-data.example.yaml 里的 <GPU_NODE_HOSTNAME>
kubectl apply -f examples/training_smoke/k8s/local-pv-ai-data.example.yaml
```

如果未来切到 NAS/NFS/CSI，再用：

```bash
kubectl apply -f examples/training_smoke/k8s/pvc-ai-data.example.yaml
```

3. 创建 Volcano Queue：

```bash
kubectl apply -f examples/training_smoke/k8s/volcano-queue-llm-finetune-vgpu.yaml
```

4. 安装 LLM 微调 ClearML Agent。

host local PV 场景：

```bash
# 修改 localpv overlay 里的 <GPU_NODE_HOSTNAME>
helm upgrade --install clearml-agent-llm-finetune \
  clearml/clearml-agent \
  -n clearml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml
```

NAS/NFS/CSI PVC 场景：

```bash
helm repo add clearml https://clearml.github.io/clearml-helm-charts
helm repo update

helm upgrade --install clearml-agent-llm-finetune \
  clearml/clearml-agent \
  -n clearml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml
```

## 需要按环境修改的地方

必须修改：

```text
apiServerUrlReference
fileServerUrlReference
webServerUrlReference
defaultContainerImage
ai-data-pvc
nodeSelector
Volcano Queue capability
```

如果使用 host local PV：

```text
local-pv-ai-data.example.yaml 里的 <GPU_NODE_HOSTNAME>
clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml 里的 <GPU_NODE_HOSTNAME>
local PV path: /data/ai-local
local PV capacity: 2Ti
```

如果你们已经有 NAS/NFS/MinIO CSI 对应 PVC，直接把 values 里的 `claimName: ai-data-pvc` 改成真实 PVC 名称。

## host local PV 注意事项

host local PV 适合冒烟测试，但有几个限制：

```text
数据只在一台 GPU 节点本地
训练 Pod 必须调度到这台节点
多节点训练无法共享这块本地盘
节点故障会影响数据可用性
换节点需要重新同步模型和数据
```

因此本方案里用 overlay 明确加：

```yaml
nodeSelector:
  kubernetes.io/hostname: <GPU_NODE_HOSTNAME>
```

模型和数据需要先放到该节点：

```bash
sudo mkdir -p /data/ai-local/models /data/ai-local/datasets /data/ai-local/output
```

Pod 内看到的是：

```text
/data/models
/data/datasets
/data/output
```

## 路径约定

该模板把共享 PVC 挂载到训练 Pod 的：

```text
/data
```

LLM 训练脚本中应使用：

```text
/data/models/...
/data/datasets/...
/data/output/...
/data/cache/huggingface
/data/cache/modelscope
```

不要填写提交机本地路径。

## 资源语义

该队列是 HAMi/vGPU-only 集群示例，使用：

```text
volcano.sh/vgpu-number
volcano.sh/vgpu-memory
volcano.sh/vgpu-cores
```

不要在这个队列里使用：

```text
nvidia.com/gpu
```

算法任务通过 ClearML 的 `CONFIGURATION -> VGPU` 段控制资源，例如：

```python
task.connect(
    {"vgpu_number": 1, "vgpu_memory": 24, "vgpu_cores": 100},
    name="VGPU",
)
```
