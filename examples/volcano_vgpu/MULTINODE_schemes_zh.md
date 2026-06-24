# 多机多卡训练：ClearML 冒烟 + Volcano Job 推荐路线

本文档面向当前集群形态：

- K3s + containerd
- Volcano scheduler
- `volcano-vgpu-device-plugin` / HAMi-core
- GPU 节点以 `volcano.sh/vgpu-*` 暴露资源，`nvidia.com/gpu` 可能为 0
- NVIDIA runtime 使用 CDI，训练 Pod 保留 `runtimeClassName: nvidia`
- `volcano-vgpu-device-plugin` 不强制开启 `cdi.enabled=true`

当前只保留两条路线：

| 路线 | 用途 | gang | 是否经过 ClearML Agent |
|---|---|---:|---:|
| 方案 1：`launch_multi_node` | ClearML 原生多机冒烟、验证 NCCL 与队列 | 否 | 是 |
| 方案 3：Volcano Job | 推荐的 gang 调度与生产化方向 | 是 | 否 |

旧的「PodGroup + N 个 ClearML Task」路线已移除。原因是它需要静态 PodGroup、Agent 模板注解和任务数量严格一致；并发、重试、不同节点数任务都会变得脆弱。需要 gang 时优先走 Volcano Job。

---

## 0. 路径约定

Linux / macOS：

```bash
export CLEARML_REPO=/path/to/clearml
export HELM_REPO=/path/to/clearml-helm-charts
cd "$CLEARML_REPO/examples/volcano_vgpu"
```

Windows PowerShell：

```powershell
$env:CLEARML_REPO = "D:\Python\clearml"
$env:HELM_REPO = "D:\Python\clearml-helm-charts"
cd "$env:CLEARML_REPO\examples\volcano_vgpu"
```

以下默认命名空间为 `clearml`。

---

## 1. 文件清单

| 文件 | 用途 |
|---|---|
| `train/multinode_launch_smoke.py` | 方案 1：ClearML `launch_multi_node` 冒烟 |
| `train/multinode_launch_ddp_train.py` | 方案 1：ClearML `launch_multi_node` 多机 DDP 训练 |
| `submit/submit_volcano_job.py` | 方案 3：生成 ConfigMap + Volcano Job |
| `train/volcano_job_nccl_smoke.py` | 方案 3：Job Pod 内 NCCL 冒烟入口 |
| `train/volcano_job_ddp_train.py` | 方案 3：Volcano Job 多机 DDP 训练模板 |
| `k8s/agent-values-multinode-launch-vgpu.yaml` | 方案 1：HAMi/vGPU ClearML Agent values |
| `k8s/queue-multinode-vgpu.yaml` | HAMi/vGPU Volcano Queue |
| `k8s/job-template-volcano-vgpu.yaml` | 方案 3：静态 Volcano Job 模板 |
| `k8s/CLEARML_K3S_VOLCANO_VGPU_DEPLOYMENT_zh.md` | 当前完整部署指南 |

仍保留在仓库里的 `full-gpu`、`podgroup`、`service` 示例只作为历史/对照材料；当前 HAMi/vGPU 路线不要照它们操作。

---

## 2. 平台前置检查

### 2.1 基础工具

```bash
kubectl version --client
kubectl cluster-info
helm version
python --version
pip show clearml
```

ClearML 本地连通性只影响方案 1，以及方案 3 的可选 WebUI scalar：

```bash
python -c "from clearml.backend_api.session.client import APIClient; print('OK', APIClient().users.get_current_user())"
```

### 2.2 Volcano 组件

```bash
kubectl get pods -n volcano-system
kubectl get crd | grep volcano
kubectl get crd jobs.batch.volcano.sh queues.scheduling.volcano.sh
```

必须能看到 Volcano scheduler/controller Running，并存在 `jobs.batch.volcano.sh`、`queues.scheduling.volcano.sh`。

### 2.3 GPU 资源类型

```bash
kubectl get nodes -o json | jq -r '.items[] |
  "\(.metadata.name) nvidia.com/gpu=\(.status.allocatable["nvidia.com/gpu"]) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"]) vgpu-cores=\(.status.allocatable["volcano.sh/vgpu-cores"])"'
```

当前推荐状态：

```text
nvidia.com/gpu=0 或不用它
volcano.sh/vgpu-number 有值
volcano.sh/vgpu-memory 有值
volcano.sh/vgpu-cores 有值
```

在这种集群里，「每 Pod 一张整卡」表达为：

```yaml
volcano.sh/vgpu-number: "1"
volcano.sh/vgpu-memory: "24"   # 4090 示例；A100-80G 改 80
volcano.sh/vgpu-cores: "100"
```

### 2.4 Runtime CDI 状态

每个 GPU 节点检查：

```bash
nvidia-smi -L
sudo nvidia-ctk --debug cdi list
sudo grep -R "mode\|default-kind\|management.nvidia\|nvidia.com/gpu" \
  /etc/nvidia-container-runtime \
  /usr/local/nvidia/toolkit/.config/nvidia-container-runtime \
  /etc/cdi /var/run/cdi 2>/dev/null
```

当前建议：

```text
runtime mode = cdi
runtime default-kind 与 nvidia-ctk --debug cdi list 输出一致
当前环境通常是 management.nvidia.com/gpu
```

注意：这里要求的是 NVIDIA runtime CDI 正常，不要求 `volcano-vgpu-device-plugin cdi.enabled=true`。插件继续负责 `volcano.sh/vgpu-*` 调度与 HAMi-core 注入。

---

## 3. 共用平台准备

### 3.1 给 GPU 节点打 label

```bash
kubectl label node gpu-server-5 gpu.present=true --overwrite
kubectl label node gpu-server-6 gpu.present=true --overwrite
kubectl get nodes -l gpu.present=true -o wide
```

### 3.2 创建 Volcano Queue

HAMi/vGPU 集群使用 vGPU 版队列：

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
kubectl apply -f k8s/queue-multinode-vgpu.yaml
kubectl get queue multinode-full-gpu
kubectl describe queue multinode-full-gpu
```

验证 `capability` 里是 `volcano.sh/vgpu-*`，不是 `nvidia.com/gpu`：

```bash
kubectl describe queue multinode-full-gpu | egrep -i 'volcano.sh/vgpu|nvidia.com/gpu'
```

如果看到 `nvidia.com/gpu`，说明误用了原生整卡队列文件，需要重新 apply vGPU 版队列。

---

## 4. 方案 1：ClearML `launch_multi_node` 冒烟

方案 1 用来验证 ClearML 队列、k8s-glue、runtime CDI、vGPU 注入和 NCCL 基本链路。它不是 gang 调度，master Pod 通常先启动，worker Pod 后启动。

### 4.1 创建 ClearML Queue

WebUI：

1. Workers & Queues
2. 新建队列 `multinode-full-gpu`

或 Python：

```bash
python -c "
from clearml.backend_api.session.client import APIClient
api = APIClient()
q = 'multinode-full-gpu'
try:
    api.queues.create(name=q)
    print('created', q)
except Exception as e:
    print('queue may exist:', e)
print([x.name for x in api.queues.get_all()])
"
```

### 4.2 部署 multinode Agent

在 Helm chart 仓库执行：

```bash
cd "$HELM_REPO"

helm upgrade --install clearml-agent-multinode charts/clearml-agent -n clearml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-multinode-launch-vgpu.yaml"

kubectl rollout status deploy/clearml-agent-multinode -n clearml
kubectl logs -n clearml deploy/clearml-agent-multinode --tail=80
```

关键 values 应为：

```yaml
agentk8sglue:
  queue: multinode-full-gpu
  createQueueIfNotExists: false
  vgpuHook:
    enabled: true
  basePodTemplate:
    schedulerName: volcano
    runtimeClassName: nvidia
    annotations:
      volcano.sh/vgpu-mode: hami-core
      volcano.sh/queue-name: multinode-full-gpu
```

不要叠加 `examples/volcano-vgpu/custom_values.yaml`，避免把单机队列配置混入 multinode Agent。

### 4.3 提交冒烟任务

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

验证：

```bash
kubectl get pods -n clearml -o wide --sort-by=.metadata.creationTimestamp
kubectl logs -n clearml <master-pod-name> --tail=80
kubectl logs -n clearml <worker-pod-name> --tail=80
```

ClearML WebUI 中 master Task 应看到：

```text
smoke/allreduce_sum = 3.0
allreduce_ok = 1
```

### 4.4 提交真实 DDP 训练任务

方案 1 也可以跑真实训练，只是它不提供 gang 调度语义。先确认 4.3 的冒烟通过，再提交训练模板：

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python train/multinode_launch_ddp_train.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --epochs 5 \
  --batch-size 128 \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

预期在 ClearML WebUI 的 master Task 中看到：

```text
SCALARS -> train/loss
SCALARS -> train/accuracy
MODELS  -> launch-multinode-ddp
ARTIFACTS -> checkpoint
```

### 4.5 方案 1 常见问题

`unresolvable CDI devices management.nvidia.com/gpu=GPU-...`：

```bash
kubectl get pod <pod> -n clearml -o yaml | egrep -i 'runtimeClassName|volcano.sh/vgpu|NVIDIA_VISIBLE_DEVICES|management.nvidia'
sudo nvidia-ctk --debug cdi list
```

修 runtime CDI，不要删除 CDI 文件后继续跑。确认两台 GPU 节点都能列出本机 GPU UUID，且 runtime `default-kind` 与 `cdi list` 一致。

`PodGroup ... unschedulable`：

方案 1 不依赖手工 PodGroup。若事件里出现自动 PodGroup，优先看真正失败的训练 Pod 事件，通常是 worker 容器创建失败后的连带现象。

NCCL 跨节点失败：

```bash
kubectl exec -n clearml deploy/clearml-agent-multinode -- ip route
kubectl -n clearml exec <pod> -- ip addr
```

如实际网卡不是 `eth0`，修改 `agent-values-multinode-launch-vgpu.yaml` 中的：

```yaml
- name: NCCL_SOCKET_IFNAME
  value: "<实际网卡名>"
```

然后重新 `helm upgrade` 并重提任务。

`Timed out after 601 seconds waiting for clients. 1/2 clients joined`：

这表示 master 已经进入 `torch.distributed.init_process_group`，但 worker 没有在超时时间内加入。它通常不是 master 本身的问题，而是 worker Pod 没有跑到同一行代码。

先查 worker 是否真的创建并运行：

```bash
kubectl get pods -n clearml -o wide --sort-by=.metadata.creationTimestamp
kubectl get events -n clearml --sort-by=.lastTimestamp | tail -120
```

再查 worker 日志和 Pod 事件：

```bash
kubectl logs -n clearml <worker-pod-name> --tail=120
kubectl describe pod -n clearml <worker-pod-name>
```

重点看：

```text
ImagePullBackOff / ContainerCreating 时间过长
pip install torch/clearml 太慢
Unschedulable / vGPU 资源不够
failed to inject CDI devices
CUDA is not available
MASTER_ADDR 无法访问
```

如果只是 worker 启动慢，可以先增大脚本的 rendezvous 等待时间：

```bash
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --dist-timeout-sec 1800
```

如果 worker 长期 Pending 或 CrashLoop，增大 timeout 没意义，先修 worker Pod 事件。方案 1 没有 gang 语义，master 先启动是正常现象；需要严格齐射时使用方案 3 Volcano Job。

---

## 5. 方案 3：Volcano Job gang 推荐路线

方案 3 是当前推荐的 gang 路线。它绕过 ClearML k8s-glue，直接创建 Volcano Job：

```text
submit/submit_volcano_job.py
  -> 创建可选 ClearML Task
  -> 创建 ConfigMap，挂载 train/volcano_job_nccl_smoke.py 或 train/volcano_job_ddp_train.py
  -> 创建 Volcano Job
  -> Volcano env/svc 插件注入 rank 与 DNS
  -> 每个 Pod 申请 volcano.sh/vgpu-*，runtimeClassName=nvidia
```

### 5.1 可行性结论

当前方案 3 **可行，但定位是平台提交器/冒烟器，不是完整 ClearML Agent 替代品**。

已经满足的条件：

- `submit/submit_volcano_job.py` 默认 `--gpu-mode vgpu`，渲染 `volcano.sh/vgpu-*`
- Job Pod 使用 `schedulerName: volcano`
- Job Pod 使用 `runtimeClassName: nvidia`，匹配当前 runtime CDI 路线
- `train/volcano_job_nccl_smoke.py` 严格要求 CUDA 可用，不再静默退回 CPU
- `train/volcano_job_ddp_train.py` 已提供最小 DDP 训练闭环：DataLoader、DistributedSampler、DDP、loss/accuracy、rank0 checkpoint
- ClearML scalar 上报是弱集成；没有凭证时不会影响 NCCL 冒烟
- 默认镜像已改为 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`，避免 `nvidia/cuda:runtime` 中缺 Python/pip/torch 的问题

必须现场验证的条件：

| 条件 | 验证命令 |
|---|---|
| Volcano Job CRD 存在 | `kubectl get crd jobs.batch.volcano.sh` |
| env/svc 插件生效 | Pod 内有 `VK_TASK_INDEX` 或 `VC_TASK_INDEX`；`MASTER_ADDR` 可解析 |
| Queue 是 vGPU 版 | `kubectl describe queue multinode-full-gpu` |
| 两台节点 runtime CDI 正常 | `sudo nvidia-ctk --debug cdi list` |
| PyTorch 镜像可拉取 | `kubectl describe pod <job-pod> -n clearml` |
| NCCL 网卡正确 | Pod 日志和 `NCCL_SOCKET_IFNAME` |

### 5.2 方案 3 当前边界

方案 3a 当前已经解决：

- gang 齐射
- 每 Pod 一进程
- vGPU 整卡或切片资源申请
- NCCL all-reduce 冒烟
- 最小 DDP 训练模板
- 可选 ClearML Task 指标

它暂不解决：

- ClearML Agent 的自动代码打包、依赖复现、docker build
- 数据集、PVC、Secrets、环境变量的通用模板化
- 多角色 Job，例如 master/worker/evaluator 分离
- 每 Pod 多 GPU / 每 Pod 多进程 `torchrun --nproc_per_node`
- 失败重试后的 ClearML 状态自动同步

这些属于后面的生产化开发计划。

### 5.3 预览 Volcano Job

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python submit/submit_volcano_job.py --num-nodes 2 --dry-run
```

默认生成 HAMi/vGPU Job：

```yaml
runtimeClassName: nvidia
resources:
  limits:
    volcano.sh/vgpu-number: "1"
    volcano.sh/vgpu-memory: "24"
    volcano.sh/vgpu-cores: "100"
```

### 5.4 提交 gang 冒烟

```bash
python submit/submit_volcano_job.py \
  --num-nodes 2 \
  --apply
```

提交真实 DDP 训练模板：

```bash
python submit/submit_volcano_job.py \
  --num-nodes 2 \
  --apply \
  --script volcano_job_ddp_train.py \
  --script-args "--epochs 5 --batch-size 128 --lr 0.001"
```

如果要测试切片：

```bash
python submit/submit_volcano_job.py \
  --num-nodes 2 \
  --apply \
  --script volcano_job_ddp_train.py \
  --script-args "--epochs 3 --batch-size 128" \
  --vgpu-memory 4 \
  --vgpu-cores 30
```

如果要写 ClearML WebUI scalar：

```bash
python submit/submit_volcano_job.py \
  --num-nodes 2 \
  --apply \
  --with-clearml-task \
  --script volcano_job_ddp_train.py \
  --script-args "--epochs 5"
```

注意：`--with-clearml-task` 只是创建 Task 并把 ID 注入 Job。Pod 内仍需要能访问 ClearML Server，且镜像内有 ClearML 包，或你自行扩展镜像/命令安装它。否则脚本只打印 warning，不影响 NCCL 冒烟。

### 5.5 观察 Job

```bash
export JOB_NAME=clearml-vjob-xxxxxxxx

kubectl get job "$JOB_NAME" -n clearml
kubectl get pods -n clearml -l volcano.sh/job-name="$JOB_NAME" -o wide -w
kubectl get podgroup -n clearml | grep "$JOB_NAME"
```

预期：

```text
2 个 Pod 同时 Pending
资源满足后同时 Running
日志出现 rank=0/rank=1 DONE ok=True
allreduce sum=3.0
```

查看日志：

```bash
kubectl logs -n clearml -l volcano.sh/job-name="$JOB_NAME" --all-containers --prefix --tail=120
```

### 5.6 验证 vGPU/CDI 注入

```bash
POD=$(kubectl get pods -n clearml -l volcano.sh/job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}')

kubectl get pod "$POD" -n clearml -o yaml | egrep -i 'runtimeClassName|volcano.sh/vgpu|NVIDIA_VISIBLE_DEVICES|LD_PRELOAD|MASTER_ADDR|VK_TASK_INDEX|VC_TASK_INDEX'
kubectl exec -n clearml "$POD" -- python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
kubectl exec -n clearml "$POD" -- getent hosts "$JOB_NAME-worker-0.$JOB_NAME"
```

如果 `torch.cuda.is_available()` 为 `False`，不要继续看 NCCL，先修 runtime CDI / vGPU 注入。

### 5.7 清理

```bash
kubectl delete job "$JOB_NAME" -n clearml
kubectl delete configmap "${JOB_NAME}-script" -n clearml
```

---

## 6. 方案 3 故障排查

| 现象 | 优先检查 |
|---|---|
| Job 创建失败 | `kubectl describe job "$JOB_NAME" -n clearml` |
| Pod 一直 Pending | Queue capability、节点 `volcano.sh/vgpu-*` allocatable |
| gang 不齐射 | `kubectl get podgroup -n clearml | grep "$JOB_NAME"` |
| `rank env missing` | Volcano `env` 插件是否生效，Pod 内是否有 `VK_TASK_INDEX` 或 `VC_TASK_INDEX` |
| `MASTER_ADDR` 解析失败 | Volcano `svc` 插件是否生效，`getent hosts "$JOB_NAME-worker-0.$JOB_NAME"` |
| CUDA 不可用 | `runtimeClassName: nvidia`、CDI spec、vGPU 注入事件 |
| `ImagePullBackOff` | 预拉 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 或用 `--image` 指定内网镜像 |
| NCCL timeout | `NCCL_SOCKET_IFNAME`、跨节点 Pod 网络、NetworkPolicy、防火墙 |

关键事件：

```bash
kubectl describe pod "$POD" -n clearml
kubectl get events -n clearml --sort-by=.lastTimestamp | tail -80
kubectl logs -n clearml "$POD" --tail=120
```

---

## 7. 方案 3 生产化开发计划

### M0：当前冒烟器收敛

目标：让平台侧能稳定证明 Volcano Job + vGPU + CDI + NCCL 可用。

已完成/应保持：

- 默认 vGPU 模式
- 默认 PyTorch CUDA 镜像
- 默认不依赖 ClearML Task
- CUDA 不可用时失败
- ClearML logging 不可用时降级为 warning
- Job 名、ConfigMap、资源、镜像可通过 CLI 控制

验收：

```bash
python submit/submit_volcano_job.py --num-nodes 2 --dry-run
python submit/submit_volcano_job.py --num-nodes 2 --apply
kubectl logs -n clearml -l volcano.sh/job-name=<JOB_NAME> --all-containers --prefix
```

### M1：平台提交器

目标：把 `submit/submit_volcano_job.py` 从 smoke 脚本升级为通用提交器。

范围：

- YAML values/config 文件输入，而不是只靠 CLI
- 支持 image、command、args、env、resources、nodeSelector、tolerations
- 支持 PVC、hostPath、Secret、ConfigMap、imagePullSecret
- 支持 dry-run 输出到文件
- 支持 cleanup / wait / logs / status 子命令
- 失败时打印 Job、PodGroup、Pod events 的高信号摘要

建议工作量：3-5 个工作日。

### M2：ClearML 弱集成增强

目标：保留 Volcano Job 的原生 gang，同时把实验元数据回写 ClearML。

范围：

- 用 Secret 注入 `CLEARML_API_ACCESS_KEY`、`CLEARML_API_SECRET_KEY`、`CLEARML_API_HOST`
- rank0 自动创建/更新 ClearML Task
- 写入 Job name、namespace、image、git commit、资源规格
- 上传日志摘要和失败事件为 artifact
- 支持 `--clearml-project`、`--clearml-task-name`、`--tags`

建议工作量：1-2 周。

### M3：训练框架适配

目标：把当前最小 DDP 模板扩展到真实业务训练。

范围：

- 接入真实 Dataset / PVC / 对象存储数据
- 支持 `torchrun` 风格入口
- 支持每 Pod 多 GPU：`--nproc-per-node`
- 区分 `NNODES`、`NODE_RANK`、`RANK`、`LOCAL_RANK`
- 统一 checkpoint 输出目录
- 支持数据集挂载和断点续训
- 支持训练失败后的诊断包收集

建议工作量：1-2 周，取决于训练代码规范程度。

### M4：生产运维

目标：让平台能长期承载多租户训练。

范围：

- 镜像预拉或内网镜像仓库
- Queue quota 与优先级策略
- Job TTL / 历史清理
- Prometheus 指标与告警
- 节点 CDI/vGPU 健康巡检
- 标准化故障手册

建议工作量：2-3 周。

---

## 8. 扩到 4 节点

Volcano Job 不需要手工复制 PodGroup。直接提高 `--num-nodes`，并确认 Queue capability 足够：

```bash
kubectl describe queue multinode-full-gpu | egrep -i 'volcano.sh/vgpu|cpu|memory'

python submit/submit_volcano_job.py \
  --num-nodes 4 \
  --apply \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

4 节点期望：

```text
allreduce sum = 10.0
rank=0/1/2/3 均 DONE ok=True
```

如果 Queue capability 只有 2 张卡，需要先扩大 `queue-multinode-vgpu.yaml` 中的：

```yaml
capability:
  volcano.sh/vgpu-number: "4"
  volcano.sh/vgpu-memory: "96"
```

然后重新 apply。

---

## 9. 推荐执行顺序

```bash
# 1. 平台确认
kubectl apply -f k8s/queue-multinode-vgpu.yaml
kubectl describe queue multinode-full-gpu

# 2. ClearML 原生多机冒烟
python train/multinode_launch_smoke.py --num-nodes 2 --queue multinode-full-gpu

# 3. Volcano Job gang 冒烟
python submit/submit_volcano_job.py --num-nodes 2 --dry-run
python submit/submit_volcano_job.py --num-nodes 2 --apply

# 4. 通过后再进入 M1/M2/M3 生产化
```

最终判断标准：

```text
[ ] 方案 1 能跑通 allreduce_sum=3.0
[ ] 方案 3 两个 Pod gang 齐射
[ ] 方案 3 Pod 内 torch.cuda.is_available=True
[ ] 方案 3 allreduce_sum=3.0
[ ] 无 unresolvable CDI devices
[ ] 无 PodGroup overused nvidia.com/gpu
```
