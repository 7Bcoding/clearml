# 整卡多机：方案 1 / 2 / 3 操作手册

**场景**：2+ 节点 × 每 Pod 1 张整卡（`nvidia.com/gpu`），暂不做 vGPU 切分。

**路径约定**（下文命令在 **ClearML SDK 仓库根目录** 执行，或把路径换成你的绝对路径）：

```bash
# Linux / macOS
export CLEARML_REPO=/path/to/clearml
export HELM_REPO=/path/to/clearml-helm-charts
cd "$CLEARML_REPO/examples/volcano_vgpu"

# Windows PowerShell
$env:CLEARML_REPO = "D:\Python\clearml"
$env:HELM_REPO = "D:\Python\clearml-helm-charts"
cd "$env:CLEARML_REPO\examples\volcano_vgpu"
```

**命名空间**：以下默认 `clearml`，若不同请全局替换。

---

## 文件清单

| 文件 | 方案 | 作用 |
|------|------|------|
| `train_launch_multinode_wholecard.py` | 1 | 提交 + Pod 内 `launch_multi_node` + NCCL 冒烟 |
| `submit_multinode_podgroup.py` | 2 | 本地 N Task 齐入队 |
| `train_multinode_podgroup.py` | 2 | Pod 内 NCCL 冒烟（三种 MASTER_ADDR 模式） |
| `submit_volcano_job_wholecard.py` | 3 | 创建 Task + ConfigMap + Volcano Job |
| `train_volcano_job_smoke.py` | 3 | Job Pod 内 NCCL 冒烟 |
| `k8s/volcano_queue_multinode_full_gpu.example.yaml` | 共用 | Volcano Queue CR |
| `k8s/values-multinode-full-gpu.example.yaml` | 1+2 | 整卡 glue Agent values |
| `k8s/values-multinode-full-gpu-hostnetwork.example.yaml` | 2-A | hostNetwork 叠加 |
| `k8s/podgroup_clearml_gang_full_2.example.yaml` | 2 | PodGroup CR |
| `k8s/service_multinode_master.example.yaml` | 2-C | Headless Service + Endpoints |
| `k8s/values-multinode-podgroup-service.example.yaml` | 2-C | 可选 env `MULTINODE_MASTER_ADDR` |
| `k8s/volcano_job_wholecard_gang.example.yaml` | 3 | Volcano Job 静态模板 |

---

# 第 0 章：环境检查（所有方案先做）

## 0.1 本机工具

```bash
kubectl version --client
kubectl cluster-info
helm version
python --version
pip show clearml
clearml-init   # 若未配置，按提示填 Server / Web / API / 凭证
```

**验证 ClearML 连通**：

```bash
python -c "from clearml.backend_api.session.client import APIClient; print('OK', APIClient().users.get_current_user())"
```

## 0.2 集群组件

```bash
kubectl get pods -n volcano-system
kubectl get crd | grep volcano
```

应能看到 Volcano 控制器 Running，且存在 `queues.scheduling.volcano.sh`、`jobs.batch.volcano.sh` 等 CRD。

## 0.3 GPU 节点与容量（至少 2 张整卡、最好在 2 台不同节点）

```bash
kubectl get nodes -o wide
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu
```

记录节点名，后文替换 `<GPU_NODE_A>`、`<GPU_NODE_B>`。

---

# 第 1 章：共用平台准备（方案 1 / 2 必须；方案 3 需要 Queue + 节点 label）

## 步骤 1.1 给 GPU 节点打 label

```bash
kubectl label node <GPU_NODE_A> gpu.present=true --overwrite
kubectl label node <GPU_NODE_B> gpu.present=true --overwrite
kubectl get nodes -l gpu.present=true
```

## 步骤 1.2 创建 Volcano Queue

**方法 A：kubectl（推荐）**

```bash
kubectl apply -f k8s/volcano_queue_multinode_full_gpu.example.yaml
kubectl get queue multinode-full-gpu
kubectl describe queue multinode-full-gpu
```

**方法 B：Volcano WebUI / vcctl**（若集群已装）

```bash
# 视安装情况可选
vcctl queue create --name multinode-full-gpu --weight 1
vcctl queue list
```

**验证**：`capability` 中 `nvidia.com/gpu` ≥ 2。

## 步骤 1.3 创建 ClearML Queue

**方法 A：WebUI**

1. 打开 ClearML WebUI → **Workers & Queues**
2. **+ NEW QUEUE** → 名称填 `multinode-full-gpu` → 保存

**方法 B：Python（可脚本化）**

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

## 步骤 1.4 部署整卡 k8s-glue Agent（方案 1 / 2 必须）

在 **`clearml-helm-charts`** 仓库执行（先按环境改 values 里的 `NCCL_SOCKET_IFNAME`、`nodeSelector`）：

```bash
cd "$HELM_REPO"

helm upgrade --install clearml-agent-multinode charts/clearml-agent -n clearml \
  -f examples/volcano-vgpu/custom_values.yaml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu.example.yaml"
```

> 与现有 vGPU Agent **并存**时，使用不同 release 名（上例 `clearml-agent-multinode`），并确保 `agentk8sglue.queue: multinode-full-gpu`、`vgpuHook.enabled: false`。

**验证 Agent**：

```bash
kubectl get pods -n clearml | grep agent
kubectl logs -n clearml deploy/clearml-agent-multinode --tail=80
```

日志中应出现监听队列 **`multinode-full-gpu`**；不应再 patch vgpu limits。

**验证 WebUI Worker**：

WebUI → Workers & Queues → 队列 `multinode-full-gpu` 下应出现新 Worker（可能延迟 1–2 分钟）。

## 步骤 1.5 确认 values 关键项（方案 1 / 2）

```bash
grep -E "group-name|CLEARML_MULTI_NODE|MY_POD_IP|vgpuHook|nvidia.com/gpu" \
  "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu.example.yaml"
```

应包含：

- `vgpuHook.enabled: false`
- `nvidia.com/gpu: "1"`
- `scheduling.k8s.io/group-name: clearml-gang-full-2`（方案 2 用）
- `CLEARML_MULTI_NODE_MASTER_DEF_ADDR`（方案 1 用）
- `MY_POD_IP`（方案 2 task-poll 用）

---

# 方案 1：`launch_multi_node`（ClearML 原生，无 gang）

**不需要** PodGroup。需要 **步骤 1.1–1.4** 全部完成。

## 步骤 1.1 提交任务

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python train_launch_multinode_wholecard.py --num-nodes 2 --queue multinode-full-gpu
```

记录终端输出的 Task id（master）。

## 步骤 1.2 观察 ClearML WebUI

1. WebUI → 项目 **volcano-vgpu**
2. 找到任务名含 `multinode-launch-wholecard` 的 **master Task**
3. 等待出现 **N−1 个子 Task**（`multi_node_instance` 标签）
4. master Task → **SCALARS** → `smoke/allreduce_sum` 应为 **3.0**（2 节点）
5. **CONFIGURATION** → `allreduce_ok` = 1

## 步骤 1.3 观察 K8s Pod（非 gang：先后 Running）

```bash
kubectl get pods -n clearml -o wide --sort-by=.metadata.creationTimestamp
kubectl logs -n clearml <master-pod-name> --tail=50
kubectl logs -n clearml <worker-pod-name> --tail=50
```

**预期**：master Pod 往往 **先** Running，worker Pod **后** 创建并 Running（不是齐射）。

## 步骤 1.4 验证 NCCL 环境变量（可选）

在 master Pod 日志中应看到类似：

```text
MASTER_ADDR=10.x.x.x  RANK=0  WORLD_SIZE=2
all_reduce sum=3.0 expected=3.0 ok=True
```

## 步骤 1.5 NCCL 跨节点失败时的修复

**方法 A：改网卡名**（编辑 values 后重新 helm upgrade）

```bash
# 查 K3s 默认网卡
kubectl exec -n clearml deploy/clearml-agent-multinode -- ip route

# 修改 k8s/values-multinode-full-gpu.example.yaml 中 NCCL_SOCKET_IFNAME 后：
cd "$HELM_REPO"
helm upgrade clearml-agent-multinode charts/clearml-agent -n clearml \
  -f examples/volcano-vgpu/custom_values.yaml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu.example.yaml"
```

**方法 B：叠加 hostNetwork**（与方案 2-A 相同 overlay）

```bash
helm upgrade clearml-agent-multinode charts/clearml-agent -n clearml \
  -f examples/volcano-vgpu/custom_values.yaml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu.example.yaml" \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu-hostnetwork.example.yaml"
```

然后 **重新提交** 步骤 1.1。

---

# 方案 2：PodGroup gang + N Task 齐入队

需要 **第 1 章全部** + 下面 PodGroup 步骤。  
**MASTER_ADDR** 三选一（建议先试 **B task-poll**）。

## 步骤 2.1 创建 PodGroup

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
kubectl apply -f k8s/podgroup_clearml_gang_full_2.example.yaml
kubectl get podgroup clearml-gang-full-2 -n clearml
kubectl describe podgroup clearml-gang-full-2 -n clearml
```

确认 `Spec.MinMember: 2`、`Spec.Queue: multinode-full-gpu`。

## 步骤 2.2 确认 Agent Pod 模板带 group-name 注解

```bash
# 任选一个近期 clearml task pod（或等提交后再查）
kubectl get pods -n clearml -o yaml | grep -A2 "group-name"
```

应看到 `scheduling.k8s.io/group-name: clearml-gang-full-2`。若无，重新 **步骤 1.4** helm upgrade。

---

## 步骤 2.B 提交：MASTER_ADDR 补法 B — task-poll（默认）

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python submit_multinode_podgroup.py --num-nodes 2 --queue multinode-full-gpu --master-addr-mode task-poll
```

记录输出的 `Master task (rank0): <task-id>` 与 `gang_id`。

**观察 gang**：

```bash
kubectl get podgroup clearml-gang-full-2 -n clearml -o wide
kubectl get pods -n clearml -o wide | grep podgroup-gang
kubectl describe pod <pod-name> -n clearml | grep -E "group-name|nvidia.com/gpu"
```

**预期**：2 Pod 先 **Pending**，资源满足后 **几乎同时 Running**；limits 为 `nvidia.com/gpu: 1`（无 vgpu 注解）。

**WebUI 验证**：

```bash
# 将 <master-task-id> 换成 submit 输出
# WebUI 打开该 Task → SCALARS → smoke/allreduce_sum = 3.0
```

**日志**：

```bash
kubectl logs -n clearml <rank0-pod> --tail=80
kubectl logs -n clearml <rank1-pod> --tail=80
```

---

## 步骤 2.A 提交：MASTER_ADDR 补法 A — fixed（hostNetwork + 节点 IP）

### 2.A.1 固定 rank0 节点

```bash
kubectl label node <GPU_NODE_A> multinode.master=true --overwrite
kubectl get nodes -l multinode.master=true -o wide
```

查节点 IP：

```bash
kubectl get node <GPU_NODE_A> -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}{"\n"}'
# 记下，例如 10.0.1.11
```

### 2.A.2 Helm 叠加 hostNetwork

```bash
cd "$HELM_REPO"
helm upgrade clearml-agent-multinode charts/clearml-agent -n clearml \
  -f examples/volcano-vgpu/custom_values.yaml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu.example.yaml" \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu-hostnetwork.example.yaml"
kubectl rollout status deploy/clearml-agent-multinode -n clearml
```

### 2.A.3 提交

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python submit_multinode_podgroup.py --num-nodes 2 --queue multinode-full-gpu \
  --master-addr-mode fixed --master-addr 10.0.1.11
```

验证命令同 **步骤 2.B**（gang + scalar 3.0）。

---

## 步骤 2.C 提交：MASTER_ADDR 补法 C — Service DNS

### 2.C.1 编辑 Service / Endpoints

编辑 `k8s/service_multinode_master.example.yaml`，将 `10.0.1.11` 改为 rank0 **节点 IP**（常与 2.A 相同）。

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
kubectl apply -f k8s/service_multinode_master.example.yaml
kubectl get svc,endpoints clearml-multinode-master -n clearml
```

### 2.C.2 （可选）Helm 注入 env

```bash
cd "$HELM_REPO"
helm upgrade clearml-agent-multinode charts/clearml-agent -n clearml \
  -f examples/volcano-vgpu/custom_values.yaml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-full-gpu.example.yaml" \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/values-multinode-podgroup-service.example.yaml"
```

### 2.C.3 提交

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python submit_multinode_podgroup.py --num-nodes 2 --queue multinode-full-gpu \
  --master-addr-mode service \
  --master-addr clearml-multinode-master.clearml.svc.cluster.local
```

**DNS 解析验证**（在任意训练 Pod 内）：

```bash
kubectl exec -n clearml <pod-name> -- getent hosts clearml-multinode-master.clearml.svc.cluster.local
```

验证命令同 **步骤 2.B**。

---

## 方案 2 故障排查命令

| 现象 | 命令 |
|------|------|
| PodGroup 状态 | `kubectl describe podgroup clearml-gang-full-2 -n clearml` |
| 队列资源 | `kubectl describe queue multinode-full-gpu` |
| Pod 调度事件 | `kubectl describe pod <pod> -n clearml` |
| NCCL 日志 | `kubectl logs <pod> -n clearml \| grep -i nccl` |
| 是否误走 vgpu | `kubectl get pod <pod> -n clearml -o jsonpath='{.spec.containers[0].resources}'` |

---

# 方案 3：Volcano Job（原生 gang + 原生 MASTER_ADDR）

| | 3a（本节，现仓库） | 3b（改 Agent，算法最省心） |
|---|-------------------|---------------------------|
| 提交 | `submit_volcano_job_wholecard.py` / kubectl | **`python train.py` + `execute_remotely` 一次** |
| 改 helm | ❌ | ✅（见 [`PLATFORM_scheme3b_volcano_job_agent_zh.md`](PLATFORM_scheme3b_volcano_job_agent_zh.md)） |
| 工作量 | 0（已可用） | 约 **2 人周（M1）** / **3–5 人周（M1+M2 生产化）** |

**不需要** ClearML glue Agent 监听队列来跑 **3a** Job，但需要：

- **步骤 1.1**（节点 label）
- **步骤 1.2**（Volcano Queue `multinode-full-gpu`）
- 本机 **kubectl** 可用
- （可选）**步骤 0.1** ClearML 连通，用于 rank0 写 scalar

**不需要** PodGroup CR、**不需要** 步骤 1.4 glue Agent（除非同时跑方案 1/2）。

---

## 步骤 3.A 辅助脚本提交（推荐）

### 3.A.1 预览生成的 YAML

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python submit_volcano_job_wholecard.py --num-nodes 2 --dry-run
```

### 3.A.2 一键 apply

```bash
python submit_volcano_job_wholecard.py --num-nodes 2 --apply
```

终端会打印 **ClearML Task id** 与 **Job 名**（如 `clearml-vjob-abc12345`），请记录。

### 3.A.3 观察 Job / Pod（gang 齐射）

```bash
export JOB_NAME=clearml-vjob-abc12345   # 换成你的

kubectl get job "$JOB_NAME" -n clearml
kubectl get pods -n clearml -l volcano.sh/job-name="$JOB_NAME" -o wide -w
```

**预期**：2 Pod 同时 Pending → 同时 Running。

### 3.A.4 看日志

```bash
kubectl logs -n clearml -l volcano.sh/job-name="$JOB_NAME" --all-containers --prefix --tail=80
# 或分别看 rank0 / rank1
kubectl get pods -n clearml -l volcano.sh/job-name="$JOB_NAME" -o name
kubectl logs -n clearml <job-name>-worker-0-xxx --tail=80
kubectl logs -n clearml <job-name>-worker-1-xxx --tail=80
```

日志应含：`rank=0 DONE ok=True sum=3.0`、`MASTER_ADDR=<job-name>-worker-0.<job-name>`（svc 插件 hostname.subdomain）。

### 3.A.5 WebUI 验证（若未加 --no-clearml-task）

打开步骤 3.A.2 打印的 Task id → **SCALARS** → `smoke/allreduce_sum` = **3.0**。

### 3.A.6 清理

```bash
kubectl delete job "$JOB_NAME" -n clearml
kubectl delete configmap "${JOB_NAME}-script" -n clearml
```

---

## 步骤 3.B 手工 kubectl 提交

### 3.B.1 创建 ClearML Task（可选，用于 rank0 日志）

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python -c "
from clearml import Task
t = Task.create(
    project_name='volcano-vgpu',
    task_name='volcano-job-smoke-manual',
    task_type=Task.TaskTypes.training,
    script='train_volcano_job_smoke.py',
    add_task_init_call=False,
)
t.set_tags(['scheme-3', 'smoke'])
print('CLEARML_TASK_ID=', t.id)
"
```

记下输出的 `CLEARML_TASK_ID`。

### 3.B.2 创建 ConfigMap

```bash
kubectl create configmap clearml-vjob-smoke-example-script -n clearml \
  --from-file=train_volcano_job_smoke.py="$CLEARML_REPO/examples/volcano_vgpu/train_volcano_job_smoke.py" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl get configmap clearml-vjob-smoke-example-script -n clearml
```

### 3.B.3 编辑并 apply Job YAML

```bash
# 将 k8s/volcano_job_wholecard_gang.example.yaml 中
#   value: "<clearml-task-id>"  替换为 3.B.1 的 id（不需要 WebUI 可留空 ""）

kubectl apply -f k8s/volcano_job_wholecard_gang.example.yaml
kubectl get job clearml-vjob-smoke-example -n clearml
kubectl get pods -n clearml -l volcano.sh/job-name=clearml-vjob-smoke-example -o wide
```

### 3.B.4 验证与清理

验证命令同 **步骤 3.A.3–3.A.5**（Job 名改为 `clearml-vjob-smoke-example`）。

```bash
kubectl delete job clearml-vjob-smoke-example -n clearml
kubectl delete configmap clearml-vjob-smoke-example-script -n clearml
```

---

## 方案 3 故障排查命令

| 现象 | 命令 |
|------|------|
| Job 未创建 | `kubectl describe job "$JOB_NAME" -n clearml` |
| 队列不存在 | `kubectl get queue multinode-full-gpu` |
| gang 未齐射 | `kubectl get podgroup -n clearml \| grep "$JOB_NAME"` |
| env 插件 | `kubectl exec <pod> -n clearml -- env \| grep VK_` |
| MASTER_ADDR | `kubectl exec <pod> -n clearml -- env \| grep MASTER` |

---

# 三方案对比与推荐顺序

| | 方案 1 | 方案 2 | 方案 3 |
|---|--------|--------|--------|
| gang | ❌ | ✅ | ✅ |
| 走 ClearML glue | ✅ | ✅ | ❌ |
| ClearML Task 数 | N | N | 0~1 |
| MASTER_ADDR | SDK 自动 | B / A / C 三选一 | svc DNS |
| 提交命令 | `python train_launch_multinode_wholecard.py ...` | `python submit_multinode_podgroup.py ...` | `python submit_volcano_job_wholecard.py --apply` |

**推荐冒烟顺序**：

```bash
# 1) NCCL + ClearML 多 Task
python train_launch_multinode_wholecard.py --num-nodes 2 --queue multinode-full-gpu

# 2) gang + 最简 MASTER_ADDR
python submit_volcano_job_wholecard.py --num-nodes 2 --apply

# 3) gang + glue 全 Task
python submit_multinode_podgroup.py --num-nodes 2 --master-addr-mode task-poll
```

---

# 附录：扩到 4 节点

1. 复制 `k8s/podgroup_clearml_gang_full_2.example.yaml` → `clearml-gang-full-4.yaml`，改 `minMember: 4`、`minResources.nvidia.com/gpu: "4"`
2. values 注解改为 `clearml-gang-full-4`
3. 提交时 `--num-nodes 4`
4. Volcano Job：`submit_volcano_job_wholecard.py --num-nodes 4 --apply`

```bash
kubectl apply -f k8s/podgroup_clearml_gang_full_4.yaml   # 需自行复制修改
python submit_multinode_podgroup.py --num-nodes 4 --podgroup clearml-gang-full-4
python submit_volcano_job_wholecard.py --num-nodes 4 --apply
```
