# 方案 3b：Agent 生成 Volcano Job（平台实现指南）

**目标**：算法工程师仍只写 **一个训练脚本 + 一次 `execute_remotely`**，由 k8s-glue Agent 在检测到多机任务时 **创建 Volcano Job**（gang + svc MASTER_ADDR），而不是 1 Task → 1 Pod。

**读者**：`clearml-helm-charts` / Agent 维护者  
**前提**：现网已有 vgpuHook（`bootstrap_k8s_vgpu_patch.py` 模式），Volcano 已安装。

---

## 1. 方案对比（为何要做 3b）

| | 方案 1 | 方案 2 | 方案 3a | **方案 3b** |
|---|--------|--------|---------|-------------|
| 算法提交 | `python train.py` | `submit_*.py` 每次 | `submit_volcano_job --apply` | **`python train.py`** |
| gang | ❌ | ✅ | ✅ | ✅ |
| MASTER_ADDR | SDK | 自补 | svc 插件 | **svc 插件** |
| 改 Agent | ❌ | ❌ | ❌ | **✅** |

---

## 2. 现网 glue 能力与缺口

### 2.1 当前数据流（单 Pod）

```text
ClearML Task 入队
  → k8s-glue 拉 Task（k8s_glue_example.py / K8sIntegration）
  → 读 template.yaml（ConfigMap clearml-agent-*-pt）
  → _kubectl_apply() 合并容器/环境/ClearML bootstrap
  → kubectl apply → 1 Pod
  → Pod 内 clearml-agent 装环境并执行 Task 脚本
```

相关代码（上游 `clearml-agent`）：

- `clearml_agent/glue/k8s.py` → `K8sIntegration._kubectl_apply`
- 仅支持 **`kind: pod`** 或 **`kind: job`（batch/v1 单 Pod Job）**
- `SUPPORTED_KIND = ("pod", "job")`，模板结构为 `spec.containers` 或 `spec.template.spec.containers`

**Volcano Job**（`batch.volcano.sh/v1alpha1`）结构完全不同：

```text
spec.minAvailable + spec.tasks[].replicas + spec.tasks[].template
spec.plugins: env / svc
```

现有 `_kubectl_apply` **无法**通过只改 `template.yaml` 变成 Volcano Job。

### 2.2 现网 vgpuHook 已证明的路径

`clearml-helm-charts` 已在 **不改上游 agent 仓库** 的前提下，用 **同进程 monkeypatch** 扩展 glue：

| 文件 | 作用 |
|------|------|
| `files/run_k8s_glue_with_vgpu_hook.py` | 启动 glue 前加载 bootstrap |
| `files/bootstrap_k8s_vgpu_patch.py` | patch `K8sIntegration._kubectl_apply` |
| `files/vgpu_template_module.py` | 读 Task `VGPU` 段，改 Pod YAML |
| `templates/agentk8sglue-vgpu-hook-configmap.yaml` | 挂载到 Agent Pod |
| `agentk8sglue-deployment.yaml` | `vgpuHook.enabled` 时走 `vgpu_glue_entrypoint.sh` |

**3b 建议复用同一模式**：新增 `volcanoJobHook`，在 `_kubectl_apply` 处 **整单替换为 Volcano Job YAML**（或短路原函数）。

### 2.3 RBAC 缺口

`templates/agentk8sglue-rbac.yaml` 当前仅有：

- `pods`, `secrets`, `services`, `events`
- 可选 `batch`/`extensions` **标准 Job**（模板含 `taskAsJob`，values 未暴露）

**3b 必须新增**：

```yaml
- apiGroups: ["batch.volcano.sh"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "create", "patch", "delete"]
# 可选
- apiGroups: ["scheduling.volcano.sh"]
  resources: ["podgroups"]
  verbs: ["get", "list", "watch", "create", "patch", "delete"]
```

---

## 3. 目标架构（3b 完成后）

```text
算法: python train.py  (connect Cluster + execute_remotely 一次)

ClearML Task 入队 (multinode-full-gpu)
  → glue 拉 Task
  → volcanoJobHook: 读 Cluster/nnodes >= 2
       → 构建 Volcano Job YAML（minAvailable=nnodes, replicas=nnodes）
       → plugins svc/env；MASTER_ADDR = <job-name>-worker-0.<job-name>（hostname.subdomain）
       → 每个 replica 容器仍走 clearml-agent bootstrap（与现 Pod 相同）
       → kubectl apply Volcano Job
  → Job watcher：Job/Pod 状态 → 回写 ClearML Task

算法脚本：dist.init_process_group / torchrun；勿 launch_multi_node / submit_*
```

---

## 4. 算法侧约定（SDK）

```python
task.connect(
    {
        "nnodes": 2,
        "nproc_per_node": 1,
    },
    name="Cluster",
)
task.execute_remotely(queue_name="multinode-full-gpu")
```

整卡：多机队列 `vgpuHook: false`，用 `nvidia.com/gpu`。  
vGPU：Job 每个 replica patch `volcano.sh/vgpu-*`（扩展 vgpu 模块）。

---

## 5. 实现任务拆分（helm 仓库）

### Epic A — 检测与 Job 构建（核心）

| # | 任务 | 文件（建议） | 说明 |
|---|------|--------------|------|
| A1 | 读 Task 超参 | `volcano_job_module.py` | `Cluster/nnodes`；`>=2` 触发 |
| A2 | Job 命名 | 同上 | `clearml-{task_id前缀}`，符合 DNS-1123 |
| A3 | 组装 Volcano Job | 同上 | `minAvailable`、`tasks[].replicas` |
| A4 | svc/env 插件 | 同上 | `MASTER_ADDR={job}-worker-0.{job}`（subdomain 必带） |
| A5 | 合并 basePodTemplate | 同上 | nodeSelector、NCCL env、volume |
| A6 | 注入 agent bootstrap | 同上 | 复用 glue 容器 command/args/env 生成逻辑 |
| A7 | patch 入口 | `bootstrap_k8s_volcano_job_patch.py` | `nnodes<2` 走原路径 |

**难点**：需在 patch 内 **整文档替换** 或 **短路** `_kubectl_apply`，不能仅改 Pod template。

### Epic B — 与 vgpuHook 共存

| # | 任务 | 说明 |
|---|------|------|
| B1 | patch 顺序 | volcano 分支 + vgpu 改 Job 内 resources |
| B2 | YAML 路径 | `spec.tasks[0].template.spec.containers[0].resources` |
| B3 | 分队列 | 推荐 multinode 整卡 Agent 不开 vgpuHook |

### Epic C — 生命周期

| # | 任务 | 说明 |
|---|------|------|
| C1 | Job watch | Complete/Failed 监听 |
| C2 | Task 状态 | Failed Job → `tasks.failed` |
| C3 | Stop | WebUI 停止 → delete Job |
| C4 | 并发限制 | `max_pods` 语义改为 Job×nnodes 或新参数 |

现 `resource_applied(pod_name)` 需扩展为 **job_name**。

### Epic D — Helm

| # | 文件 |
|---|------|
| D1 | `values.yaml` → `volcanoJobHook` |
| D2 | `agentk8sglue-volcano-job-hook-configmap.yaml` |
| D3 | `volcano_job_glue_entrypoint.sh` |
| D4 | RBAC volcano jobs |
| D5 | deployment 挂载与 entrypoint 分支 |

### Epic E — 测试

| # | 说明 |
|---|------|
| E1 | 算法模板 `train_multinode_volcano_job.py` |
| E2 | 2 节点 gang + scalar 3.0 |
| E3 | 单卡无回归 |

---

## 6. patch 示意（与 vgpu 同级）

```python
# bootstrap_k8s_volcano_job_patch.py（新建）
def _kubectl_apply_with_volcano_job(self, *args, **kwargs):
    task_id, task_data = ...  # 同 vgpu hook
    cluster = volcano_job_module.resolve_cluster_params(task_data)
    if not cluster or cluster["nnodes"] < 2:
        return K8sIntegration._kubectl_apply_original(self, *args, **kwargs)

    job_doc = volcano_job_module.build_job(
        task_id=task_id, task_data=task_data, cluster=cluster,
        base_template=self.template_dict, glue=self, ...)
    return volcano_job_module.kubectl_apply_job(job_doc, namespace=self.namespace)
```

参考现网：`clearml-helm-charts/charts/clearml-agent/files/bootstrap_k8s_vgpu_patch.py`。

---

## 7. 工作量估算（人天）

| 阶段 | 范围 | 估算 |
|------|------|------|
| **M1 冒烟** | 整卡 2 节点 Job + gang + 跑通 Task 脚本 | **8–12** |
| **M2 生产化** | watch、Stop、失败回写、并发 | **+10–15** |
| **M3 vGPU** | Job 内 vgpu limits | **+5–8** |
| **M4** | nproc_per_node>1、WebUI 改 nnodes | **+5–10** |
| **可选** | 上游 clearml-agent 正式 PR | **+15–25** |

**给算法可用（M1+M2）**：约 **3–5 人周**（1 人熟悉 glue）。  
**仅验证值不值得做（M1）**：约 **2 人周**。

---

## 8. 验收标准（M1）

```bash
# 算法仅一条命令，无 submit
python train_multinode_volcano_job.py --nnodes 2
```

1. `kubectl get job -n clearml` 有 `clearml-*`；2 Pod gang 齐射。  
2. `MASTER_ADDR=<job>-worker-0.<job>`（svc 插件 hostname.subdomain）。  
3. 1 个 ClearML Task；rank0 scalar 正确。  
4. 无 Cluster 的单卡任务仍走 Pod，无回归。

---

## 9. 风险与决策

| 风险 | 缓解 |
|------|------|
| agent 升级 break patch | pin 镜像 tag；长期 PR 上游 |
| N 副本各跑 full bootstrap | 后续 initContainer/缓存优化 |
| 算法误用 launch_multi_node | 文档 + 检测告警 |

**开工前确认**：

1. M1 是否只做整卡？（推荐）  
2. 是否独立 multinode Agent？（推荐）  
3. 每 replica 是否 full agent？（推荐是，算法零改）

---

## 10. 实施顺序

```text
1. RBAC + hook 空壳打日志
2. 硬编码 2 节点 Job → gang + NCCL
3. 接 glue bootstrap（真正跑 Task）
4. 读 Cluster 段
5. Job watch + Task 状态
6. vGPU（可选）
```

---

## 11. 参考（本仓库）

| 资产 | 用途 |
|------|------|
| `submit_volcano_job_wholecard.py` | Job YAML / svc MASTER_ADDR |
| `k8s/volcano_job_wholecard_gang.example.yaml` | 静态模板 |
| `PLATFORM_scheme3b_volcano_job_agent_zh.md` | 本文 |
| `MULTINODE_schemes_zh.md` | 方案 1/2/3a 操作手册 |
| `clearml-helm-charts/.../bootstrap_k8s_vgpu_patch.py` | patch 范本 |

3b 完成后：**方案 2 可废弃**；**方案 3a** 留作 Agent 未上线前的调试手段。
