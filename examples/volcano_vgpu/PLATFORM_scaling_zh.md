# 平台扩容最佳实践：多机多卡与大模型训练

**读者**：平台 / MLOps 管理员（非算法工程师）  
**配套**：算法工程师手册见 [`USAGE_zh.md`](USAGE_zh.md)  
**集群现状**：K3s + ClearML Server + k8s-glue Agent + Volcano + HAMi vGPU；5×RTX 4090 + 2×A100  

本文说明：在现有架构上，要支持 **跨节点多机多卡** 以及 **FSDP / Megatron / DeepSpeed / MoE** 时，平台侧还需做哪些适配，并给出分阶段落地路径。

---

## 1. 结论摘要

| 问题 | 答案 |
|------|------|
| Volcano 已有 gang，能否直接多机多卡？ | **不能**。gang 解决 Pod **齐射**；还需 Agent **创建多 Pod Job** |
| 现网 `execute_remotely` 能否跨节点？ | **不能**。当前 **1 ClearML Task → 1 Pod → 1 节点** |
| ClearML 是否不适合大模型？ | **适合**。实验跟踪与训练并行正交；**global rank 0** 写 ClearML 即可 |
| 最小改造路径 | Phase 1：多机 DDP（Volcano Job + torchrun）→ Phase 2：FSDP/DeepSpeed → Phase 3：Megatron/MoE |

---

## 2. 现网架构与能力边界

### 2.1 分层示意

```
┌──────────────────────────────────────────────────────────────┐
│ ClearML Server (API / Web / Files)     ← 一般无需改           │
├──────────────────────────────────────────────────────────────┤
│ clearml-agent (k8s-glue Deployment)                          │
│   出队 → git/代码 → 生成 K8s 清单 → kubectl apply            │
│   【现状】单 Pod YAML + vgpuHook patch 单个 Pod               │
├──────────────────────────────────────────────────────────────┤
│ Volcano Scheduler + Queue + PodGroup (gang)    ← 已有 ✅        │
├──────────────────────────────────────────────────────────────┤
│ GPU 节点：HAMi vGPU (4090) / 整卡 (A100)                      │
│ CNI：K3s 默认 (flannel/calico 等)              ← NCCL 需实测 ⚠️ │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 能力矩阵

| 场景 | 现网 | 目标 |
|------|------|------|
| 单卡 / 单 Pod 多卡 DDP | ✅ | ✅ |
| 同一物理机多卡（Pod 落单机） | ✅ | ✅ |
| 跨节点一个 DDP/FSDP Job | ❌ | ✅ |
| Megatron TP/PP/EP、DeepSpeed MoE | ❌ | ✅ |
| 7 台各跑 7 个独立 ClearML Task | ✅ | ✅ |

### 2.3 为何 gang 不够

1. **K8s**：一个 Pod 只能调度到一个 Node；`vgpu_number=N` 表示单 Pod 要 N 张卡，不会跨机拆分。  
2. **k8s-glue**：默认 `kubectl apply` **一份** Pod；`vgpuHook` 只 patch **一个** manifest。  
3. **多机训练**：需要 `nnodes` 个 Pod（每节点 1 Pod 为常见模式）、统一 `MASTER_ADDR`、**同时 Ready**（gang）、每 Pod 独立 GPU 配额。

---

## 3. 目标架构（最佳实践）

### 3.1 统一原则

1. **ClearML Task 仍是一个实验**；多 Pod 是执行层细节，对算法工程师尽量透明。  
2. **资源与拓扑进 CONFIGURATION**：业务脚本 `task.connect(..., name="VGPU")` + `task.connect(..., name="Cluster")`，Agent **只读**这两段生成 Job。  
3. **同构 GPU**：一个 Volcano Job 内不要 4090 与 A100 混用；用 **队列 + nodeSelector** 隔离。  
4. **启动器统一 torchrun**（PyTorch 系）；Megatron/DeepSpeed 在 torchrun 或各自 launcher 之下，ClearML 挂在 **global rank 0**。  
5. **大模型用镜像固化依赖**，少依赖 Pod 内临时 `pip install`。

### 3.2 目标数据流

```
算法工程师本地
  Task.init → connect(VGPU) → connect(Cluster) → execute_remotely
        │
        ▼
k8s-glue Agent 出队
  解析 Cluster.nnodes / nproc_per_node / queue
  nnodes==1 → 现有单 Pod 路径（兼容现网）
  nnodes>1  → 生成 Volcano Job（minAvailable=nnodes）
              vgpuHook patch 每个 task template
              注入 CLEARML_TASK_ID, NODE_RANK, MASTER_ADDR
        │
        ▼
Volcano gang 调度 N 个 Pod 齐射
        │
        ▼
各 Pod: torchrun → 训练脚本 → rank0 写 ClearML
```

### 3.3 与算法侧的契约（平台发布给团队）

**VGPU 段**（每 Pod / 每节点 GPU 配额，与现网一致）：

```python
task.connect({
    "vgpu_number": 2,      # 本 Pod 使用的 GPU 数 = nproc_per_node
    "vgpu_memory": 24,     # GiB（factor=1024）；整卡 A100 填实际 GiB
    "vgpu_cores": 100,
}, name="VGPU")
```

**Cluster 段**（平台读，用于多机 Job）：

```python
task.connect({
    "nnodes": 2,
    "nproc_per_node": 2,
    "launcher": "torchrun",       # 默认 torchrun；megatron/deepspeed 可扩展
}, name="Cluster")
```

规则：

- `world_size = nnodes × nproc_per_node`
- `nnodes == 1`：走**现有单 Pod**逻辑（`mp.spawn` 或单 Pod 内 `torchrun` 均可）
- `nnodes > 1`：必须走 **Volcano Job + torchrun**；Agent 禁止只起一个 Pod

---

## 4. 分阶段实施路线图

| 阶段 | 目标 | 交付物 | 建议周期 |
|------|------|--------|----------|
| **P0** | 夯实单 Pod + 队列拆分 | 4090 vGPU 队列、A100 整卡队列、NCCL 单节点基准 | 1–2 周 |
| **P1** | 2 节点 DDP 冒烟 | Agent 多 Pod Job、vgpuHook 多副本、2 机 torchrun 通 | 2–4 周 |
| **P2** | FSDP / DeepSpeed ZeRO | 训练镜像、共享存储 checkpoint、rank0 ClearML | 2–3 周 |
| **P3** | Megatron / MoE | 专用镜像、长稳 Job、拓扑与队列规划 | 按模型规模 |

以下各节按 **P0→P3** 展开可执行项。

---

## 5. P0：基线与队列规划（先做）

### 5.1 队列拆分（强烈建议）

| 队列名 | 节点 | GPU 模式 | 用途 |
|--------|------|----------|------|
| `volcano-queue`（现有） | 4090 | HAMi vGPU 切分 | 日常实验、单 Pod DDP |
| `volcano-4090-full` | 4090 | 整卡 | 单机多卡中大模型试验 |
| `a100-full` | A100 | 整卡 | FSDP、多机 Job、微调 7B+ |
| `a100-multinode`（可选） | A100 多机 | 整卡 + gang | 仅多机大 Job，避免与小任务抢资源 |

Helm：每个队列可对应 **一个 agentk8sglue Deployment**（不同 `queue` + `basePodTemplate` / `nodeSelector`），或一个 Agent 监听多队列（简单但模板难区分）。

**nodeSelector 示例**（按你环境改 label）：

```yaml
nodeSelector:
  gpu.model: "4090"    # 或 nvidia.com/gpu.product 等
```

### 5.2 网络与 NCCL 基线（P0 必测）

在 **2 台 GPU 节点**上用手动 Volcano Job（见 `k8s/volcano_job_multinode.example.yaml`）跑通后再接 Agent。

```bash
# 每个训练 Pod 建议注入（值按实际网卡名改）
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=eth0,enp0s8
NCCL_IB_DISABLE=1          # 无 IB/RDMA 时先禁用
NCCL_P2P_DISABLE=0         # 同机 P2P 保持
```

**验收**：2 节点 × 1 卡，`torchrun` all_reduce 延迟可接受；无 `NCCL WARN` 级超时。

### 5.3 存储

| 用途 | 建议 |
|------|------|
| 代码 | git（Agent 已处理 SSH→HTTPS 一次性配置） |
| 数据集 | NFS / CSI / object storage 挂载到各 Pod **相同路径** |
| Checkpoint | 多机必须 **共享读写**（NFS、S3、cephfs）；rank0 写、各 rank 读 |
| ClearML Artifacts | 大模型 checkpoint **不要**默认 `upload_artifact` 全量；rank0 登记路径或 manifest |

### 5.4 git / 依赖 / vGPU（维持现网）

| 事项 | 做法 |
|------|------|
| git SSH→HTTPS | Agent 镜像或 init 一次性 `git config` |
| 依赖 | 小任务 `requirements-remote.txt`；大任务 **预装镜像** |
| vGPU 注入 | 保持 `vgpuHook`；扩展见 §6.3 |

---

## 6. P1：多机多卡（核心平台改造）

### 6.1 Volcano Job 模板

参考 [`k8s/volcano_job_multinode.example.yaml`](k8s/volcano_job_multinode.example.yaml)。关键点：

```yaml
spec:
  minAvailable: <nnodes>       # gang：全部 worker 就绪才启动
  schedulerName: volcano
  queue: <queue>
  plugins:
    svc: []                     # 为 task-0 提供 headless DNS → MASTER_ADDR
  tasks:
    - replicas: <nnodes>
      name: worker
      template:
        spec:
          schedulerName: volcano
          containers:
            - command: ["/bin/bash", "-lc"]
              args:
                - |
                  exec torchrun \
                    --nnodes=${NNODES} \
                    --nproc_per_node=${NPROC_PER_NODE} \
                    --node_rank=${NODE_RANK} \
                    --rdzv_backend=c10d \
                    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                    <agent 生成的训练入口命令>
```

**NODE_RANK**：由 Agent 按 Pod 序号写入（`metadata.name` 或 Volcano `VK_TASK_INDEX` 等），**不要**所有 Pod 写死 0。

**MASTER_ADDR**：使用 Volcano `svc` 插件生成的 DNS（`<job-name>-worker-0` 类名），与 `torchrun --rdzv_endpoint` 一致。

### 6.2 k8s-glue Agent 改造要点

在 `clearml-helm-charts`（或自建 glue 镜像）中增加 **multinode 分支**（伪代码逻辑）：

```
on_task_dequeued(task):
    cluster = task.get_parameters_section("Cluster") or {}
    nnodes = int(cluster.get("nnodes", 1))

    if nnodes <= 1:
        manifest = build_single_pod(task)          # 现网路径
    else:
        manifest = build_volcano_job(task, nnodes) # 新路径

    manifest = vgpu_hook.apply(manifest, task)     # 见 6.3
    kubectl_apply(manifest)
    wait_for_job_completion()                        # 或 watch PodGroup
```

**build_volcano_job** 需包含：

- 从 Task 取镜像、`container_args`（原训练脚本命令）
- 写入 `CLEARML_TASK_ID`、`CLEARML_API_HOST` 等（与单 Pod 一致）
- `replicas: nnodes`，`minAvailable: nnodes`
- 队列与 `nodeSelector` 来自队列配置或 Cluster 段

**注意**：现网 `bootstrap_k8s_vgpu_patch.py` 若 monkeypatch `_kubectl_apply`，需确认对 **Volcano Job** 的 apply 同样生效，且 patch 的是 **每个 task 的 pod template**。

### 6.3 vgpuHook 多 Pod 扩展

现网 `vgpu_template_module.py` 从 Task CONFIGURATION 读 `VGPU` 段，写入 **单个** Pod 的 `resources.limits`。

多机时需：

1. 解析 Job manifest 中 **每个** `tasks[].template.spec.containers[]`  
2. 对每个 container 写入相同的 `volcano.sh/vgpu-number/memory/cores`（或支持按节点差异化，一般先统一）  
3. `vgpu-number` 应等于 **Cluster.nproc_per_node**（每 Pod 卡数），不是 `nnodes × nproc_per_node` 的总和打在一个 Pod 上  

**整卡 A100 队列**：可配置 `vgpuHook.enabled: false`，改用 `nvidia.com/gpu: "8"` 等传统资源名。

### 6.4 ClearML 集成（平台注入）

每个训练容器环境变量：

```bash
CLEARML_TASK_ID=<task.id>
CLEARML_API_HOST=...
CLEARML_API_ACCESS_KEY=...   # 或由 Agent 挂载 conf
CLEARML_API_SECRET_KEY=...
```

训练脚本内 rank 0：

```python
if int(os.environ["RANK"]) == 0:
    task = Task.get_task(task_id=os.environ["CLEARML_TASK_ID"])
    logger = task.get_logger()
```

**不要**在每个 local rank 都 `Task.init()`（会产生多个实验或竞争写）。

### 6.5 P1 验收清单

- [ ] 2×4090 节点，2 Pod，`nnodes=2, nproc_per_node=1`，DDP 训练 loss 下降  
- [ ] 故意让节点 1 晚启动 → gang 下不会单节点先 hang 超时（或明确失败可观测）  
- [ ] WebUI 可见 rank0 scalar；CONFIGURATION 含 VGPU + Cluster  
- [ ] Job 结束 Pod 清理；失败时 ClearML Task 标为 failed  
- [ ] 与单 Pod 任务并行跑，队列资源不互相饿死（Volcano queue capability）

---

## 7. P2：FSDP 与 DeepSpeed

### 7.1 与多机 Job 的关系

**执行层与 P1 相同**（Volcano Job + torchrun）；差别在 **容器入口与镜像**：

| 框架 | 启动方式 | 镜像 |
|------|----------|------|
| PyTorch FSDP | `torchrun` + 脚本内 `FullyShardedDataParallel` | `pytorch:2.x-cuda12.x` 或自建 |
| DeepSpeed ZeRO | `deepspeed --num_nodes ... launcher.py` | `deepspeed/deepspeed` 或自建 |

Agent 可在 `Cluster.launcher` 分支：

- `torchrun` → §6 命令  
- `deepspeed` → `deepspeed --hostfile ...` 或 ZeRO-3 多节点 [官方 launch 文档](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)

### 7.2 平台额外要求

| 项 | 说明 |
|----|------|
| 显存 | FSDP/ZeRO 为 **省显存**；A100 整卡队列为主 |
| 共享存储 | ZeRO-3 / FSDP checkpoint 聚合在 rank0 或 sharded 目录，路径必须共享 |
| 镜像 | 固定 `torch==x`, `deepspeed==y`, `flash-attn` 等；避免任务内 pip |
| NCCL | 多机 all-gather 流量大于 DDP；优先同机架、万兆以上 |
| ClearML | rank0 每 N step `report_scalar`；`report_single_value` 记最终 loss |

### 7.3 DeepSpeed 与 ClearML

可选：DeepSpeed 配置里不集成 ClearML，由训练脚本 rank0 手动上报（最简单、可控）。  
若用 HF Trainer + DeepSpeed，可评估 `report_to` 集成（与你们栈是否一致）。

### 7.4 P2 验收

- [ ] 单机 A100 FSDP 7B 级模型（或更小）OOM 前能稳跑  
- [ ] 2 机 DeepSpeed ZeRO-2 与单机等效 batch 下 loss 趋势一致  
- [ ] Checkpoint 写入共享目录，rank0 在 ClearML 登记路径

---

## 8. P3：Megatron-LM / 张量并行 / 流水线 / MoE

### 8.1 定位

Megatron、DeepSpeed-MoE 等是 **独立训练系统**；平台职责是：

1. 提供 **稳定多机 Job**（P1）  
2. 提供 **专用镜像 + 队列 + 共享存储**  
3. 保证 **环境变量与网络** 满足其 launch 脚本假设  

**不要**在 clearml-agent 里硬编码 Megatron 参数；由算法镜像内的 `pretrain_gpt.py` + 配置文件驱动。

### 8.2 拓扑与资源

示例：8 机 × 8 卡 = 64 GPU，TP=4, PP=4, DP=4（示意，按模型定）：

- `nnodes=8`, `nproc_per_node=8`  
- 队列 `a100-multinode` 仅允许 A100 节点  
- Volcano `minAvailable=8`  
- 镜像：`megatron-deepspeed:tag` 预装 NCCL、Apex、Transformer Engine 等  

MoE **专家并行** 通常在同一 Megatron/DeepSpeed 启动器内配置 `expert_model_parallel_size`，平台仍是一套 Job 机制。

### 8.3 ClearML 薄集成

在 Megatron **driver**（global rank 0）：

```python
task = Task.init(...)  # 仅 rank 0 执行
cfg = task.connect_configuration("/workspace/megatron_config.yaml", name="General")
task.connect({"nnodes": 8, "nproc_per_node": 8, "launcher": "megatron"}, name="Cluster")
logger = task.get_logger()
# 调用 megatron.training.pretrain
```

指标：hook Megatron 的 iteration callback，或定时读 log 解析；避免每 micro-batch 上报。

Checkpoint：Megatron 自带分布式 checkpoint；ClearML 只登记 **最终** 或 **按 epoch** 的 manifest。

### 8.4 MoE 注意点

- 显存与通信模式比稠密模型更复杂；**homogeneous GPU** 更重要  
- 节点数需与 expert parallel 拓扑整除关系匹配（由 Megatron/DS 配置决定，平台只保证资源数）  
- 建议单独 **长任务队列**，`MAX_PODS` / queue weight 防止占满集群  

---

## 9. Helm / Agent 配置清单（汇总）

| 配置项 | 单 Pod（现网） | 多机（目标） |
|--------|----------------|--------------|
| `agentk8sglue.queue` | `volcano-queue` 等 | 多队列多 Deployment |
| `basePodTemplate.schedulerName` | `volcano` | 同左 |
| `vgpuHook.enabled` | `true`（4090） | `true` + 多 template patch |
| `vgpuHook.mode` | 按 chart 文档 | 确保 patch Job CRD |
| `defaultContainerImage` | cuda runtime | 大模型专用镜像 tag |
| `maxPods` | 按集群容量 | 多机 Job 算 **nnodes** 而非 1 |
| git / `CLEARML_AGENT_NO_UPDATE` | 按现网实践 | 同左 |

---

## 10. 风险与决策

| 风险 | 缓解 |
|------|------|
| NCCL 跨节点慢 / 不稳定 | P0 实测；调 `NCCL_*`；考虑升级网络或 IB |
| 4090 vGPU 2GB 误导大模型任务 | 队列隔离 + WebUI 文档；大模型仅 `a100-full` |
| 异构 4090+A100 同 Job | nodeSelector + 队列禁止混部 |
| vgpuHook 只 patch 单 Pod | P1 必测 Job manifest |
| ClearML 任务状态与 Volcano Job 不同步 | Agent watch Job，失败回写 Task |
| 单机 glue `maxPods` 占满 | 多机 Job 一次占 nnodes 个 Pod 额度 |

---

## 11. 推荐实施顺序（Checklist）

**本周可开始**

- [ ] 划分节点 label、`volcano-queue` / `a100-full`  
- [ ] 手动 apply `k8s/volcano_job_multinode.example.yaml`（改 nnodes=2）做 NCCL 冒烟  
- [ ] 记录 `MASTER_ADDR` DNS 与 `NODE_RANK` 注入方式  

**P1 开发**

- [ ] Agent：`Cluster.nnodes > 1` → Volcano Job  
- [ ] vgpuHook：patch Job 内每个 Pod template  
- [ ] 注入 `CLEARML_TASK_ID` + torchrun 入口  
- [ ] 文档同步算法侧 §8（`USAGE_zh.md`）  

**P2**

- [ ] A100 训练镜像（torch + deepspeed + flash-attn）  
- [ ] 共享 checkpoint PVC  
- [ ] FSDP / DeepSpeed 验收任务进 ClearML 示例库  

**P3**

- [ ] Megatron/MoE 镜像与队列  
- [ ] 与算法团队约定 config 仓库与 ClearML Project 划分  

---

## 12. 参考文件

| 文件 | 说明 |
|------|------|
| [`k8s/volcano_job_multinode.example.yaml`](k8s/volcano_job_multinode.example.yaml) | Volcano gang 多 Pod Job 样例 |
| [`USAGE_zh.md`](USAGE_zh.md) | 算法工程师手册（§8 为多机就绪后的写法） |
| `clearml-helm-charts` | `vgpuHook`、`bootstrap_k8s_vgpu_patch.py` 等 |
| [Volcano Job Plugins (svc)](https://volcano.sh/en/docs/v1-7-0/plugins/) | MASTER_ADDR DNS |
| [PyTorch torchrun](https://pytorch.org/docs/stable/elastic/run.html) | 多机启动参数 |

---

## 13. 附录：并行方式与平台责任对照

| 训练并行 | 平台要保证什么 | ClearML 谁写 |
|----------|----------------|--------------|
| 单卡 | 单 Pod + vGPU | 单进程 |
| 单 Pod DDP | 单 Pod 多 GPU | rank 0 |
| 多机 DDP | Volcano Job gang + NCCL | global rank 0 |
| FSDP / ZeRO | 同上 + 大显存 + 共享存储 | global rank 0 |
| Megatron TP/PP | 同上 + 专用镜像 | driver rank 0 |
| MoE EP | 同上 + 拓扑匹配节点数 | driver rank 0 |

ClearML Server **无需为每种并行单独升级**；扩展 Agent 与 Volcano Job 即可支撑算法侧持续用 `execute_remotely` + rank0 上报。
