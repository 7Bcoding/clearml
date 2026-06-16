# ClearML 算法工程师使用手册（Volcano vGPU 环境）

面向在「自建 ClearML Server + K8s Glue Agent + Volcano/HAMi vGPU」平台上提交训练任务的算法工程师。

**核心理念：你的代码 = 标准 ClearML 官方用法 + 两行平台相关代码。** 不需要任何自定义模块。

---

## 1. 平台契约：只有「两行」是平台相关的

把一段普通训练脚本变成「按任务申请 vGPU 的集群任务」，你只需要加两处平台相关代码，其余全是 ClearML 官方 API：

```python
# 平台相关 1/2 —— 声明 Pod 内依赖 (官方 API, 须在 Task.init 前)
Task.force_requirements_env_freeze(force=True, requirements_file="requirements-remote.txt")

# 平台相关 2/2 —— ★ 申请 per-task vGPU (这就是平台唯一约定) ★
task.connect({"vgpu_number": 1, "vgpu_memory": 2, "vgpu_cores": 30}, name="VGPU")
```

第 2 行是**唯一的平台特定知识**：Agent 的 vgpuHook 只读取名为 `VGPU` 的超参段里的三个键。`task.connect(dict, name="VGPU")` 是 100% 官方 API。

> 早期版本里的 `vgpu.py` / `setup_task` / `connect_vgpu` 等封装**不再需要**。`vgpu.py` 仅平台验证脚本仍在用，算法工程师可忽略。依赖、git、vGPU 注入等由平台在 Agent 侧一次性配置（见 `PLATFORM_scaling_zh.md`）。

---

## 2. 心智模型：脚本被运行两次

```
本地开发机                                集群任务 Pod (Agent 拉起)
─────────────                            ──────────────────────────
python train.py
  Task.init / connect(VGPU)  ── 上传 ──▶  Agent 出队 → 建 Pod
  execute_remotely()         ── 入队 ──▶  pip 装 requirements-remote.txt
  (本地到此 exit)                          再次运行 train.py
                                          这次 execute_remotely 不退出 → 继续训练
```

**一条铁律：** `task.execute_remotely(...)` 这行，本地提交时会在此退出；它**之后**的代码只在 Pod 内执行。

由此带来一个优雅结果：把 `import torch` 放在 `execute_remotely()` **之后**，本地提交根本不会执行到——**所以本机不需要装 torch，也不需要把 import 塞进函数里**。

---

## 3. 前置条件（一次性）

```bash
pip install clearml          # 只需 clearml; torch 由 Pod 内安装
clearml-init                 # 或手写 ~/clearml.conf
```

平台地址（当前环境）：

| 服务 | 地址 |
|------|------|
| API server | `http://10.10.36.6:30008` |
| Web server | `http://10.10.36.6:30080` |
| Files server | `http://10.10.36.6:30081` |
| 队列 | `volcano-queue` |

`~/clearml.conf` 示例：

```
api {
    api_server: http://10.10.36.6:30008
    web_server: http://10.10.36.6:30080
    files_server: http://10.10.36.6:30081
    credentials { access_key: "<你的key>"; secret_key: "<你的secret>" }
}
```

---

## 4. 超参定义：四种常见方式（按场景选，不是只有 argparse）

ClearML 会把超参写入任务的 **CONFIGURATION**，WebUI 里 Clone 后可直接改再 Enqueue。**训练脚本怎么定义超参，取决于你的项目习惯**——平台不强制 argparse。

| 方式 | 适用场景 | WebUI 位置 | 本目录示例 |
|------|----------|------------|------------|
| **argparse** | CLI 驱动、最常见 | CONFIGURATION → **Args** | 四个模板默认 |
| **`task.connect(dict)`** | Notebook、无 CLI、少量扁平超参 | CONFIGURATION → 自定义段名 (如 `Training`) | 模板内注释 [B] |
| **`task.connect_configuration(yaml)`** | 嵌套/大配置、多模块参数 | CONFIGURATION → **General** (YAML 编辑器) | `config.example.yaml` + 模板注释 [C] |
| **Hydra / OmegaConf** | 已有 Hydra 项目 | 取决于你怎么 `connect` | 见下文 |

### 4.1 argparse（默认推荐）

在 `Task.init(...)` **之后**、`execute_remotely()` **之前** 创建 parser 并 `parse_args()`。ClearML 默认 `auto_connect_arg_parser=True`，会自动把参数挂到 **Args** 段：

```python
task = Task.init(project_name="volcano-vgpu", task_name="my-training")
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()
# 训练里用 args.epochs, args.lr
```

命令行覆盖：`python train.py --epochs 20 --lr 3e-4`（本地与 Pod 内均生效）。

若已有 parser、不想自动连接，可 `Task.init(..., auto_connect_arg_parser=False)` 再手动 `task.connect_arg_parser(parser)`。

### 4.2 `task.connect(dict)` —— 扁平字典

适合参数不多、或从 Notebook 直接改 dict 的场景：

```python
hp = task.connect(
    {"epochs": 5, "batch_size": 128, "lr": 1e-3, "hidden": 256, "seed": 42},
    name="Training",   # WebUI 里显示为 CONFIGURATION → Training
)
# hp 是 live dict: WebUI 改值后 Pod 内读到的也是新值
epochs = hp["epochs"]
```

**注意：** 与 **VGPU** 段一样，`name` 决定 WebUI 分组名；业务超参请用 `Training` / `Hyperparameters` 等，**不要**和 `VGPU` 混在同一段。

### 4.3 `task.connect_configuration(yaml)` —— 嵌套 YAML

适合 model/data/training 分块的配置文件。同目录提供 `config.example.yaml`：

```python
cfg_path = os.path.join(os.path.dirname(__file__), "config.example.yaml")
cfg = task.connect_configuration(cfg_path, name="General")
epochs = cfg["training"]["epochs"]
batch_size = cfg["training"]["batch_size"]
```

WebUI 复跑：Clone → CONFIGURATION → **General** → 编辑 YAML → Enqueue。

### 4.4 Hydra / 其他配置框架

若项目已用 Hydra，**不必强行改成 argparse**。常见做法：

1. 在 `Task.init` 后把 Hydra 解析结果 `task.connect(OmegaConf.to_container(cfg))` 或写入 yaml 再 `connect_configuration`；
2. 或用 ClearML 的 `@hydra.main` 集成（见 [ClearML Hydra 文档](https://clear.ml/docs/latest/docs/integrations/hydra/)）。

原则：**只要最终超参进入 Task 的 CONFIGURATION，就能 WebUI 改参复跑**；与是否 argparse 无关。

### 4.5 DDP 里怎么传超参

多进程 worker 里不要再 `parse_args()`（子进程 argv 不可靠）。在 **`main()`** 里解析/连接一次，经 `vars(args)` 或 `hp` dict 传给 worker：

```python
mp.spawn(ddp_worker, args=(world_size, vars(args), task.id), nprocs=world_size, join=True)

def ddp_worker(rank, world_size, hp, task_id):
    epochs = int(hp["epochs"])   # hp 来自 main() 的 argparse 或 connect(dict)
    ...
```

仅 **rank 0** 调用 `Task.get_task(task_id)` 写日志/存模型；超参 dict 只读、各 rank 一致即可。

### 4.6 平台段 vs 业务超参

| 段名 | 内容 | 谁维护 |
|------|------|--------|
| **VGPU** | `vgpu_number`, `vgpu_memory`, `vgpu_cores` | 平台约定，Agent 读此段注入 Pod |
| **Args** / **Training** / **General** | 学习率、batch、模型结构等 | 算法工程师，按上面四种方式任选 |

---

## 5. ClearML 训练集成：算法工程师常用能力清单

以下均为官方 API，与是否 Volcano/vGPU **无关**；本目录模板已按需演示，可按表勾选加入你的脚本。

| 能力 | API / 用法 | WebUI 位置 | 模板中的体现 |
|------|------------|------------|--------------|
| **任务元数据** | `Task.init(project_name=..., task_name=...)` | 项目 / 实验名 | 全部 |
| **标签** | `task.set_tags(["single-gpu", "v1"])` | 实验列表筛选 | 全部 |
| **文本日志** | `logger.report_text("...")` | CONSOLE / DEBUG SAMPLES | 全部 |
| **训练曲线** | `logger.report_scalar("train", "loss", v, iteration=step)` | SCALARS | 全部 |
| **最终指标** | `logger.report_single_value("final_loss", v)` | 实验列表可排序 | 全部 |
| **产出模型** | `OutputModel(task=task, framework="pytorch").update_weights(path)` | MODELS → 可 Publish | `train_volcano_vgpu.py`, DDP |
| **预训练权重** | `InputModel(model_id=...).get_weights()` | 从 Model Registry 拉取 | `train_volcano_vgpu.py` 注释 |
| **任意文件** | `task.upload_artifact("checkpoint", artifact_object=path)` | ARTIFACTS | 全部 |
| **远程依赖** | `Task.force_requirements_env_freeze(...)` | CONFIGURATION → Installed packages | 全部（平台） |
| **自定义镜像** | `task.set_base_docker("registry/img:tag")` | EXECUTION → Container | `train_volcano_vgpu.py` 注释 |
| **远程执行** | `task.execute_remotely(queue_name="volcano-queue")` | 入队到 Agent | 全部（平台） |
| **复跑改参** | WebUI Clone → 改 CONFIGURATION → Enqueue | — | §11 |
| **资源监控** | Agent 自动上报 | SCALARS `:monitor:gpu` / `:monitor:machine` | 长跑 CNN 示例 |

### 5.1 指标怎么记

- **step 级 loss**：`report_scalar(..., iteration=global_step)` —— 看收敛细节。
- **epoch 级 loss/acc**：`iteration=epoch`，title 用 `epoch_loss` / `accuracy` —— 看整体趋势。
- **单次汇总**（便于对比实验）：`report_single_value("final_accuracy", 0.92)` —— 出现在实验列表列里。

### 5.2 模型与权重

```python
from clearml import OutputModel, InputModel

# 训练结束: 注册产出模型 (framework 按实际框架改)
output_model = OutputModel(task=task, framework="pytorch", name="my-model")
output_model.update_weights("model.pt")   # 可在 WebUI Publish 到 Model Registry

# 微调: 加载已有模型
input_model = InputModel(model_id="xxxxxxxx")  # 或 model_name="project/name"
weights_path = input_model.get_weights()
model.load_state_dict(torch.load(weights_path, map_location=device))
```

`OutputModel` 与 `upload_artifact` 可并存：前者面向 **Model Registry / 部署**；后者适合 **checkpoint、日志包、任意中间产物**。

### 5.3 标签与组织

```python
task.set_tags(["detection", "yolov8", "exp-042"])
```

便于在 WebUI 按标签过滤、对比同系列实验。建议包含：**任务类型**（detection/llm）、**阶段**（baseline/finetune）、**版本或日期**。

### 5.4 依赖与镜像二选一

| 策略 | 做法 | 优点 |
|------|------|------|
| **requirements 文件**（当前默认） | `force_requirements_env_freeze` + `requirements-remote.txt` | 改版本方便，与官方示例一致 |
| **预装镜像** | `task.set_base_docker("your/pytorch:2.5.1-cuda12.4")` | Pod 启动快，适合固定栈 |

两者可同时用：镜像提供基础环境，requirements 补充/锁定小版本。

### 5.5 不建议在业务代码里做的

| 事项 | 推荐归宿 |
|------|----------|
| git SSH→HTTPS | Agent 一次性 git 配置（运维） |
| vGPU 注入 Pod | Agent vgpuHook（运维）；你只写 `connect(VGPU)` |
| 封装 `setup_task` / `vgpu.py` | 训练脚本不需要；验证脚本可选 |

---

## 6. 标准提交模板（单卡，可直接复制）

完整可运行版见同目录 `train_template.py`（含超参三选一注释 + 标签 + scalar + artifact）。

```python
import argparse, os
from clearml import Task

_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)

task = Task.init(project_name="volcano-vgpu", task_name="my-training")
task.set_tags(["my-project"])

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

task.connect({"vgpu_number": 1, "vgpu_memory": 2, "vgpu_cores": 30}, name="VGPU")
task.execute_remotely(queue_name="volcano-queue")

import torch
logger = task.get_logger()
logger.report_scalar("train", "loss", 0.0, iteration=0)
logger.report_single_value("final_loss", 0.0)
task.upload_artifact("checkpoint", artifact_object="model.pt")
```

配套 `requirements-remote.txt`（与脚本同目录）：

```text
clearml==2.1.8
numpy==2.2.6
torch==2.5.1
torchvision==0.20.1
```

---

## 7. 多卡 DDP 模板

DDP 用 `torch.multiprocessing.spawn`，子进程会**重新 import 本模块**，所以必须把平台代码与启动逻辑放进 `main()` 并用 `if __name__ == "__main__"` 保护，`torch` 在 worker 函数内导入。完整版见 `train_ddp_volcano_vgpu.py`（MLP）和 `train_ddp_cnn_volcano_vgpu.py`（CNN 长跑）。骨架：

```python
import os
from clearml import Task

def ddp_worker(rank, world_size, hp, task_id):
    import torch, torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    if rank == 0:
        task = Task.get_task(task_id=task_id)
        logger = task.get_logger()
    # init_process_group / 绑定 cuda:rank / 包 DDP / 训练
    ...

def main():
    Task.force_requirements_env_freeze(force=True, requirements_file=".../requirements-remote.txt")
    task = Task.init(project_name="volcano-vgpu", task_name="train-ddp")
    task.set_tags(["ddp"])
    # 超参: argparse 或 task.connect(dict) —— 见 §4.5
    vgpu = task.connect({"vgpu_number": 2, "vgpu_memory": 2, "vgpu_cores": 30}, name="VGPU")
    task.execute_remotely(queue_name="volcano-queue")

    import torch, torch.multiprocessing as mp
    world_size = int(vgpu["vgpu_number"])
    mp.set_start_method("spawn", force=True)
    mp.spawn(ddp_worker, args=(world_size, vars(args), task.id), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

**DDP 注意点：**
- `--batch-size` 是**每卡** batch，全局 = batch × vgpu_number
- 每卡显存**独立**：2 卡 × 2GB ≠ 4GB 共享池，OOM 按单卡判定
- 只在 **rank 0** 上报 ClearML / 存模型 / `upload_artifact`
- 训练循环里「是否继续/分支」判断必须**全 rank 一致**（用 `dist.broadcast` 同步），否则集合通信会 hang

---

## 8. 多机与大模型训练（平台就绪后）

> 跨节点多机、FSDP、Megatron 等需要**平台侧**先支持多 Pod Job 与专用队列；就绪前请继续用 `train_template.py` / `train_ddp_*.py`（单 Pod）。平台改造见同目录 **`PLATFORM_scaling_zh.md`**（平台管理员文档）。

### 8.1 多机多卡：torchrun 入口（替代 mp.spawn）

平台支持后，业务脚本由 **torchrun** 在各节点拉起进程（不再用 `mp.spawn`）。ClearML 仍只在 **global rank 0** 写日志。

```python
#!/usr/bin/env python3
"""多机 DDP 入口（平台用 torchrun 启动）。"""
import argparse
import os

from clearml import OutputModel, Task

_REQS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-remote.txt")
Task.force_requirements_env_freeze(force=True, requirements_file=_REQS)

task = Task.init(project_name="volcano-vgpu", task_name="train-multinode-ddp")
task.set_tags(["multinode", "ddp"])

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64, help="每卡 batch")
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

task.connect({"vgpu_number": 2, "vgpu_memory": 24, "vgpu_cores": 100}, name="VGPU")
task.connect({"nnodes": 2, "nproc_per_node": 2}, name="Cluster")

task.execute_remotely(queue_name="volcano-queue")  # 队名由平台约定

# ========== 以下在训练 Pod 内，由 torchrun 为每个 rank 执行 ==========
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

logger = None
output_model = None
if rank == 0:
    tid = os.environ.get("CLEARML_TASK_ID") or Task.current_task().id
    task = Task.get_task(task_id=tid)
    logger = task.get_logger()
    output_model = OutputModel(task=task, framework="pytorch", name="multinode-ddp")

torch.manual_seed(42 + rank)
dataset = TensorDataset(torch.randn(4096, 784), torch.randint(0, 10, (4096,)))
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)

model = DDP(nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device),
            device_ids=[local_rank], output_device=local_rank)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    sampler.set_epoch(epoch)
    model.train()
    for bx, by in loader:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(bx), by)
        loss.backward()
        optim.step()
        if rank == 0 and logger:
            logger.report_scalar("train", "loss", float(loss.item()), iteration=epoch)

if rank == 0 and output_model:
    torch.save(model.module.state_dict(), "model.pt")
    output_model.update_weights("model.pt")
    logger.report_single_value("final_loss", float(loss.item()))

dist.destroy_process_group()
```

与单 Pod DDP 的差异：

| 项 | 单 Pod (`train_ddp_*.py`) | 多机 (`torchrun`) |
|----|---------------------------|-------------------|
| 进程启动 | `mp.spawn` | 平台 torchrun |
| `MASTER_ADDR` | `127.0.0.1` | 由平台注入（节点 0 Service DNS） |
| ClearML 写日志 | rank 0 | global rank 0 |
| `execute_remotely` | 本地提交一次 | 仍一次 |

### 8.2 大模型（FSDP / Megatron / DeepSpeed / MoE）

用成熟训练栈 + **rank0 薄集成** ClearML，不必为每种并行各写小模板。

- 超参：`task.connect_configuration("config.yaml", name="General")`
- 指标：预训练步数多，每 N step 打一次 `report_scalar`
- Checkpoint：大文件在 rank0 登记路径或上传 manifest，避免每个分片都进 Artifact

```python
task = Task.init(project_name="llm", task_name="pretrain-7b")
cfg = task.connect_configuration("config.yaml", name="General")
task.connect({"nnodes": 4, "nproc_per_node": 8}, name="Cluster")

if int(os.environ.get("RANK", "0")) == 0:
    logger = task.get_logger()

# megatron.training.pretrain(...) / deepspeed.launch 等由训练栈接管
```

---

## 9. vGPU 三个字段怎么填

平台 `gpu-memory-factor=1024`，所以 **`vgpu_memory` 单位是 GiB**。在 `task.connect({...}, name="VGPU")` 的字典里改：

| 字段 | 含义 | 取值 | 备注 |
|------|------|------|------|
| `vgpu_number` | 申请几张卡 | ≥1 | 多卡 = DDP world_size |
| `vgpu_memory` | 每卡显存(GiB) | 2 = 2GB | **不是 2048**；2GB 卡实际可用 ~1.5GiB（CUDA/框架开销 ~0.5GiB） |
| `vgpu_cores` | 每卡算力 | 1–100 | 百分比；30=30% |

> WebUI 复跑时，Clone 任务后在 CONFIGURATION → **VGPU** 段直接改这三个值再 Enqueue，无需动代码。

---

## 10. 支持的训练场景

| 场景 | 支持 | 怎么做 |
|------|------|--------|
| 单卡训练 | ✅ | `train_template.py` |
| 单 Pod 多卡 DDP | ✅ | `train_ddp_*.py`，`vgpu_number`=该机卡数 |
| 同一物理机多卡 | ✅ | 同上；Pod 落在单机，不要求 gang |
| 自带依赖镜像 | ✅ | `task.set_base_docker("你的GPU镜像")`，可省去 pip |
| 非 PyTorch 框架 | ✅ | 同样写法，import 你的框架，改 `requirements-remote.txt` |
| 长跑 / 采集监控 | ✅ | 任务需跑过监控周期(≥1–2min)，见 CNN 示例 `--target-minutes` |
| YAML / Hydra 配置 | ✅ | §4.3 / §4.4 |
| 多机多卡 / 大模型（FSDP 等） | 🔧 视平台 | 平台就绪后见 §8；管理员见 `PLATFORM_scaling_zh.md` |

---

## 11. 提交与复跑

```bash
python train_template.py                       # 单卡, 入队 volcano-queue
python train_ddp_volcano_vgpu.py               # 双卡 DDP
python train_ddp_cnn_volcano_vgpu.py           # 双卡 CNN 长跑 (监控曲线)
```

- **本地调试**：注释掉 `execute_remotely(...)` 那一行，即可在本机直接跑（需本机有 torch）。
- **WebUI 复跑**：实验 → **Clone** → 改 CONFIGURATION（**Args** / **Training** / **General** / **VGPU**）→ **Enqueue**，不改代码再跑一次。
- **对比实验**：同一脚本多次 Enqueue，或 Clone 后只改一个超参，用 `report_single_value` 在列表页排序对比。

---

## 12. 在 WebUI 看什么

| 位置 | 内容 |
|------|------|
| SCALARS | 你 `report_scalar` 的曲线（如 `train/loss`） |
| SCALARS（`:monitor:machine`） | CPU/内存/磁盘/网络（任务需 ≥1–2 分钟） |
| SCALARS（`:monitor:gpu`） | GPU 利用率/显存 |
| CONFIGURATION → VGPU / Args / Training / General | vGPU 配额与各方式定义的超参 |
| MODELS | `OutputModel.update_weights` 的模型，可 Publish |
| ARTIFACTS | `upload_artifact` 的文件 |
| DEBUG SAMPLES | `report_text`、matplotlib 图（若 `report_matplotlib_figure`） |
| CONSOLE | 标准输出与 Agent 日志 |

> 资源监控曲线没出来？多半任务太短（<30s 未到首个上报点），用 `train_ddp_cnn_volcano_vgpu.py --target-minutes 5` 跑久点即可。

---

## 13. 常见问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| 本地 `import torch` 报错 | torch 在 `execute_remotely` 之前导入了 | 把 import 移到 `execute_remotely()` 之后（DDP 放 worker 内） |
| Pod 内缺包 / 装错版本 | 依赖没声明对 | 用 `requirements-remote.txt` + `force_requirements_env_freeze` |
| Pod 内 git clone 失败 | Agent 无 SSH key | 平台侧给 Agent 配 git（运维），业务代码无需处理 |
| 2GB 卡 OOM 在 ~1.5GB | CUDA/框架固定开销 ~0.5GB | 正常；按 ~1.5GiB 规划 batch/模型 |
| 多卡 DDP 卡住(hang) | 各 rank 集合通信不同步 | 分支决策全 rank 一致 + `dist.broadcast` |
| GPU 监控曲线缺失 | 任务太短 / NVML 限制 | 跑久点；查 Console 是否有 `GPU monitoring ... switching off` |
| WebUI 改超参不生效 | 用了硬编码常量、未 connect | 超参必须来自 argparse / connect / yaml（§4） |
| Clone 后 VGPU 没变 | 改错段或未 Enqueue | 改 CONFIGURATION → **VGPU** 段后再 Enqueue |

---

## 14. 参考示例（本目录）

| 文件 | 用途 |
|------|------|
| `train_template.py` | **单卡训练模板（推荐起点）**：超参三选一注释 + 标签 + scalar + artifact |
| `train_volcano_vgpu.py` | 单卡完整示例：OutputModel + InputModel 注释 + set_base_docker 注释 |
| `train_ddp_volcano_vgpu.py` | 双卡 DDP（MLP，`--min-runtime-sec` 便于监控） |
| `train_ddp_cnn_volcano_vgpu.py` | 长跑 DDP（CNN，默认 ≥5 分钟，多指标 + artifact） |
| `config.example.yaml` | `connect_configuration` 示例配置 |
| `PLATFORM_scaling_zh.md` | **平台管理员**：多机多卡 / 大模型架构与改造（非算法工程师必读） |
| `requirements-remote.txt` | Pod 内依赖清单 |
| `smoke_test_vgpu.py` / `test_vgpu_per_task.py` | 平台验证脚本（仍用 `vgpu.py`，非算法工程师必读） |
| `vgpu.py` | 平台验证脚本的辅助库（训练模板**不需要**） |

---

## 15. 快速对照：我要加什么？

```
只做远程训练、尽快跑通     → train_template.py 复制，改 VGPU + 训练逻辑
要 Model Registry          → 加 OutputModel.update_weights (见 train_volcano_vgpu.py)
要加载已有权重微调         → InputModel (见 train_volcano_vgpu.py 注释)
大 YAML / 多模块配置       → connect_configuration + config.example.yaml
已有 Hydra 项目            → §4.4，不必改成 argparse
要 GPU 监控曲线            → 任务跑 ≥5min，或 train_ddp_cnn_volcano_vgpu.py
单机多卡 DDP               → train_ddp_*.py，vgpu_number = 该机卡数
多机 / 大模型              → §8（平台就绪后）；平台改造见 PLATFORM_scaling_zh.md
```
