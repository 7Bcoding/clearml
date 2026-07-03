# ClearML 训练最佳实践与冒烟测试流程

> 目标：只讨论训练部分，验证 `ClearML + Volcano + volcano-vgpu-device-plugin/HAMi-core` 能否承接 FiboCV 和 LLM 微调两类通用训练场景。  
> 调研基准时间：2026-07-02。  
> 示例脚本目录：`D:\Python\clearml\examples\training_smoke`

---

## 1. 总体最佳实践

### 1.1 平台分工

```text
ClearML：任务模板、参数、日志、指标、产物、复现
Volcano：队列、优先级、批任务调度、gang scheduling
volcano-vgpu-device-plugin/HAMi-core：vGPU 数量、显存、算力切分
NAS/MinIO/PVC：模型、数据、checkpoint、训练产物
Harbor：训练镜像
```

### 1.2 算法同学看到的界面语义

算法同学不应该理解 Kubernetes 资源名，只需要理解：

```text
训练模板
模型路径
数据路径
训练参数
输出路径
GPU 规格
```

在 ClearML WebUI 里，建议固定两个配置段：

| 配置段 | 用途 |
|---|---|
| `Args` / 业务配置段 | 模型、数据、训练参数、输出路径 |
| `VGPU` | 平台资源规格，供 Agent vGPU Hook 注入 Pod |

`VGPU` 段固定为：

```python
task.connect(
    {"vgpu_number": 1, "vgpu_memory": 24, "vgpu_cores": 100},
    name="VGPU",
)
```

当前 HAMi-core 路线下，Agent 应注入：

```yaml
volcano.sh/vgpu-number
volcano.sh/vgpu-memory
volcano.sh/vgpu-cores
```

不要在 vGPU-only 集群里使用：

```yaml
nvidia.com/gpu: 1
```

### 1.3 镜像最佳实践

训练镜像应该预装主要框架：

```text
FiboCV 镜像：fibo_cv_train + 依赖 + 内置训练入口
LLM 镜像：CUDA + PyTorch + ms-swift/LLaMA Factory + 常用依赖
```

ClearML 任务里只保留最小 wrapper 依赖：

```text
clearml
```

不要让每个训练任务临时安装大依赖，否则 Pod 启动慢、失败率高、环境不可控。

### 1.4 路径最佳实践

ClearML 远程训练里，参数中的模型和数据路径应理解为：

```text
最终训练 Pod 容器内能访问的路径
```

它不是提交脚本那台开发机的本地路径。第一次在开发机执行 smoke 脚本时，`task.execute_remotely()` 会把任务提交到 ClearML 队列，真正训练发生在 Kubernetes Pod 里。因此：

```text
正确：/data/models/Qwen2.5-0.5B-Instruct
正确：/data/datasets/toolace/data.json
正确：/app/fibo_cv_train/workspace/...
错误：C:\Users\xxx\Downloads\model
错误：/home/user/local-models，仅开发机存在但训练 Pod 不存在
```

推荐统一容器内路径：

| 类型 | 建议容器路径 |
|---|---|
| 数据集 | `/data/datasets/...` |
| 基座模型 | `/data/models/...` |
| 训练输出 | `/data/output/...` |
| FiboCV workspace | `/app/fibo_cv_train/workspace/...` |

这些路径背后可以来自：

```text
NAS 目录挂载
PVC 挂载
MinIO/S3 下载后的本地缓存
ClearML Dataset / Model 下载后的本地副本
```

冒烟测试阶段建议最简单：先把模型和数据下载到 NAS，再把 NAS 挂载到训练 Pod 的 `/data`。

---

## 2. 模型与数据下载入库

### 2.1 推荐原则

不要让每个训练任务都临时从公网下载大模型或大数据集。推荐把“下载/上传/入库”做成独立步骤：

```text
外部模型/数据源
-> 下载到 NAS/MinIO
-> 校验文件
-> 记录来源、版本、路径
-> 训练任务只引用稳定路径
```

这样训练任务更稳定，也避免重复消耗网络和磁盘。

### 2.2 模型从哪里下载

常见来源：

| 来源 | 适合内容 | 入库方式 |
|---|---|---|
| Hugging Face Hub | 开源 LLM、embedding、reranker、公开数据集 | 使用 `hf download` 或 `huggingface_hub.snapshot_download` |
| ModelScope | 国内访问更友好的模型和数据集 | 使用 `modelscope download` |
| 公司内部模型仓库 | 已评审或私有基座模型 | 同步到 NAS/MinIO |
| 算法同学本地已有模型 | 临时验证模型 | 上传到 NAS/MinIO 后再训练 |

示例，下载模型到 NAS：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /mnt/nas/models/Qwen2.5-0.5B-Instruct
```

或：

```bash
modelscope download \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir /mnt/nas/models/Qwen2.5-0.5B-Instruct
```

训练 Pod 中统一通过挂载看到：

```text
/data/models/Qwen2.5-0.5B-Instruct
```

### 2.3 数据集从哪里下载或上传

常见来源：

| 场景 | 来源 | 入库方式 |
|---|---|---|
| CV 检测/分割 | 标注平台、业务数据、公开 COCO/VOC 子集 | 转成 COCO/VOC 后上传 NAS |
| LLM SFT | 内部 JSON/JSONL、Hugging Face Dataset、ModelScope Dataset | 下载或上传到 NAS/MinIO |
| 冒烟测试 | FiboCV 内置 VOC、小样本 JSONL | 优先使用小数据 |

示例，LLM SFT 数据上传后建议路径：

```text
/data/datasets/toolace/data.json
/data/datasets/demo_sft/train.jsonl
```

示例，CV 数据上传后建议路径：

```text
/data/datasets/cv/person_hand_face/train/JPEGImages
/data/datasets/cv/person_hand_face/train/annotations.json
/data/datasets/cv/person_hand_face/val/JPEGImages
/data/datasets/cv/person_hand_face/val/annotations.json
```

### 2.4 是否使用 ClearML Dataset / Model

MVP 阶段建议：

```text
模型和大数据文件：放 NAS/MinIO
ClearML：记录路径、参数、manifest、评估结果
```

后续成熟后再逐步接：

```text
ClearML Dataset：管理中小型版本化数据集
ClearML Model：登记最终模型、ONNX、adapter 或模型 URI
```

大模型 checkpoint、完整 merged model 不建议直接上传到 ClearML Server 文件服务，优先放 NAS/MinIO，ClearML 记录 URI。

---

## 3. 模型和数据目录在哪里挂载

### 3.1 推荐结论

模型和数据目录挂载应该在平台侧完成，具体位置是：

```text
Kubernetes PVC/PV
-> ClearML Agent/K8s Glue 的 basePodTemplate
-> Agent 创建训练 Pod 时自动挂载
-> 训练脚本只使用容器内路径
```

不建议让算法同学在训练脚本里处理挂载，也不建议每个任务手写 Pod YAML。

### 3.2 是否给微调单独定义 Pod Template

建议按“队列/资源/存储需求”定义 Pod Template，而不是按每一个框架定义。

MVP 推荐两种方式：

| 方式 | 适用场景 | 建议 |
|---|---|---|
| 一个通用训练 Pod Template | CV 和 LLM 都挂同一个 `/data`，存储结构统一 | 最快落地 |
| LLM 微调单独一个 Pod Template | LLM 需要更大 `/dev/shm`、HF/ModelScope 缓存、大模型目录、特殊环境变量 | 推荐生产使用 |

也就是说，微调可以单独定义一个队列和 Pod Template，例如：

```text
llm-finetune-vgpu
```

但不是因为它叫 ms-swift 或 LLaMA Factory，而是因为 LLM 微调的资源和存储需求更重。

### 3.3 推荐队列划分

```text
cv-train-vgpu
llm-finetune-vgpu
llm-merge-cpu
llm-eval-vgpu
```

每个队列由一个 ClearML Agent/K8s Glue Deployment 监听，并绑定自己的 `basePodTemplate`。

如果使用开源 ClearML，工程上最稳的方式是：

```text
一个队列
一个 Agent Deployment
一个 basePodTemplate
```

这样比“单个 Agent 动态生成任意 Pod 模板”更容易控制和排障。

### 3.4 PVC 挂载示例

先由平台侧准备一个 PVC。冒烟阶段可以先使用 host local PV，不必立即上 NAS/NFS。

host local PV 示例文件：

```text
examples/training_smoke/k8s/local-pv-ai-data.example.yaml
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml
```

host local PV 的核心限制：

```text
数据只在一台 GPU 节点本地
训练 Pod 必须调度到这台节点
多节点训练不能把它当共享存储
节点故障会影响数据可用性
```

因此要在 local PV 和 Agent overlay 里填同一个节点名：

```text
<GPU_NODE_HOSTNAME>
```

如果未来改成 NAS/NFS，再使用共享 PVC，例如：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ai-data-pvc
  namespace: clearml
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Ti
  storageClassName: nfs-client
```

然后在 ClearML Agent values 里挂载到训练 Pod。

本仓库已给出可直接改造的 LLM 微调安装示例：

```text
examples/training_smoke/k8s/local-pv-ai-data.example.yaml
examples/training_smoke/k8s/pvc-ai-data.example.yaml
examples/training_smoke/k8s/volcano-queue-llm-finetune-vgpu.yaml
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml
```

LLM 微调队列示例：

```yaml
agentk8sglue:
  queue: llm-finetune-vgpu

  basePodTemplate:
    schedulerName: volcano
    runtimeClassName: nvidia

    annotations:
      volcano.sh/vgpu-mode: "hami-core"

    volumes:
      - name: ai-data
        persistentVolumeClaim:
          claimName: ai-data-pvc
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 128Gi

    volumeMounts:
      - name: ai-data
        mountPath: /data
      - name: dshm
        mountPath: /dev/shm

    env:
      - name: HF_HOME
        value: /data/cache/huggingface
      - name: TRANSFORMERS_CACHE
        value: /data/cache/huggingface
      - name: MODELSCOPE_CACHE
        value: /data/cache/modelscope
      - name: PYTORCH_CUDA_ALLOC_CONF
        value: expandable_segments:True
```

这样训练脚本里填写：

```text
--model /data/models/Qwen2.5-0.5B-Instruct
--dataset /data/datasets/demo_sft/train.jsonl
--output-dir /data/output/clearml-swift-sft-smoke
```

就能在训练 Pod 内访问到真实文件。

### 3.5 FiboCV 队列挂载示例

FiboCV 需要 `/data` 和 `/app/fibo_cv_train/workspace`。

可以使用同一个 PVC 的不同子目录：

```yaml
agentk8sglue:
  queue: cv-train-vgpu

  basePodTemplate:
    schedulerName: volcano
    runtimeClassName: nvidia

    annotations:
      volcano.sh/vgpu-mode: "hami-core"

    volumes:
      - name: ai-data
        persistentVolumeClaim:
          claimName: ai-data-pvc
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 16Gi

    volumeMounts:
      - name: ai-data
        mountPath: /data
        subPath: data
      - name: ai-data
        mountPath: /app/fibo_cv_train/workspace
        subPath: workspaces/fibocv
      - name: dshm
        mountPath: /dev/shm
```

这样 FiboCV YAML 里继续使用：

```text
train_img_path: /data/datasets/cv/person_hand_face/train/JPEGImages
train_ann_path: /data/datasets/cv/person_hand_face/train/annotations.json
```

训练产物写到：

```text
/app/fibo_cv_train/workspace
```

实际落在 PVC 的：

```text
workspaces/fibocv
```

### 3.6 不推荐的方式

不推荐：

```text
让算法同学自己写 -v 宿主机路径:容器路径
每个 ClearML Task 动态拼 Pod volume
训练脚本里写某台 GPU 机器的本地路径
使用 hostPath 作为长期方案
```

`hostPath` 只适合单机临时验证。多节点训练时，hostPath 要求每台 Worker 都有同一路径和同样内容，维护成本很高。

---

## 4. 冒烟测试前置检查

### 4.1 集群检查

在集群侧检查节点是否暴露 vGPU 资源：

```bash
kubectl get nodes -o json | jq -r '.items[] |
  "\(.metadata.name) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"]) vgpu-cores=\(.status.allocatable["volcano.sh/vgpu-cores"])"'
```

应该能看到 `volcano.sh/vgpu-*` 有值。

### 4.2 ClearML 队列检查

至少准备一个冒烟队列：

```text
volcano-queue
```

或按资源规格命名：

```text
dev-4090-vgpu-4g
dev-4090-vgpu-24g
train-a100-vgpu-40g
```

### 4.3 Agent 检查

ClearML Agent/K8s Glue 需要：

```text
监听目标队列
basePodTemplate.schedulerName = volcano
basePodTemplate.runtimeClassName = nvidia
vgpuHook.enabled = true
vgpuHook.section = VGPU
```

### 4.4 镜像检查

Worker 节点需要能拉取：

```text
fibo_cv_train:latest
ms-swift-cuda12.1:latest
```

如果镜像在 Harbor，需要使用完整地址，例如：

```text
harbor.company.local/ai/fibo_cv_train:latest
harbor.company.local/ai/ms-swift-cuda12.1:latest
```

---

## 5. FiboCV 训练最佳实践

### 5.1 推荐实现方式

FiboCV 是最适合先落地的模板，因为它天然符合：

```text
固定 Docker 镜像
固定 YAML 配置
固定 workspace 输出
训练完成自动导出 ONNX
```

推荐 ClearML 流程：

```text
算法同学 Clone FiboCV 模板任务
修改 Args 里的模型类型、类别、数据路径、epoch
修改 VGPU 里的资源规格
Enqueue 到 Volcano 队列
Agent 创建 Pod
Pod 内 wrapper 生成 YAML
调用 FiboCV 原始训练入口
ClearML 记录日志、ONNX、best.pth、loss 图、配置文件
```

### 5.2 冒烟测试

冒烟测试优先使用 FiboCV 镜像内置 VOC 示例，不依赖外部数据。

本地提交命令：

```bash
python examples/training_smoke/fibocv_clearml_smoke.py \
  --queue volcano-queue \
  --docker-image fibo_cv_train:latest \
  --clearml-project training-smoke/fibocv \
  --clearml-task-name fibocv-voc-smoke \
  --vgpu-number 1 \
  --vgpu-memory 4 \
  --vgpu-cores 30
```

如果你的镜像入口不是：

```text
python /app/fibo_cv_train/tools/train_dispatcher.py <config>
```

则通过 `--entry-cmd` 覆盖：

```bash
--entry-cmd "你的入口命令 {config_path}"
```

### 5.3 验收标准

```text
ClearML WebUI 中出现 training-smoke/fibocv 项目
任务进入 volcano-queue
Pod 被 Volcano 调度
Pod 资源使用 volcano.sh/vgpu-*
训练日志能在 ClearML CONSOLE 看到
ARTIFACTS 中有 work_dirs / onnx / auto_generated_configs
MODELS 中能看到 ONNX 模型，或日志明确说明未找到 ONNX
```

### 5.4 正式模板扩展

FiboCV 冒烟通过后，建议扩展为：

| 模板 | 说明 |
|---|---|
| `FiboCV Train` | 正式训练，自动导出 ONNX |
| `FiboCV Finetune` | 设置 `load_from` 做继续训练 |
| `FiboCV ONNX Eval` | 离线评测 ONNX |
| `FiboCV ONNX Infer` | 批量推理和可视化 |

---

## 6. LLM 微调训练最佳实践

### 6.1 推荐实现方式

LLM 微调不一定要“一个框架一个完全独立模板”。更好的做法是分两层：

```text
通用 LLM FineTune 模板：统一模型、数据、输出、资源、训练方法等公共参数
框架适配器：负责把公共参数翻译成 ms-swift / LLaMA Factory / 自研脚本的命令行
```

也就是说，用户看到的可以是一个“LLM 微调通用模板”，里面选择：

```text
backend = ms-swift / llama-factory / custom
```

平台内部再映射到不同命令。

MVP 阶段可以先做 `ms-swift` 适配器；后续新增 LLaMA Factory 时，不必推翻模板，只增加一个 backend adapter。

推荐 ClearML 流程：

```text
算法同学 Clone LLM SFT 模板任务
填写 base model、dataset、output_dir、LoRA 参数
填写 VGPU 资源规格
Enqueue 到 Volcano 队列
Agent 创建 Pod
Pod 内执行 swift sft
输出 adapter/checkpoint 到 NAS/PVC
ClearML 记录日志、训练参数、manifest、关键指标
```

### 6.2 通用模板参数建议

通用 LLM 微调模板建议包含：

| 参数 | 说明 |
|---|---|
| `backend` | `ms-swift` / `llama-factory` / `custom` |
| `image` | 训练镜像 |
| `model_path` | 容器内基座模型路径，或平台模型 URI |
| `dataset_path` | 容器内数据路径，或平台数据 URI |
| `output_dir` | 容器内输出路径 |
| `train_method` | `sft` / `dpo` / `rm` / `ppo` / `grpo` |
| `tuner_type` | `lora` / `qlora` / `full` |
| `epochs/max_steps` | 训练轮数或最大 step |
| `batch/grad_accum/lr/max_length` | 通用训练超参 |
| `lora_rank/lora_alpha/target_modules` | LoRA 参数 |
| `extra_args` | 高级参数，原样传给 backend |
| `VGPU` | `vgpu_number` / `vgpu_memory` / `vgpu_cores` |

通用模板内部伪逻辑：

```text
if backend == "ms-swift":
    render swift sft command
elif backend == "llama-factory":
    render llamafactory-cli train command
elif backend == "custom":
    run user provided command
```

最佳实践是：

```text
先做一个通用 LLM 微调模板
先实现 ms-swift adapter
保留 extra_args 兜底
等 LLaMA Factory 稳定后再加 llama-factory adapter
```

不要一开始做成完全自由命令模板，否则算法同学仍然要理解大量框架命令；也不要一开始把所有框架参数都做成表单，否则维护成本会很高。

### 6.3 冒烟测试

LLM 冒烟测试目标不是训练效果，而是确认：

```text
模型能加载
数据能读取
GPU 能分配
swift sft 能跑 1-2 step
输出目录能写入
ClearML 能记录日志和产物路径
```

本地提交命令：

```bash
python examples/training_smoke/llm_swift_sft_smoke.py \
  --queue volcano-queue \
  --docker-image ms-swift-cuda12.1:latest \
  --clearml-project training-smoke/llm \
  --clearml-task-name swift-sft-lora-smoke \
  --model /data/models/Qwen3.5-4B \
  --dataset /data/datasets/ToolACE/data.json \
  --output-dir /data/output/clearml-swift-sft-smoke \
  --max-steps 2 \
  --per-device-train-batch-size 1 \
  --max-length 512 \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

如果已有更小的基座模型，优先用小模型做 smoke，例如：

```text
/data/models/Qwen2.5-0.5B
```

这里的 `--model` 和 `--dataset` 都是训练 Pod 内路径。它们应该提前通过 NAS/PVC 挂载好，例如：

```text
宿主 NAS：/mnt/nas/models/Qwen2.5-0.5B-Instruct
Pod 内部：/data/models/Qwen2.5-0.5B-Instruct
```

### 6.4 验收标准

```text
ClearML WebUI 中出现 training-smoke/llm 项目
任务进入 volcano-queue
Pod 被 Volcano 调度
Pod 资源使用 volcano.sh/vgpu-*
Console 里能看到 swift sft 启动日志
至少完成 1-2 个 step
/data/output/clearml-swift-sft-smoke 下有输出
ARTIFACTS 中有 training-output-manifest
```

### 6.5 正式模板扩展

LLM 冒烟通过后，建议扩展为：

| 模板 | 说明 |
|---|---|
| `LLM FineTune - Universal` | 用户主入口，一个模板选择 backend |
| `ms-swift adapter` | `LLM FineTune` 内部的第一个适配器 |
| `LLaMA Factory adapter` | 第二个适配器，不一定暴露成独立入口 |
| `LLM Merge LoRA` | 合并 adapter |
| `LLM Eval` | 离线评测 |
| `LLM Quantization` | AWQ/GPTQ/GGUF 等量化 |

---

## 7. WebUI 复跑流程

冒烟测试第一次可以从命令行提交，之后算法同学应该在 ClearML WebUI 操作：

```text
打开成功的 smoke task
Clone
修改 Args 中的模型、数据、epoch、batch、output_dir
修改 VGPU 中的 vgpu_number、vgpu_memory、vgpu_cores
Enqueue 到目标队列
查看 Console、Scalars、Artifacts、Models
```

这就是后续 Portal 表单化之前的最低学习成本路径。

---

## 8. 常见问题

### 8.1 Pod Pending

优先检查：

```text
VGPU 段是否存在
队列是否正确
节点是否有 volcano.sh/vgpu-* allocatable
Pod 是否误用了 nvidia.com/gpu
Volcano Queue capability 是否足够
```

### 8.2 FiboCV 找不到配置或输出

优先检查：

```text
entry-cmd 是否匹配镜像真实入口
workspace 是否持久化挂载
容器内路径是否以 /data 或 /app/fibo_cv_train/workspace 开头
```

### 8.3 LLM 找不到模型或数据

优先检查：

```text
模型路径是否存在于训练 Pod 内
数据路径是否存在于训练 Pod 内
NAS/PVC 是否挂载到 /data
镜像内是否有 ms-swift
```

### 8.4 ClearML Artifact 上传太慢

大模型训练不要默认上传整个 checkpoint 目录到 ClearML Server。

推荐：

```text
checkpoint / adapter / merged model 保存在 NAS/MinIO
ClearML 上传 manifest、配置、评估报告
必要时只注册最终模型 URI
```

---

## 9. 最小落地顺序

```text
1. 确认 volcano-queue + vgpuHook 能跑 single_vgpu_minimal.py
2. 跑 FiboCV 内置 VOC 冒烟
3. 跑 LLM ms-swift 2 step 冒烟
4. 把成功任务 Clone 成团队模板
5. 让算法同学用 ClearML WebUI Clone/改参/Enqueue
6. 后续再封装 Portal 表单
```
