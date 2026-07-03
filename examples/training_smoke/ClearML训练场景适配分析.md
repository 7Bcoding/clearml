# ClearML 训练场景适配分析

> 基于算法同学提供的两个训练场景文档整理：  
> 1. `FiboCV训练操作说明_20260206.docx`  
> 2. `大语言模型微调训练流程_20260701.docx`  
> 调研基准时间：2026-07-02。

---

## 1. 总体结论

这两个场景都适合纳入 `ClearML + Volcano + volcano-vgpu-device-plugin/HAMi-core` 训练体系，但适配方式不同。

| 场景 | 当前使用方式 | ClearML 适配难度 | 推荐形态 |
|---|---|---:|---|
| FiboCV 训练 | Docker 镜像 + YAML 配置 + 数据挂载 + workspace 输出 | 低 | ClearML 训练模板 |
| LLM 微调 | Docker 容器 + Python 环境 + ms-swift/LLaMA Factory 命令 + merge + vLLM 验证 | 中 | 通用 LLM FineTune 模板 + backend adapter |

核心判断：

```text
FiboCV：非常适合先做成 ClearML MVP 模板。
LLM 微调：适合拆成训练、合并、评测几个 ClearML 任务阶段。
vLLM 长期服务：不建议用 ClearML 常驻运行，应该交给 KServe。
交互式调试：不建议塞进 ClearML，应该交给工作空间或自定义任务。
```

### 1.1 单独使用 ClearML + Volcano + volcano-vgpu-device-plugin 的能力边界

该组合本质上适合承接“离线批任务训练”，包括：

```text
CV 训练 / finetune / ONNX 导出 / 离线评测
LLM SFT / LoRA / QLoRA / merge / export / 离线评测
量化 / 蒸馏 / 剪枝 / Benchmark
```

它不适合直接承接：

```text
长期在线推理服务
需要人工频繁进容器交互开发的场景
任意用户自定义长期服务
完整 Kubernetes 资源管理平台
```

推荐训练链路：

```text
算法同学在 ClearML 克隆模板任务
修改超参、模型路径、数据路径、资源规格
enqueue 到指定 ClearML Queue
ClearML Agent/K8s Glue 创建 Pod
Volcano 调度到 GPU 节点
volcano-vgpu-device-plugin/HAMi-core 分配 vGPU 或等价整卡资源
训练脚本写入 NAS/MinIO/PVC
ClearML 记录日志、指标、参数、产物
```

在当前 HAMi-core 路线里，GPU 资源语义应使用：

```yaml
metadata:
  annotations:
    volcano.sh/vgpu-mode: "hami-core"
spec:
  schedulerName: volcano
  runtimeClassName: nvidia
  containers:
  - name: trainer
    resources:
      limits:
        volcano.sh/vgpu-number: "1"
        volcano.sh/vgpu-memory: "24"
        volcano.sh/vgpu-cores: "100"
```

不要在 vGPU-only 集群里写：

```yaml
nvidia.com/gpu: 1
```

`volcano.sh/vgpu-memory` 的单位要以插件实际配置为准。你们当前文档按 `gpu-memory-factor=1024` 配置时，ClearML 参数里的 `vgpu_memory=24` 表示 24 GiB；官方示例里也可能看到按 MB 填写的方式，所以平台侧必须统一封装，避免让算法同学直接理解底层单位。

---

## 2. FiboCV 训练场景

### 2.1 当前算法同学的使用流程

当前流程可以概括为：

```text
准备 COCO/VOC 数据集
复制并修改 YAML 配置
docker run --gpus all
挂载数据目录到 /data
挂载 workspace 到 /app/fibo_cv_train/workspace
传入用户 YAML 路径
训练完成后自动导出 ONNX
结果保存到 workspace/work_dirs、workspace/onnx、workspace/auto_generated_configs
```

当前核心参数：

```text
project_name
model_type：fibodet_base / fibodet_light / fiboseg_base / fiboseg_light
classes
train_img_path
train_ann_path
val_img_path
val_ann_path
total_epochs
gpu_id
load_from
extra.input_size
```

### 2.2 ClearML 下的实现方式

FiboCV 最适合做成一个 ClearML Task Template。

用户不再手写 Docker 命令，也不需要登录机器，只需要在 ClearML 或 Portal 页面填写：

```text
项目名
任务类型：目标检测 / 语义分割
模型类型
类别列表
训练集路径
验证集路径
训练 epoch
是否 finetune
finetune 权重路径
资源规格
输出目录
```

平台侧负责：

```text
选择镜像 fibo_cv_train:latest
选择 ClearML Queue
挂载 NAS/MinIO/PVC
生成用户 YAML
启动容器训练
采集 stdout 日志
上传 work_dirs、onnx、loss 图、自动生成配置
注册最佳权重和 ONNX 模型
```

### 2.3 Docker 参数到 ClearML/K8s 的映射

| 原 Docker 参数 | ClearML/K8s 下的实现 |
|---|---|
| `--gpus all` | 由 ClearML Queue / K8s Resource Profile 决定 |
| `--shm-size=16g` | Pod 设置 `emptyDir.medium=Memory` 或容器 shm 配置 |
| `-v <data>:/data` | 挂载 NAS/PVC 到 `/data` |
| `-v <workspace>:/app/fibo_cv_train/workspace` | 挂载任务 workspace PVC/NAS |
| `fibo_cv_train:latest` | ClearML Task 的 base docker image |
| `<用户配置文件路径>` | 平台生成 YAML 后传入容器 |

### 2.4 推荐任务模板

建议先做 4 个模板：

| 模板 | 用途 |
|---|---|
| `FiboCV Train` | 正式训练，训练后自动导出 ONNX |
| `FiboCV Finetune` | 基于已有 `.pth` 权重继续训练 |
| `FiboCV ONNX Eval` | 对 ONNX 模型做评测 |
| `FiboCV ONNX Infer` | 对图片或目录做推理验证 |

MVP 阶段可以只做一个 `FiboCV Train`，把 `load_from` 作为可选参数。

### 2.5 FiboCV Wrapper 逻辑

推荐维护一个平台 wrapper，而不是让算法同学继续写 YAML。

Wrapper 逻辑：

```text
读取 ClearML 参数
把参数渲染成 FiboCV 用户 YAML
写入 /app/fibo_cv_train/workspace/<task_id>/config.yaml
调用 FiboCV 原始训练入口
等待训练完成
扫描 workspace 输出目录
上传 artifacts
注册 best.pth 和 onnx 文件
解析日志中的 mAP / mIoU / loss 指标
```

### 2.6 FiboCV 需要注意的问题

1. `gpu_id` 在 K8s 里不应该让用户感知物理卡号。建议写成 `auto` 或固定 `0`，实际资源由 K8s 调度控制。
2. 数据路径必须是容器内路径，例如 `/data/cv_dataset/...`，不能填写开发机本地路径。
3. `workspace` 必须挂载到持久化存储，否则 Pod 结束后产物会丢。
4. 训练完成后的 ONNX、loss 图、best checkpoint 都应该挂到 ClearML Artifact/Model。
5. 内置 VOC 子集可以做成 `FiboCV Smoke Test`，用于验证镜像和队列是否正常。

---

## 3. LLM 微调场景

### 3.1 当前算法同学的使用流程

当前流程可以概括为：

```text
docker run --gpus count=4
挂载个人目录到 /data
进入容器
创建 Python 虚拟环境
安装 ms-swift / LLaMA Factory 等训练框架
准备基座模型和训练数据
执行 swift sft
训练得到 LoRA adapter / checkpoint / loss 日志
执行 swift export 合并 LoRA
启动 vLLM serve 做推理验证
```

这个流程里包含 4 类工作：

```text
环境准备
SFT/LoRA 训练
模型合并
推理验证/评测
```

### 3.2 ClearML 下的实现方式

LLM 场景不建议做成一个大而全任务，而是拆成 Pipeline：

```text
Step 1：模型/数据准备
Step 2：SFT/LoRA 训练
Step 3：LoRA merge / export
Step 4：评测
Step 5：发布到 KServe/vLLM
```

其中：

| 阶段 | 推荐承载 |
|---|---|
| SFT/LoRA 训练 | ClearML Task |
| merge/export | ClearML Task |
| 离线评测 | ClearML Task |
| 临时 vLLM 验证 | 自定义任务或临时 KServe |
| 长期 vLLM 服务 | KServe |
| 交互式进容器调试 | 工作空间或自定义任务 |

### 3.3 Docker/手工操作到 ClearML 的映射

| 当前操作 | ClearML/平台实现 |
|---|---|
| `docker run --gpus count=4` | 选择 `a100-full-4`、`4090-full-4` 等队列或资源规格 |
| `-v /data3/user:/data` | 挂载 NAS/PVC 到 `/data` |
| 进入容器创建 venv | 改为预构建训练镜像，或由 ClearML Agent 管理环境 |
| `export CUDA_VISIBLE_DEVICES=1` | 不让用户填物理 GPU，由 K8s 调度决定 |
| `swift sft ...` | ClearML Task 根据表单参数生成命令 |
| `swift export ...` | 独立 Merge Task |
| `nohup vllm serve ...` | KServe 推理服务或自定义服务，不建议常驻 ClearML |
| `>> log 2>&1` | ClearML 自动采集 stdout/stderr，必要时同步 vLLM 日志 |

### 3.4 推荐 LLM 通用模板

不一定要一个训练框架一个模板。更合理的是：

```text
一个 LLM FineTune 通用模板
多个 backend adapter
```

用户侧只看到一个主入口：

| 通用参数 | 说明 |
|---|---|
| `backend` | `ms-swift` / `llama-factory` / `custom` |
| `model_path` | 容器内模型路径，或平台模型 URI |
| `dataset_path` | 容器内数据路径，或平台数据 URI |
| `output_dir` | 容器内输出路径 |
| `train_method` | `sft` / `dpo` / `rm` / `ppo` / `grpo` |
| `tuner_type` | `lora` / `qlora` / `full` |
| `batch/lr/max_length` | 通用训练超参 |
| `lora_rank/lora_alpha/target_modules` | LoRA 参数 |
| `extra_args` | 高级参数，原样透传给 backend |

平台内部根据 backend 翻译命令：

```text
ms-swift       -> swift sft ...
llama-factory  -> llamafactory-cli train ...
custom         -> 用户自定义命令
```

与微调模板并列保留几个独立阶段模板：

| 模板 | 用途 |
|---|---|
| `LLM Merge LoRA` | 合并 adapter，生成可推理模型 |
| `LLM Eval` | 离线评测模型效果 |
| `LLM Quantization` | AWQ/GPTQ/GGUF 等量化 |

### 3.5 LLM Portal 表单参数建议

训练表单：

```text
任务名称
训练框架：ms-swift / LLaMA Factory
基座模型路径或模型 ID
训练数据路径
输出目录
tuner_type：lora / full / qlora
torch_dtype
num_train_epochs
per_device_train_batch_size
gradient_accumulation_steps
learning_rate
lora_rank
lora_alpha
target_modules
max_length
eval_steps
save_steps
save_total_limit
logging_steps
资源规格
```

高级参数可以先放 JSON/YAML 扩展框，避免模板一开始做得过死。

### 3.6 LLM 产物管理

训练阶段应记录：

```text
LoRA adapter
checkpoint
trainer_state.json
训练日志
loss 曲线
评估指标
训练参数
```

合并阶段应产出：

```text
merged model
tokenizer
config
generation config
模型元信息
```

推理发布阶段使用：

```text
merged model URI
runtime：vLLM / SGLang / TensorRT-LLM
KServe InferenceService
```

### 3.7 LLM 需要注意的问题

1. 不建议让算法同学每次进容器创建 Python 虚拟环境。LLM 训练依赖重，应该维护标准镜像。
2. 基座模型不要从开发机本地路径读取，应该先入库到 NAS/MinIO。
3. 数据集路径统一使用 NAS/MinIO/平台数据集 URI。
4. `CUDA_VISIBLE_DEVICES` 不应该暴露给用户；GPU 数量和类型由资源规格决定。
5. vLLM 长期服务不要作为 ClearML Task 常驻，应该走 KServe。
6. 如果算法同学需要“进容器自己调”，应该通过工作空间或自定义任务入口实现。

---

## 4. 推荐落地顺序

### 4.1 第一阶段：FiboCV MVP

优先做 FiboCV，因为它已经容器化、参数清晰、输出目录固定。

交付：

```text
FiboCV ClearML Task Template
FiboCV 参数表单
NAS/PVC 挂载
自动生成 YAML
训练日志采集
best.pth / onnx / loss 图上传
```

验收：

```text
算法同学不登录机器
只填写数据路径、模型类型、类别、epoch、资源规格
能完成训练并在 ClearML 查看日志和产物
```

### 4.2 第二阶段：LLM SFT MVP

交付：

```text
ms-swift 训练镜像
LLM SFT LoRA Task Template
基座模型路径规范
数据集路径规范
adapter/checkpoint 产物上传
```

验收：

```text
算法同学不手动 docker run
不手动创建 venv
通过 ClearML/Portal 提交一次 LoRA 微调任务
```

### 4.3 第三阶段：LLM Pipeline

交付：

```text
训练 -> Merge -> Eval -> KServe 发布
```

验收：

```text
训练产物可以自动进入合并任务
合并后的模型可以发布为 vLLM 推理服务
```

---

## 5. 对平台方案的影响

这两个场景进一步验证了平台模块划分是合理的：

```text
训练 Tab：承接 FiboCV、LLM SFT、量化、蒸馏等模板化训练
推理 Tab：承接 vLLM/KServe 长期服务发布
自定义任务 Tab：承接临时脚本、临时服务、模型转换、benchmark
工作空间 Tab：承接进容器调试、二次开发、临时验证
```

ClearML 的重点应该是：

```text
任务模板
参数追踪
日志采集
指标记录
产物归档
任务复现
```

不应该把 ClearML 扩展成：

```text
长期推理服务平台
任意容器交互平台
完整 Kubernetes Portal
```

---

## 6. KServe 如何承接两个场景的推理

训练部分由 ClearML 承接，推理部分建议由 KServe 承接。原因是推理服务是长期在线负载，需要服务地址、扩缩容、健康检查、滚动更新、灰度、日志、监控和回滚能力，这不是 ClearML 的核心职责。

### 6.1 FiboCV 推理

FiboCV 训练完成后会产出：

```text
best.pth
ONNX 模型
loss 图
auto generated config
评估结果
```

其中适合发布为在线服务的是 ONNX 模型。

推荐 KServe 方案：

| 需求 | 推荐实现 |
|---|---|
| 标准 ONNX 在线推理 | KServe + Triton Runtime 或 ONNX Runtime ServingRuntime |
| 需要画框、分割 mask、业务后处理 | 自定义 KServe ServingRuntime |
| 批量图片评测、mAP/mIoU 计算 | 不走 KServe，继续作为 ClearML 离线评测任务 |
| 端侧部署验证 | ClearML 产出 ONNX，KServe 只负责在线验证服务 |

FiboCV 推理链路：

```text
ClearML 训练任务产出 ONNX
ONNX 上传到 NAS/MinIO
ClearML Model 记录模型 URI
Portal/KServe 创建 InferenceService
KServe Runtime 加载 ONNX
对外提供 HTTP/gRPC 推理接口
```

如果只是算法同学想看一批图片的推理可视化结果，仍建议用 ClearML `FiboCV ONNX Infer` 离线任务，不需要发布 KServe 服务。

### 6.2 LLM 推理

LLM 微调完成后通常会有：

```text
LoRA adapter
checkpoint
merged model
tokenizer
config
评测报告
```

适合发布为在线服务的是 `merged model`，或者支持 LoRA 动态加载时的 base model + adapter。

推荐 KServe 方案：

| 需求 | 推荐实现 |
|---|---|
| 通用 OpenAI API 兼容服务 | KServe + vLLM Runtime |
| 更高性能优化 | KServe + TensorRT-LLM Runtime |
| SGLang 特性 | KServe + 自定义 SGLang ServingRuntime |
| 临时手工验证 | 自定义任务或临时 KServe，不放 ClearML 常驻 |
| 生产长期服务 | KServe InferenceService |

LLM 推理链路：

```text
ClearML SFT 任务产出 LoRA/checkpoint
ClearML Merge 任务产出 merged model
merged model 上传 NAS/MinIO
Portal/KServe 创建 InferenceService
vLLM/SGLang/TRT-LLM Runtime 加载模型
对外提供 OpenAI-compatible API
```

原始命令中的参数可以映射到 KServe Runtime args，例如：

```text
--served-model-name
--tensor-parallel-size
--max-model-len
--reasoning-parser
--enable-auto-tool-choice
--tool-call-parser
--trust-remote-code
--enable-prefix-caching
--gpu-memory-utilization
--max-num-batched-tokens
--max-num-seqs
```

### 6.3 KServe 与 HAMi/vGPU 的资源语义问题

KServe 官方示例常见写法是：

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

但如果你的推理节点也只暴露 `volcano.sh/vgpu-*`，则不能直接照抄官方 GPU 资源写法，需要让 KServe 生成的 Predictor Pod 也带上：

```yaml
metadata:
  annotations:
    volcano.sh/vgpu-mode: "hami-core"
spec:
  schedulerName: volcano
  runtimeClassName: nvidia
  containers:
  - resources:
      limits:
        volcano.sh/vgpu-number: "1"
        volcano.sh/vgpu-memory: "24"
        volcano.sh/vgpu-cores: "100"
```

这块需要在落地时做一次技术验证。可选实现方式：

```text
方式一：KServe/ServingRuntime 模板中暴露对应 pod spec、annotation、resource
方式二：使用 MutatingWebhook 给 KServe Predictor Pod 注入 schedulerName、annotation、runtimeClassName、volcano.sh/vgpu-*
方式三：推理服务单独使用 NVIDIA device plugin 整卡/MIG 资源池，KServe 继续使用 nvidia.com/gpu 或 nvidia.com/mig-*
```

工程上最稳的方案是：

```text
训练调试、小模型推理：HAMi/vGPU 资源池
高 SLA LLM 推理：整卡或 MIG 资源池
KServe 先在整卡/MIG 池跑通，再评估 HAMi/vGPU 池
```

---

## 7. 最终建议

建议先把 FiboCV 做成第一批训练模板，作为 ClearML 平台 MVP 的样板场景。

LLM 微调作为第二批模板，先支持 ms-swift SFT/LoRA，再扩展 LLaMA Factory、merge、eval 和 KServe 发布。

平台面向算法同学的最终体验应是：

```text
选择训练模板
选择模型和数据
填写少量关键参数
选择 GPU 规格
提交任务
查看日志和指标
下载或发布模型产物
```

底层的 Docker、K8s YAML、GPU 资源名、PVC、模型缓存和日志归档由平台统一处理。
