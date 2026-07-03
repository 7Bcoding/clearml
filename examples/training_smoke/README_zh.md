# ClearML 训练冒烟测试

这组脚本用于验证 `ClearML + Volcano + volcano-vgpu-device-plugin/HAMi-core` 是否能承接两类训练场景：

```text
FiboCV：Docker 镜像 + YAML 配置 + workspace 输出
LLM：ms-swift SFT/LoRA + 模型/数据路径 + checkpoint 输出
```

## 前置条件

1. ClearML Server 可访问。
2. ClearML Agent/K8s Glue 已部署，并监听 `volcano-queue` 或你的实际队列。
3. Agent 启用了 vGPU Hook，读取 ClearML Task 的 `CONFIGURATION -> VGPU` 段。
4. GPU 节点暴露 `volcano.sh/vgpu-number`、`volcano.sh/vgpu-memory`、`volcano.sh/vgpu-cores`。
5. 训练镜像已在节点可拉取：
   - FiboCV：`fibo_cv_train:latest`
   - LLM：`llamafactory:latest`、`ms-swift-cuda12.1:latest` 或你的实际镜像
6. 训练 Pod 能访问 NAS/PVC/MinIO 中的模型和数据路径。

## 路径语义

冒烟命令里的路径不是提交脚本所在开发机的路径，而是远程训练 Pod 容器内能访问的路径。

例如：

```text
--model /data/models/Qwen2.5-0.5B-Instruct
--dataset /data/datasets/demo_sft/train.jsonl
```

表示训练 Pod 内存在这些路径。它们通常来自 NAS/PVC 挂载：

```text
NAS: /mnt/nas/models/Qwen2.5-0.5B-Instruct
Pod: /data/models/Qwen2.5-0.5B-Instruct
```

不要填写只在提交机存在的路径，例如：

```text
C:\Users\xxx\Downloads\model
/home/user/local-models
```

## 挂载在哪里完成

挂载在 ClearML Agent/K8s Glue 的 `basePodTemplate` 里完成，而不是在训练脚本里完成。

推荐：

```text
Kubernetes PVC/PV
-> ClearML Agent values.yaml 的 basePodTemplate.volumes
-> basePodTemplate.volumeMounts 挂到 /data
-> 训练脚本使用 /data/models、/data/datasets、/data/output
```

LLM 微调建议单独一个队列和 Pod Template，例如：

```text
queue: llm-finetune-vgpu
mount: /data
shm: /dev/shm 128Gi
cache: /data/cache/huggingface, /data/cache/modelscope
```

FiboCV 可以单独一个队列和 Pod Template，例如：

```text
queue: cv-train-vgpu
mount: /data
mount: /app/fibo_cv_train/workspace
shm: /dev/shm 16Gi
```

MVP 也可以先所有训练队列共用一个 `/data` 挂载；生产上再按 CV/LLM/评测拆队列和 Pod Template。

## 模型和数据从哪里来

推荐先做“下载入库”，再跑训练。

模型常见来源：

```text
Hugging Face Hub
ModelScope
公司内部模型仓库
算法同学本地已有模型上传到 NAS/MinIO
```

示例：

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

数据常见来源：

```text
CV：标注平台导出的 COCO/VOC 数据
LLM：内部 JSON/JSONL、Hugging Face Dataset、ModelScope Dataset
冒烟：小样本 JSONL 或 FiboCV 镜像内置 VOC
```

大模型和大 checkpoint 不建议上传到 ClearML Server 文件服务，优先放 NAS/MinIO，ClearML 记录路径和 manifest。

## FiboCV 冒烟测试

优先使用 FiboCV 镜像内置 VOC 示例，避免一开始依赖外部数据。

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

如果 FiboCV 镜像入口不是 `python /app/fibo_cv_train/tools/train_dispatcher.py <config>`，用 `--entry-cmd` 覆盖：

```bash
--entry-cmd "你的入口命令 {config_path}"
```

通过标准：

```text
ClearML Task 进入队列并被 Agent 拉起
Pod 使用 volcano scheduler
Pod 资源为 volcano.sh/vgpu-*
日志里能看到 FiboCV 训练启动
训练结束后 ClearML ARTIFACTS 有 work_dirs / onnx / auto_generated_configs
ClearML MODELS 有 ONNX 模型或日志说明未找到 ONNX
```

## LLM FineTune 通用模板冒烟测试

LLM 冒烟测试建议使用小模型或真实模型的极小步数。目标是验证链路，不是验证效果。

推荐优先使用 LLaMA-Factory backend：

```bash
python examples/training_smoke/llm_finetune_universal.py \
  --backend llama-factory \
  --queue volcano-queue \
  --docker-image harbor.example.com/ai/llamafactory:latest \
  --clearml-project training-template/llm \
  --clearml-task-name llama-factory-sft-lora-smoke \
  --model-path /data/models/Qwen2.5-0.5B-Instruct \
  --dataset-path /data/datasets/demo_sft/train.jsonl \
  --dataset-format alpaca \
  --output-dir /data/output/llamafactory-sft-lora-smoke \
  --train-method sft \
  --finetuning-type lora \
  --template qwen \
  --max-steps 2 \
  --per-device-train-batch-size 1 \
  --max-length 512 \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

通过标准：

```text
ClearML Task 进入队列并被 Agent 拉起
训练容器能访问 /data/models 和 /data/datasets
swift sft 正常启动
至少完成 1-2 个 step
output_dir 下有 checkpoint 或 adapter 相关输出
ClearML ARTIFACTS 有 training-output-manifest
```

## LLM 微调模板建议

生产上不一定要“一个框架一个入口模板”。现在已经提供通用脚本：

```text
examples/training_smoke/llm_finetune_universal.py
```

建议做成：

```text
一个 LLM FineTune 通用模板
+ 多个 backend adapter
```

用户看到：

```text
backend = ms-swift / llama-factory / custom
model_path
dataset_path
output_dir
tuner_type
batch/lr/max_length/lora 参数
extra_args
VGPU 资源规格
```

平台内部把通用参数翻译成不同框架的命令：

```text
ms-swift      -> swift sft ...
LLaMA Factory -> llamafactory-cli train ...
custom        -> 用户自定义命令
```

MVP 先实现 `ms-swift adapter` 即可，后续再增加 LLaMA Factory。

## WebUI 复跑方式

1. 在 ClearML WebUI 中打开冒烟任务。
2. Clone 任务。
3. 修改 `CONFIGURATION -> Args` 中的模型、数据、epoch、batch 等参数。
4. 修改 `CONFIGURATION -> VGPU` 中的资源规格。
5. Enqueue 到对应队列。

## 注意事项

1. 在 HAMi/vGPU-only 集群里不要使用 `nvidia.com/gpu`，要使用 `volcano.sh/vgpu-*`。
2. 不要让算法同学填写物理 `gpu_id` 或 `CUDA_VISIBLE_DEVICES`。
3. FiboCV 的 `gpu_id` 建议固定为 `auto`，实际资源由 Kubernetes/Volcano 决定。
4. LLM 依赖较重，建议使用预构建镜像，不要每次任务里临时安装 ms-swift 或 LLaMA Factory。
5. 大模型 checkpoint 不建议默认上传到 ClearML Server，建议保存在 NAS/MinIO，ClearML 只记录 manifest 和路径。
