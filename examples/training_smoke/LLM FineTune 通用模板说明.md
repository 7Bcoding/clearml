# LLM FineTune 通用模板说明

> 目标：用一个 ClearML Task 模板统一承接 LLaMA-Factory、ms-swift 和自定义命令的 LLM 微调任务。  
> 调研基准时间：2026-07-03。

---

## 1. 结论

现在已经有一个真正的通用模板脚本：

```text
examples/training_smoke/llm_finetune_universal.py
```

它不是固定 `ms-swift` 的 smoke wrapper，而是：

```text
同一个 ClearML Task
同一套模型/数据/输出/资源参数
backend = llama-factory / ms-swift / custom
内部按 backend 渲染不同训练命令
统一接 ClearML VGPU、Volcano 队列和 /data 挂载
```

---

## 2. 官方依据

LLaMA-Factory 官方 README 说明：

```text
支持 100+ 模型和多种训练方式
支持 CLI 和 WebUI
训练入口：llamafactory-cli train <yaml>
合并入口：llamafactory-cli export <yaml>
WebUI 入口：llamafactory-cli webui
自定义数据需要 dataset_info.json
```

参考：

- LLaMA-Factory GitHub: https://github.com/hiyouga/LLaMA-Factory
- LLaMA-Factory Data README: https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/data/README.md

---

## 3. 模板参数分层

### 3.1 平台参数

```text
queue
docker-image
clearml-project
clearml-task-name
VGPU: vgpu_number / vgpu_memory / vgpu_cores
```

### 3.2 通用训练参数

```text
backend
model_path
dataset / dataset_dir / dataset_path
output_dir
run_name
train_method
finetuning_type
template
dtype
max_steps / num_train_epochs
batch / gradient_accumulation / learning_rate / max_length
lora_rank / lora_alpha / lora_target
extra_args
```

### 3.3 后端参数

```text
backend=llama-factory
  生成 llamafactory_train.yaml
  自动生成 dataset_info.json
  执行 llamafactory-cli train <yaml>

backend=ms-swift
  拼接 swift sft 命令
  执行 swift sft ...

backend=custom
  执行用户传入的 custom-command
```

---

## 4. LLaMA-Factory 冒烟示例

前提：模型和数据已经在训练 Pod 内可见。

```text
/data/models/Qwen2.5-0.5B-Instruct
/data/datasets/demo_sft/train.jsonl
```

提交命令：

```bash
python examples/training_smoke/llm_finetune_universal.py \
  --backend llama-factory \
  --queue llm-finetune-vgpu \
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
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target all \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

这会在远程 Pod 里生成：

```text
/data/output/llamafactory-sft-lora-smoke/_clearml_template/llamafactory_train.yaml
/data/output/llamafactory-sft-lora-smoke/_clearml_template/llamafactory_dataset/dataset_info.json
```

然后执行：

```bash
llamafactory-cli train /data/output/.../_clearml_template/llamafactory_train.yaml
```

---

## 5. ms-swift 后端示例

```bash
python examples/training_smoke/llm_finetune_universal.py \
  --backend ms-swift \
  --queue llm-finetune-vgpu \
  --docker-image harbor.example.com/ai/ms-swift-cuda12.1:latest \
  --clearml-project training-template/llm \
  --clearml-task-name ms-swift-sft-lora-smoke \
  --model-path /data/models/Qwen2.5-0.5B-Instruct \
  --dataset-path /data/datasets/demo_sft/train.jsonl \
  --output-dir /data/output/ms-swift-sft-lora-smoke \
  --max-steps 2 \
  --per-device-train-batch-size 1 \
  --max-length 512 \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

---

## 6. custom 后端示例

```bash
python examples/training_smoke/llm_finetune_universal.py \
  --backend custom \
  --queue llm-finetune-vgpu \
  --docker-image harbor.example.com/ai/custom-trainer:latest \
  --clearml-project training-template/llm \
  --clearml-task-name custom-finetune-smoke \
  --model-path /data/models/Qwen2.5-0.5B-Instruct \
  --dataset-path /data/datasets/demo_sft/train.jsonl \
  --output-dir /data/output/custom-finetune-smoke \
  --custom-command "python train.py --model {model_path} --data {dataset_path} --out {output_dir}" \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

---

## 7. 数据格式说明

### 7.1 Alpaca 格式

默认格式是 `alpaca`。

数据列默认：

```text
instruction
input
output
system
history
```

模板会自动生成：

```json
{
  "clearml_dataset": {
    "file_name": "train.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

### 7.2 ShareGPT / OpenAI messages 格式

ShareGPT：

```bash
--dataset-format sharegpt \
--messages-column conversations
```

OpenAI messages：

```bash
--dataset-format sharegpt \
--dataset-openai-messages \
--messages-column messages
```

---

## 8. ClearML WebUI 复跑

第一次从命令行提交成功后：

```text
Clone Task
修改 CONFIGURATION -> Args:
  backend
  model_path
  dataset_path
  output_dir
  train_method
  finetuning_type
  batch/lr/max_length/lora 参数
修改 CONFIGURATION -> VGPU:
  vgpu_number
  vgpu_memory
  vgpu_cores
Enqueue 到 llm-finetune-vgpu
```

这就是 Portal 之前的可用版本。

---

## 9. 与旧脚本的关系

| 脚本 | 定位 |
|---|---|
| `llm_swift_sft_smoke.py` | 固定 ms-swift 的早期 smoke wrapper |
| `llm_finetune_universal.py` | 新的通用 LLM FineTune 模板 |

后续建议把 `llm_finetune_universal.py` 作为团队模板主入口。

