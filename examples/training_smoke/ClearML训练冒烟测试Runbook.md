# ClearML LLM 微调冒烟测试 Runbook

> 目标：用最小闭环验证 `ClearML + Kubernetes + Volcano + volcano-vgpu-device-plugin/HAMi-core + host local PV` 能承接 LLM 微调训练任务。  
> 本次只验证训练链路，不验证 KServe 推理。  
> 本次先不做 FiboCV，只跑 LLaMA-Factory 优先的 LLM FineTune 通用模板。  
> 调研基准时间：2026-07-03。

---

## 0. 本次冒烟要验证什么

本次冒烟只验证一条链路：

```text
ClearML Task
-> ClearML Agent / K8s Glue
-> Kubernetes Training Pod
-> Volcano Scheduler
-> HAMi/vGPU
-> LLaMA-Factory 镜像
-> SFT LoRA 1-2 step
-> /data/output 产物 + ClearML 日志/Artifact
```

本次先使用一台 GPU 节点的本地目录作为 host local PV：

```text
GPU 节点本地：/data/ai-local
训练 Pod 内：/data
```

先不引入 NFS/NAS/MinIO CSI，减少变量。等这条链路跑通后，再把 `/data` 从 host local PV 切换成共享存储。

---

## 1. 前置条件

### 1.1 已有组件

需要先确认：

```text
Kubernetes 集群可用
Volcano 已安装
volcano-vgpu-device-plugin/HAMi-core 已安装
ClearML Server 可访问
ClearML Agent Helm Chart 可安装
kubectl / helm 可操作集群
GPU 节点可以拉取 LLaMA-Factory 训练镜像
```

### 1.2 本地文件

本仓库里已经准备好这些文件：

```text
examples/training_smoke/k8s/local-pv-ai-data.example.yaml
examples/training_smoke/k8s/volcano-queue-llm-finetune-vgpu.yaml
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml

examples/training_smoke/llm_finetune_universal.py
examples/training_smoke/requirements-smoke.txt
```

其中 `llm_finetune_universal.py` 是本次要跑的通用微调模板，支持：

```text
backend = llama-factory / ms-swift / custom
```

本次冒烟建议优先使用：

```text
backend = llama-factory
```

### 1.3 路径原则

冒烟脚本参数里的路径都是 **训练 Pod 内路径**，不是提交脚本所在机器的路径。

```text
GPU 节点本地路径：/data/ai-local/models/xxx
训练 Pod 内路径：/data/models/xxx
```

例如命令里填：

```text
--model-path /data/models/Qwen2.5-0.5B-Instruct
--dataset-path /data/datasets/demo_sft/train.jsonl
--output-dir /data/output/llamafactory-sft-lora-smoke
```

实际文件应该放在目标 GPU 节点：

```text
/data/ai-local/models/Qwen2.5-0.5B-Instruct
/data/ai-local/datasets/demo_sft/train.jsonl
/data/ai-local/output
```

不要填只在提交机存在的路径，例如：

```text
C:\Users\xxx\Downloads\model
/home/user/local-models
```

---

## 2. 第一步：选择 GPU 节点

在有 `kubectl` 的机器上执行：

```bash
kubectl get nodes -o wide
```

选一台用于冒烟的 GPU 节点，例如：

```text
gpu-server-6
```

查看节点 hostname label：

```bash
kubectl get node gpu-server-6 --show-labels | tr ',' '\n' | grep kubernetes.io/hostname
```

记下：

```text
<GPU_NODE_HOSTNAME>
```

后面两个文件里都要指向同一个节点：

```text
examples/training_smoke/k8s/local-pv-ai-data.yaml 的 nodeAffinity
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.local.yaml 的 nodeSelector
```

---

## 3. 第二步：准备 GPU 节点本地目录

登录目标 GPU 节点：

```bash
ssh <gpu-node>
```

创建 local PV 目录：

```bash
sudo mkdir -p /data/ai-local/{models,datasets,output,cache,clearml}
sudo chmod -R 0777 /data/ai-local
```

目录含义：

```text
/data/ai-local/models      -> Pod 内 /data/models
/data/ai-local/datasets    -> Pod 内 /data/datasets
/data/ai-local/output      -> Pod 内 /data/output
/data/ai-local/cache       -> Pod 内 /data/cache
/data/ai-local/clearml     -> 可选 ClearML 缓存
```

检查：

```bash
ls -lah /data/ai-local
```

---

## 4. 第三步：准备模型、数据和镜像

### 4.1 准备 LLaMA-Factory 镜像

本次冒烟建议准备一个预装以下组件的镜像：

```text
CUDA / PyTorch
transformers / datasets / peft / accelerate
LLaMA-Factory
clearml
```

示例镜像名：

```text
harbor.example.com/ai/llamafactory:latest
```

在 GPU 节点上确认可以拉取：

```bash
docker pull harbor.example.com/ai/llamafactory:latest
```

如果你们暂时只有 ms-swift 镜像，也可以先用同一个通用模板切到 `--backend ms-swift`，但本 Runbook 默认以 LLaMA-Factory 为主。

### 4.2 准备冒烟模型

建议用小模型，优先：

```text
Qwen2.5-0.5B-Instruct
```

下载到目标 GPU 节点的 local PV 目录。

方式 A：Hugging Face：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /data/ai-local/models/Qwen2.5-0.5B-Instruct
```

方式 B：ModelScope：

```bash
modelscope download \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir /data/ai-local/models/Qwen2.5-0.5B-Instruct
```

方式 C：如果模型已经在别的机器：

```bash
rsync -avP /path/to/Qwen2.5-0.5B-Instruct/ \
  <gpu-node>:/data/ai-local/models/Qwen2.5-0.5B-Instruct/
```

检查：

```bash
ls -lah /data/ai-local/models/Qwen2.5-0.5B-Instruct
```

至少应该能看到：

```text
config.json
tokenizer 相关文件
model 权重文件
```

### 4.3 准备冒烟数据

如果现在完全没有微调数据，冒烟阶段不要先卡在数据集建设上。建议按下面优先级准备数据：

```text
优先级 1：手工造 2-10 条最小 Alpaca 数据，用来验证链路
优先级 2：使用 LLaMA-Factory 自带 demo 数据
优先级 3：从 Hugging Face / ModelScope 下载公开 SFT 数据
优先级 4：接入公司内部真实业务数据，先脱敏、清洗、抽样
```

本次冒烟最推荐用优先级 1。它不验证模型效果，只验证 ClearML、K8s、Volcano、HAMi、镜像、挂载、日志和产物链路是否打通。

#### 4.3.1 方式 A：手工创建最小 SFT 数据

在目标 GPU 节点上执行：

```bash
mkdir -p /data/ai-local/datasets/demo_sft

cat > /data/ai-local/datasets/demo_sft/train.jsonl <<'EOF'
{"instruction":"你是谁？","input":"","output":"我是一个用于冒烟测试的助手。"}
{"instruction":"请把下面的话改写得更正式。","input":"今天活儿挺多。","output":"今天的工作任务较为繁重。"}
{"instruction":"把下面中文翻译成英文。","input":"模型微调任务已经提交成功。","output":"The model fine-tuning task has been submitted successfully."}
{"instruction":"请总结下面这句话。","input":"ClearML 负责实验追踪和任务提交，Kubernetes 负责资源调度，Volcano 和 HAMi 负责 vGPU 调度。","output":"该平台使用 ClearML 管理训练任务，并通过 Kubernetes、Volcano 和 HAMi 调度 GPU 资源。"}
EOF
```

Host 路径：

```text
/data/ai-local/datasets/demo_sft/train.jsonl
```

Pod 内路径：

```text
/data/datasets/demo_sft/train.jsonl
```

后面提交训练任务时使用：

```text
--dataset-path /data/datasets/demo_sft/train.jsonl
--dataset-format alpaca
```

检查：

```bash
ls -lah /data/ai-local/datasets/demo_sft
head -n 4 /data/ai-local/datasets/demo_sft/train.jsonl
```

#### 4.3.2 方式 B：使用 LLaMA-Factory 自带 demo 数据

你的本地仓库 `D:\Python\LlamaFactory\data` 里已经有示例数据，可以直接拿来做冒烟：

```text
alpaca_zh_demo.json       中文 Alpaca SFT 示例
alpaca_en_demo.json       英文 Alpaca SFT 示例
identity.json             身份认知类 SFT 示例
v1_sft_demo.jsonl         v1 版本 SFT 示例
```

如果 GPU 节点可以访问外网，可以在 GPU 节点上拉取 LLaMA-Factory 仓库并复制 demo 数据：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
mkdir -p /data/ai-local/datasets/llamafactory_demo
cp /tmp/LLaMA-Factory/data/alpaca_zh_demo.json \
  /data/ai-local/datasets/llamafactory_demo/alpaca_zh_demo.json
```

如果 GPU 节点不能访问外网，可以从开发机上传。Windows PowerShell 示例：

```powershell
ssh <gpu-node> "mkdir -p /data/ai-local/datasets/llamafactory_demo"
```

```powershell
scp D:\Python\LlamaFactory\data\alpaca_zh_demo.json `
  <gpu-node>:/data/ai-local/datasets/llamafactory_demo/alpaca_zh_demo.json
```

后面提交训练任务时使用：

```text
--dataset-path /data/datasets/llamafactory_demo/alpaca_zh_demo.json
--dataset-format alpaca
```

注意：当前通用模板在传入 `--dataset-path` 时，会自动为 LLaMA-Factory 生成临时 `dataset_info.json`，所以冒烟阶段不需要你手工维护 LLaMA-Factory 的 `dataset_info.json`。

#### 4.3.3 方式 C：下载公开 SFT 数据

如果你希望更接近真实微调，可以从公开数据集下载小样本。常见来源：

```text
Hugging Face Datasets
ModelScope Datasets
LLaMA-Factory 官方 dataset_info.json 中登记的数据集
```

建议第一次只抽 100-1000 条，不要直接下载大数据集。示例流程：

```text
1. 下载公开数据集
2. 抽样 100-1000 条
3. 转换成 Alpaca 或 ShareGPT 格式
4. 放到 /data/ai-local/datasets/<dataset_name>/train.jsonl
5. Pod 内通过 /data/datasets/<dataset_name>/train.jsonl 访问
```

Alpaca 格式最简单：

```json
{
  "instruction": "用户指令，必填",
  "input": "用户输入，可为空",
  "output": "模型应该学习的回答，必填"
}
```

JSONL 文件每行一个样本：

```jsonl
{"instruction":"解释什么是 LoRA。","input":"","output":"LoRA 是一种参数高效微调方法，通过训练低秩矩阵来减少显存和参数更新量。"}
{"instruction":"把下面的话总结成一句。","input":"我们使用 ClearML 提交任务，并由 Volcano 调度到 GPU 节点运行。","output":"该训练任务通过 ClearML 提交，并由 Volcano 调度到 GPU 节点执行。"}
```

#### 4.3.4 方式 D：使用内部真实业务数据

正式微调时，数据通常来自：

```text
业务问答知识库
客服/运维/售后对话，需脱敏
人工标注平台导出的指令数据
专家整理的标准问答
历史 prompt 和优质回复
RAG 问答日志中人工确认过的样本
```

正式数据进入训练前建议至少做这些处理：

```text
去除手机号、身份证、邮箱、客户名等敏感信息
去除重复样本
去除空回答、乱码、过短或过长样本
统一成 Alpaca 或 ShareGPT 格式
先抽样 100 条做冒烟，再扩大到完整数据
```

本次冒烟不要求真实业务数据。先用方式 A 或方式 B 跑通链路即可。

---

## 5. 第四步：复制并修改安装 YAML

建议不要直接改 `.example.yaml`，先复制一份本地文件。

在仓库根目录执行：

```bash
cp examples/training_smoke/k8s/local-pv-ai-data.example.yaml \
   examples/training_smoke/k8s/local-pv-ai-data.yaml

cp examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.yaml \
   examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.local.yaml
```

把 local PV 文件里的 `nodeAffinity`，以及 Agent overlay 文件里的：

```text
<GPU_NODE_HOSTNAME>
```

都改成第二步选定的节点 hostname。

例如：

```text
gpu-server-6
```

### 5.1 修改 local PV 容量

打开：

```text
examples/training_smoke/k8s/local-pv-ai-data.yaml
```

按本地磁盘实际容量修改：

```yaml
capacity:
  storage: 2Ti
```

如果只选一台 GPU 节点做冒烟，建议 `nodeAffinity` 里只保留这一台节点，降低调度误差。

### 5.2 修改 ClearML Agent values

打开：

```text
examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml
```

必须确认/修改 ClearML Server 地址：

```yaml
apiServerUrlReference: "http://<clearml-api>"
fileServerUrlReference: "http://<clearml-file-server>"
webServerUrlReference: "http://<clearml-web>"
```

确认队列名：

```yaml
queue: llm-finetune-vgpu
```

确认默认镜像。建议使用 LLaMA-Factory 镜像：

```yaml
defaultContainerImage: harbor.example.com/ai/llamafactory:latest
```

即使这里没改，后面的提交命令也会通过 `--docker-image` 指定训练镜像；但为了减少误解，建议保持一致。

### 5.3 检查 Pod Template 挂载

确认 `basePodTemplate` 里有 `/data` 挂载：

```yaml
volumes:
  - name: ai-data
    persistentVolumeClaim:
      claimName: ai-data-pvc

volumeMounts:
  - name: ai-data
    mountPath: /data
```

确认使用 Volcano 和 HAMi：

```yaml
schedulerName: volcano
runtimeClassName: nvidia
annotations:
  volcano.sh/vgpu-mode: "hami-core"
  volcano.sh/queue-name: llm-finetune-vgpu
```

在 HAMi/vGPU-only 集群里不要配置：

```text
nvidia.com/gpu
```

### 5.4 修改 Volcano Queue capability

打开：

```text
examples/training_smoke/k8s/volcano-queue-llm-finetune-vgpu.yaml
```

确认队列配额：

```yaml
capability:
  cpu: "64"
  memory: 512Gi
  volcano.sh/vgpu-number: "8"
  volcano.sh/vgpu-memory: "192"
  volcano.sh/vgpu-cores: "800"
```

如果只是单卡冒烟，可以先保守一点：

```yaml
capability:
  cpu: "16"
  memory: 128Gi
  volcano.sh/vgpu-number: "1"
  volcano.sh/vgpu-memory: "24"
  volcano.sh/vgpu-cores: "100"
```

这里的 `volcano.sh/vgpu-memory` 是队列总配额。当前模板约定 ClearML 里填的 `vgpu_memory=24` 表示 24GiB，并由 Agent vgpuHook 按 `gpuMemoryFactor=1024` 转换成 HAMi 需要的资源单位。

---

## 6. 第五步：安装 local PV / PVC

在有 `kubectl` 的机器执行：

```bash
kubectl apply -f examples/training_smoke/k8s/local-pv-ai-data.yaml
```

检查 PV/PVC：

```bash
kubectl get storageclass local-ai-data
kubectl get pv local-ai-data-pv-gpu-node
kubectl -n clearml get pvc ai-data-pvc
```

预期：

```text
PV 存在
PVC 存在
PVC 可能先是 Pending
```

`WaitForFirstConsumer` 模式下，PVC 在训练 Pod 创建前 Pending 是正常的。

---

## 7. 第六步：安装 Volcano Queue

```bash
kubectl apply -f examples/training_smoke/k8s/volcano-queue-llm-finetune-vgpu.yaml
```

检查：

```bash
kubectl get queue llm-finetune-vgpu
kubectl describe queue llm-finetune-vgpu
```

确认里面是：

```text
volcano.sh/vgpu-number
volcano.sh/vgpu-memory
volcano.sh/vgpu-cores
```

不要出现：

```text
nvidia.com/gpu
```

---

## 8. 第七步：安装 LLM 微调 ClearML Agent

### 8.1 检查 ClearML Agent Secret

```bash
kubectl -n clearml get secret clearml-agent-creds
```

如果不存在，需要先按你们 ClearML Agent 安装方式创建 API credentials secret。

### 8.2 添加 Helm repo

```bash
helm repo add clearml https://clearml.github.io/clearml-helm-charts
helm repo update
```

### 8.3 安装 Agent

```bash
helm upgrade --install clearml-agent-llm-finetune \
  clearml/clearml-agent \
  -n clearml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.local.yaml
```

### 8.4 检查 Agent

```bash
kubectl -n clearml get pods | grep llm-finetune
kubectl -n clearml get deploy | grep llm-finetune
```

看日志：

```bash
kubectl -n clearml logs deploy/clearml-agent-llm-finetune --tail=100
```

预期：

```text
Agent 正常启动
监听 llm-finetune-vgpu 队列
没有 ClearML API 认证错误
```

---

## 9. 第八步：检查 vGPU 资源

检查节点 allocatable：

```bash
kubectl get nodes -o json | jq -r '.items[] |
  "\(.metadata.name) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"]) vgpu-cores=\(.status.allocatable["volcano.sh/vgpu-cores"])"'
```

预期目标 GPU 节点有：

```text
volcano.sh/vgpu-number
volcano.sh/vgpu-memory
volcano.sh/vgpu-cores
```

如果没有，先不要跑 ClearML 冒烟，先排查 `volcano-vgpu-device-plugin/HAMi-core`。

---

## 10. 第九步：提交 LLaMA-Factory 微调冒烟任务

在配置好 ClearML SDK 的提交机上执行：

```bash
python examples/training_smoke/llm_finetune_universal.py \
  --backend llama-factory \
  --queue llm-finetune-vgpu \
  --docker-image hiyouga/llamafactory:latest \
  --clearml-project training-template/llm \
  --clearml-task-name llama-factory-sft-lora-smoke \
  --reuse-last-task-id false \
  --store-standalone-script true \
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
  --cuda-visible-devices 0 \
  --nproc-per-node 1 \
  --force-torchrun false \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

参数含义：

```text
--backend llama-factory                         使用 LLaMA-Factory 后端
--queue llm-finetune-vgpu                       投递到 LLM 微调队列
--docker-image                                  训练 Pod 使用的镜像
--reuse-last-task-id false                      强制创建新 Task，避免复用旧 Task 的 Git 仓库信息
--store-standalone-script true                  把当前脚本存入 ClearML Task，避免训练 Pod clone 远端 Git 仓库
--model-path /data/models/...                   Pod 内模型路径
--dataset-path /data/datasets/...               Pod 内数据路径
--output-dir /data/output/...                   Pod 内输出路径
--dataset-format alpaca                         数据按 Alpaca 格式解释
--finetuning-type lora                          使用 LoRA 微调
--max-steps 2                                   只跑 2 step，用于冒烟
--cuda-visible-devices 0                        只让训练进程看到 1 张 GPU，避免自动 DDP
--nproc-per-node 1                              LLaMA-Factory torchrun 进程数固定为 1
--force-torchrun false                          冒烟阶段不要强制分布式训练
--vgpu-number / --vgpu-memory / --vgpu-cores    注入 volcano.sh/vgpu-* 资源
```

提交成功后，本地命令通常会很快结束，真正训练会由 ClearML Agent 在 K8s 中拉起 Pod。

当前通用模板默认启用 `--reuse-last-task-id false` 和 `--store-standalone-script true`，适合冒烟和单文件模板。这样 ClearML 会创建一个新 Task，并把当前脚本内容存到 Task 里，训练 Pod 不需要访问 GitHub 克隆代码仓库。

注意：standalone script 模式适合当前这种单文件模板。如果后续模板拆成多个 Python 文件、依赖本地模块或大量配置文件，建议改为内网 Git 仓库、内部 Python 包或把模板代码打进训练镜像。

---

## 11. 第十步：验收任务是否跑通

### 11.1 ClearML WebUI 验收

打开 ClearML WebUI，检查：

```text
项目：training-template/llm
任务：llama-factory-sft-lora-smoke
状态：Queued -> Running -> Completed
Console：能看到 llamafactory-cli train 启动
Console：至少跑完 1-2 step
Artifacts：llm-finetune-manifest
Artifacts：llamafactory-train-config
Artifacts：llamafactory-dataset-info
```

### 11.2 K8s Pod 验收

查找训练 Pod：

```bash
kubectl get pods -A | grep llama
kubectl get pods -A | grep clearml
```

描述 Pod：

```bash
kubectl describe pod <llm-pod> -n <namespace> | egrep -i 'volcano|vgpu|runtimeClass|node|ai-data|/data|image'
```

确认：

```text
Pod 调度到 local PV 所在节点
schedulerName = volcano
runtimeClassName = nvidia
/data 挂载成功
volcano.sh/vgpu-* 资源存在
没有 nvidia.com/gpu
镜像是 harbor.example.com/ai/llamafactory:latest
```

### 11.3 GPU 节点输出验收

登录目标 GPU 节点：

```bash
ssh <gpu-node>
```

检查输出目录：

```bash
ls -lah /data/ai-local/output/llamafactory-sft-lora-smoke
```

预期能看到 LLaMA-Factory 训练输出，例如：

```text
checkpoint 相关目录或文件
trainer_state.json
adapter_config.json
adapter_model 相关文件
```

不同版本的 LLaMA-Factory 输出文件名可能略有差异，冒烟阶段以“任务进入 Pod、能读模型数据、能跑完 1-2 step、输出目录有内容”为准。

---

## 12. 第十一步：在 ClearML WebUI 复跑

冒烟命令第一次提交成功后，后续算法同学可以不用命令行。

操作：

```text
打开 ClearML WebUI
进入 training-template/llm
打开 llama-factory-sft-lora-smoke
Clone
修改 CONFIGURATION -> Args
  backend
  model_path
  dataset_path
  dataset_format
  output_dir
  train_method
  finetuning_type
  template
  max_steps
  per_device_train_batch_size
  gradient_accumulation_steps
  learning_rate
  max_length
  lora_rank
  lora_alpha
  lora_target
修改 CONFIGURATION -> VGPU
  vgpu_number
  vgpu_memory
  vgpu_cores
Enqueue 到 llm-finetune-vgpu
```

这就是 Portal 出现之前的最小可用 LLM 微调入口。

---

## 13. 可选：切换到 ms-swift 后端

如果暂时只有 ms-swift 镜像，可以使用同一个通用模板切 backend：

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

注意：本 Runbook 的主链路仍然是 LLaMA-Factory。ms-swift 只作为 LLM 微调备用验证路径。

---

## 14. 成功标准

整体验收标准：

```text
1. local PV/PVC 可用
2. llm-finetune-vgpu Volcano Queue 可用
3. ClearML Agent 正常监听 llm-finetune-vgpu
4. vGPU 资源使用 volcano.sh/vgpu-*，没有误用 nvidia.com/gpu
5. LLaMA-Factory 训练任务能进入 K8s Pod
6. Pod 能访问 /data/models 和 /data/datasets
7. llamafactory-cli train 能启动
8. 至少完成 1-2 step
9. 输出文件写到 /data/ai-local/output
10. ClearML Console 能看到日志
11. ClearML Artifacts 能看到 manifest 和训练配置
12. ClearML WebUI Clone 后能修改参数并重新 Enqueue
```

---

## 15. 常见问题排查

### 15.1 PVC 一直 Pending

如果 local PV 使用 `WaitForFirstConsumer`，训练 Pod 创建前 Pending 是正常的。

如果 Pod 创建后仍 Pending，检查：

```bash
kubectl describe pvc ai-data-pvc -n clearml
kubectl describe pv local-ai-data-pv-gpu-node
```

重点看：

```text
nodeAffinity hostname 是否写错
storageClassName 是否一致
PVC namespace 是否是 clearml
Pod nodeSelector 是否指向同一台 GPU 节点
```

### 15.2 Pod Pending

检查：

```bash
kubectl describe pod <pod> -n <namespace>
kubectl describe queue llm-finetune-vgpu
```

常见原因：

```text
VGPU 段没注入
Volcano Queue capability 不够
节点没有 volcano.sh/vgpu-*
local PV 绑定节点和 nodeSelector 不一致
误用了 nvidia.com/gpu
```

### 15.3 找不到模型或数据

进入目标 GPU 节点检查：

```bash
ls -lah /data/ai-local/models
ls -lah /data/ai-local/datasets
```

Pod 内路径应该是：

```text
/data/models
/data/datasets
```

如果 host 上存在但 Pod 内不存在，检查 Agent values 的：

```yaml
volumeMounts:
  - name: ai-data
    mountPath: /data
```

### 15.4 Agent 认证失败

检查：

```bash
kubectl -n clearml get secret clearml-agent-creds
kubectl -n clearml logs deploy/clearml-agent-llm-finetune --tail=200
```

确认：

```text
apiServerUrlReference
fileServerUrlReference
webServerUrlReference
secret 中的 access key / secret key
```

### 15.5 镜像拉取失败

检查：

```bash
kubectl describe pod <pod> -n <namespace>
```

常见原因：

```text
defaultContainerImage 写错
--docker-image 写错
Harbor imagePullSecret 没配
GPU 节点无法访问镜像仓库
```

### 15.6 llamafactory-cli 命令不存在

说明训练镜像里没有预装 LLaMA-Factory。

处理：

```text
换成已安装 LLaMA-Factory 的镜像
或重新构建 harbor.example.com/ai/llamafactory:latest
```

进入镜像验证：

```bash
docker run --rm harbor.example.com/ai/llamafactory:latest \
  bash -lc "which llamafactory-cli && llamafactory-cli version || true"
```

### 15.7 dataset_info.json 或数据格式报错

本通用模板在传入 `--dataset-path` 时，会自动为 LLaMA-Factory 生成临时 `dataset_info.json`。

如果数据格式报错，先确认：

```text
--dataset-format alpaca / sharegpt 是否填对
JSON/JSONL 是否每行合法
Alpaca 格式是否包含 instruction / input / output
ShareGPT 格式是否包含 conversations
```

### 15.8 CUDA OOM

冒烟阶段优先降低资源压力：

```text
--max-length 256
--per-device-train-batch-size 1
--gradient-accumulation-steps 1
--max-steps 1
--lora-rank 4
```

如果仍然 OOM，再提高：

```text
--vgpu-memory
```

或者换更小模型。

### 15.9 Git 仓库克隆失败

现象示例：

```text
error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly: CANCEL
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
clearml_agent: ERROR: Failed cloning repository.
```

这类错误发生在训练真正启动之前。ClearML Agent 会先在训练容器里克隆任务记录的代码仓库，例如：

```text
repository = https://github.com/7Bcoding/clearml.git
entry_point = examples/training_smoke/llm_finetune_universal.py
```

如果训练 Pod 访问 GitHub 不稳定、被代理中断、没有外网权限，任务就会在 clone 阶段失败。

本 Runbook 的通用模板已经默认启用 standalone script 模式：

```text
--store-standalone-script true
```

正常情况下，重新提交新任务后，训练 Pod 不应该再依赖 GitHub clone 这个模板脚本。注意不要直接复跑旧任务，因为旧任务已经记录了 GitHub 仓库地址。请在提交机上重新执行第 10 步命令，生成一个新的 ClearML Task。

如果日志里仍然出现同一个旧 Task ID，例如：

```text
Executing task id [5edda2bdb0ae4e6f9f74b4ff24888ab1]
repository = https://github.com/7Bcoding/clearml.git
```

说明你还在执行旧任务，或者本地提交时复用了旧 Task 元数据。请确认第 10 步命令里包含：

```text
--reuse-last-task-id false
--store-standalone-script true
```

并且是从已经更新后的 `llm_finetune_universal.py` 重新执行提交命令，而不是在 ClearML WebUI 里直接 Enqueue 旧任务。

重新提交后，在 ClearML WebUI 检查：

```text
SCRIPT -> Repository 不应再是 https://github.com/7Bcoding/clearml.git
```

如果新任务仍然记录 GitHub 仓库，检查提交命令是否带了：

```text
--store-standalone-script true
```

以及提交机上的 ClearML SDK 是否支持：

```text
Task.force_store_standalone_script
```

如果 SDK 太老，请升级提交机上的 `clearml` 包，或者改用内网 Git 仓库。

如果你明确关闭了 standalone 模式，或者后续模板变成多文件项目，需要继续走 Git 方式，则先确认训练镜像里能否 clone 仓库。可以在 GPU 节点上执行：

```bash
docker run --rm hiyouga/llamafactory:latest \
  bash -lc "git --version && git clone https://github.com/7Bcoding/clearml.git /tmp/clearml-test"
```

如果这里也失败，说明不是 ClearML 参数问题，而是训练环境访问 GitHub 的问题。

Git 处理方式 A：让 Git 使用 HTTP/1.1。  
当前 `clearml-agent-llm-finetune-vgpu-values.yaml` 已经在训练 Pod 环境变量里加入：

```yaml
- name: GIT_CONFIG_COUNT
  value: "1"
- name: GIT_CONFIG_KEY_0
  value: http.version
- name: GIT_CONFIG_VALUE_0
  value: HTTP/1.1
- name: GIT_TERMINAL_PROMPT
  value: "0"
```

修改 values 后需要重新升级 Agent：

```bash
helm upgrade --install clearml-agent-llm-finetune \
  clearml/clearml-agent \
  -n clearml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-values.yaml \
  -f examples/training_smoke/k8s/clearml-agent-llm-finetune-vgpu-localpv-overlay.local.yaml
```

然后重新 Enqueue 任务。

Git 处理方式 B：清理失败的 VCS 缓存后重试。  
如果上一次 clone 中断，`/root/.clearml/vcs-cache` 里可能留下损坏缓存。当前 Agent values 把 `/root/.clearml` 挂到了节点本地：

```text
/data/clearml-cache/llm-finetune-vgpu/clearml
```

可以在目标 GPU 节点上清理本任务相关缓存：

```bash
rm -rf /data/clearml-cache/llm-finetune-vgpu/clearml/vcs-cache/clearml.git.*
```

然后重新 Enqueue 任务。

Git 处理方式 C：使用内网 Git 镜像。  
生产环境不建议让训练 Pod 依赖公网 GitHub。建议把平台模板代码同步到公司内网 Git，例如：

```text
https://gitlab.example.com/mlops/clearml.git
```

然后提交任务时让 ClearML 记录内网仓库地址。最简单做法是在提交机本地把 Git remote 改成内网地址后重新提交任务：

```bash
git remote set-url origin https://gitlab.example.com/mlops/clearml.git
git push origin master
```

重新执行冒烟提交命令后，在 ClearML WebUI 的任务详情里确认：

```text
SCRIPT -> Repository = https://gitlab.example.com/mlops/clearml.git
```

这样训练 Pod 只需要访问内网 Git，稳定性会高很多。

长期处理方式 D：把通用模板固化为平台镜像或内部包。  
等冒烟稳定后，可以把 `llm_finetune_universal.py` 放进平台训练镜像或内部 Python 包，ClearML 任务只传参数。这更适合正式 Portal 化，也不需要训练 Pod 访问公网 GitHub。

另外，日志里的下面内容不是本次失败的根因：

```text
Python executable with version '3.12' requested by the Task, not found...
using '/opt/conda/bin/python3' (v3.11.11) instead
```

这是 Python 版本回退警告。当前失败的直接原因是 Git clone 失败。

### 15.10 参数被解析成 `[2, 2]`

现象示例：

```text
llm_finetune_universal.py: error: argument --max-steps: invalid int value: '[2, 2]'
```

这通常说明 ClearML Task 的 `CONFIGURATION -> Args` 里某个参数被保存成了列表形式，例如：

```text
max_steps = [2, 2]
```

正常应该是：

```text
max_steps = 2
```

处理方式：

```text
1. 不要继续复跑旧 Task
2. 使用最新的 llm_finetune_universal.py 重新提交新 Task
3. 确认命令里包含 --reuse-last-task-id false
4. 如果在 WebUI 里手动改参数，把 [2, 2] 改成 2
```

当前模板已经兼容这类 ClearML 参数形式。下面这些值会自动取最后一个：

```text
--max-steps "[2, 2]"                 -> 2
--vgpu-memory "[24, 24]"             -> 24
--learning-rate "[1e-5, 1e-5]"       -> 1e-5
```

### 15.11 backend 被解析成 `['llama-factory', 'llama-factory']`

现象示例：

```text
[llm-finetune] backend: ['llama-factory', 'llama-factory']
[llm-finetune] running:
bash -lc '['"'"''"'"', '"'"''"'"']'
bash: line 1: [,: command not found
```

这说明 ClearML 把字符串参数也保存成了列表形式，导致脚本没有识别出：

```text
backend = llama-factory
```

而是误判为 custom backend，最后执行了空的 `custom_command`。

当前模板已经兼容这类参数形式。下面这些值会自动恢复成普通标量：

```text
--backend "['llama-factory', 'llama-factory']"      -> llama-factory
--custom-command "['', '']"                         -> 空字符串
--dataset-path "['/data/a.jsonl', '/data/a.jsonl']" -> /data/a.jsonl
```

处理方式：

```text
1. 使用最新的 llm_finetune_universal.py 重新提交新 Task
2. 不要继续 Enqueue 旧 Task
3. 如果在 WebUI 里手动改参数，把列表形式改成普通字符串
```

### 15.12 NCCL 日志和多进程 DDP

现象示例：

```text
NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
NCCL INFO NET/Plugin: Using internal network plugin.
```

这两行通常不是致命错误。它表示 NCCL 找不到外部网络插件，会 fallback 到 internal network plugin。

真正需要关注的是日志里是否出现多个 rank，例如：

```text
[0] NCCL INFO
[1] NCCL INFO
[2] NCCL INFO
[3] NCCL INFO
```

如果本次只是单卡冒烟，却出现 4 个 rank，说明 LLaMA-Factory 在容器里看到了多张 GPU，于是自动启动了多进程 DDP。冒烟阶段建议强制单卡：

```text
--cuda-visible-devices 0
--nproc-per-node 1
--force-torchrun false
```

当前模板默认就是这个策略。重新提交新任务后，日志里应能看到：

```text
[llm-finetune] CUDA_VISIBLE_DEVICES: 0
[llm-finetune] NPROC_PER_NODE: 1
```

如果仍然出现 4 个 rank，检查 ClearML WebUI 的 `CONFIGURATION -> Args`，确认：

```text
cuda_visible_devices = 0
nproc_per_node = 1
force_torchrun = false
```

如果后续要做真正多卡训练，再把这些参数改成：

```text
--cuda-visible-devices 0,1,2,3
--nproc-per-node 4
--force-torchrun true
--vgpu-number 4
```

---

## 16. 本次冒烟后的下一步

冒烟通过后再做：

```text
1. 把成功的 LLM Task Clone 成团队模板
2. 固化 LLaMA-Factory 为默认 backend
3. 保留 ms-swift / custom 作为高级入口
4. 抽象模型下载、数据上传、路径选择流程
5. 从 host local PV 切换到 NAS/NFS/MinIO CSI
6. Portal 再封装表单：模型、数据、输出目录、训练方式、LoRA 参数、vGPU 参数
```

最终目标是算法同学在 ClearML 或 Portal 页面里只改参数，不需要登录机器手写 YAML。
