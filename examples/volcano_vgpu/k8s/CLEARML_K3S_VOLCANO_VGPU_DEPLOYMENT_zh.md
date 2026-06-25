# ClearML + K3s + Volcano + volcano-vgpu-device-plugin 部署指南

本文档面向当前环境：

- Kubernetes 发行版：K3s
- GPU 节点：`gpu-server-5`、`gpu-server-6`
- 容器运行时：containerd
- GPU 注入方式：NVIDIA Container Toolkit CDI
- 调度器：Volcano
- GPU 切分：`volcano-vgpu-device-plugin` / HAMi-core
- 训练平台：ClearML Server + `clearml-agent-k8s-glue`
- Helm chart：本地 sibling 仓库 `D:\Python\clearml-helm-charts`
- ClearML SDK 示例：当前仓库 `D:\Python\clearml\examples\volcano_vgpu`

当前推荐路线是 **NVIDIA runtime CDI + volcano-vgpu-device-plugin 普通 vGPU 模式**：

```text
K3s/containerd + NVIDIA runtime mode=cdi
    -> CDI spec 每个 GPU 节点本地生成
    -> volcano-vgpu-device-plugin 继续使用 volcano.sh/vgpu-* 资源，不强制开启插件 cdi.enabled
    -> ClearML Agent Pod 模板使用 schedulerName=volcano + runtimeClassName=nvidia
    -> 训练脚本只写 task.connect(..., name="VGPU")
```

这里的 CDI 主要指 **NVIDIA container runtime 用 CDI 注入 GPU 设备和 driver library**。`volcano-vgpu-device-plugin` 自己的 `cdi.enabled=true` 不是必需项；只有当你选择插件的 `cdi-annotations` 路线时才需要开启。之前的 `unresolvable CDI devices management.nvidia.com/gpu=GPU-...` 是运行时 CDI spec 不存在、过期，或 CDI kind 不一致造成的。

---

## 1. 仓库与文件清单

### 1.1 Helm chart 仓库

```powershell
D:\Python\clearml-helm-charts
```

关键文件：

| 文件 | 作用 |
|---|---|
| `charts/clearml/values.yaml` | ClearML Server 默认 values，默认暴露 NodePort `30008/30080/30081` |
| `charts/clearml-agent/values.yaml` | ClearML Agent values，已扩展 `runtimeClassName`、`hostPID`、`vgpuHook`、`basePodTemplate` |
| `charts/clearml-agent/files/vgpu_template_module.py` | 从 Task 的 `CONFIGURATION/VGPU` 读取 `vgpu_number/vgpu_memory/vgpu_cores` |
| `charts/clearml-agent/files/bootstrap_k8s_vgpu_patch.py` | patch k8s-glue，让开源 Agent 也能按任务注入 vGPU limits |
| `examples/volcano-vgpu/custom_values.yaml` | Helm chart 仓库内的单机/单 Pod vGPU Agent values 示例 |

### 1.2 ClearML SDK 示例仓库

```powershell
D:\Python\clearml
```

关键文件：

| 文件 | 作用 |
|---|---|
| `examples/volcano_vgpu/train/single_vgpu_minimal.py` | 单卡训练模板 |
| `examples/volcano_vgpu/train/singlepod_ddp_smoke.py` | 单 Pod 多卡 DDP 示例 |
| `examples/volcano_vgpu/train/multinode_launch_smoke.py` | ClearML `launch_multi_node` 双机/多机示例 |
| `examples/volcano_vgpu/train/multinode_launch_ddp_train.py` | ClearML `launch_multi_node` 双机/多机 DDP 训练 |
| `examples/volcano_vgpu/k8s/agent-values-single-vgpu.yaml` | 单机/单 Pod vGPU Agent values，按 `volcano-queue` 隔离 cache |
| `examples/volcano_vgpu/k8s/agent-values-multinode-launch-vgpu.yaml` | HAMi/vGPU 多机 Agent values |
| `examples/volcano_vgpu/k8s/queue-multinode-vgpu.yaml` | HAMi/vGPU 多机 Volcano Queue |
| `examples/volcano_vgpu/deprecated/k8s/podgroup-clearml-gang-full-2-vgpu.yaml` | 历史 PodGroup 示例，当前不推荐 |
| `examples/volcano_vgpu/deprecated/docs/ENABLE_CDI_full_stack_zh.md` | 旧版 CDI 迁移笔记，本文已合并并修正为当前结论 |

---

## 2. 约定变量

Linux shell：

```bash
export CLEARML_REPO=/path/to/clearml
export HELM_REPO=/path/to/clearml-helm-charts
export CLEARML_NS=clearml
export PLUGIN_NS=kube-system
export CLEARML_HOST=10.10.36.6
```

Windows PowerShell：

```powershell
$env:CLEARML_REPO = "D:\Python\clearml"
$env:HELM_REPO = "D:\Python\clearml-helm-charts"
$env:CLEARML_NS = "clearml"
$env:PLUGIN_NS = "kube-system"
$env:CLEARML_HOST = "10.10.36.6"
```

本文命令主要在 Linux 控制机或 K3s 节点上执行。涉及 `sudo`、`systemctl`、`nvidia-smi`、`nvidia-ctk` 的命令必须在 GPU 节点上执行。

---

## 3. 基础检查

### 3.1 客户端工具

```bash
kubectl version --client
kubectl cluster-info
helm version
jq --version
```

### 3.2 K3s 与节点

```bash
kubectl get nodes -o wide
kubectl get pods -A
```

确认两台 GPU 节点都 Ready：

```bash
kubectl get nodes gpu-server-5 gpu-server-6 -o wide
```

给 GPU 节点打平台 label：

```bash
kubectl label node gpu-server-5 gpu.present=true --overwrite
kubectl label node gpu-server-6 gpu.present=true --overwrite
kubectl get nodes -l gpu.present=true
```

### 3.3 GPU 资源类型

```bash
kubectl get nodes -o json | jq -r '.items[] |
  "\(.metadata.name) nvidia.com/gpu=\(.status.allocatable["nvidia.com/gpu"]) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"])"'
```

当前目标是 HAMi/vGPU 集群：

```text
nvidia.com/gpu 可能为 0
volcano.sh/vgpu-number / volcano.sh/vgpu-memory / volcano.sh/vgpu-cores 有值
```

这种情况下 ClearML 训练 Pod 不申请 `nvidia.com/gpu`，而是申请：

```yaml
volcano.sh/vgpu-number: "1"
volcano.sh/vgpu-memory: "24"
volcano.sh/vgpu-cores: "100"
```

`gpu-memory-factor=1024` 时，`vgpu_memory` 单位按 GiB 约定。4090 整卡可以用 `vgpu_memory=24`、`vgpu_cores=100`，切片训练可以用更小值，比如 `vgpu_memory=4`、`vgpu_cores=30`。

---

## 4. K3s + NVIDIA CDI 配置

这一章是当前最关键部分。目标是让每台 GPU 节点都能：

```bash
sudo nvidia-ctk --debug cdi list
```

看到本机 GPU 的 CDI 设备，并且 Kubernetes 创建容器时不再报：

```text
failed to inject CDI devices: unresolvable CDI devices ...
```

### 4.1 每个 GPU 节点检查 NVIDIA 驱动与 toolkit

在 `gpu-server-5`、`gpu-server-6` 分别执行：

```bash
nvidia-smi -L
which nvidia-ctk
which nvidia-container-runtime
nvidia-container-runtime --version
```

### 4.2 备份现有配置

每个 GPU 节点：

```bash
sudo mkdir -p ~/clearml-gpu-runtime-bak
sudo cp -a /etc/cdi ~/clearml-gpu-runtime-bak/etc-cdi 2>/dev/null || true
sudo cp -a /var/run/cdi ~/clearml-gpu-runtime-bak/var-run-cdi 2>/dev/null || true
sudo cp -a /etc/nvidia-container-runtime ~/clearml-gpu-runtime-bak/etc-nvidia-container-runtime 2>/dev/null || true
sudo cp -a /usr/local/nvidia/toolkit/.config/nvidia-container-runtime ~/clearml-gpu-runtime-bak/toolkit-runtime 2>/dev/null || true
```

### 4.3 生成 CDI spec

每个 GPU 节点：

```bash
sudo mkdir -p /var/run/cdi
sudo rm -f /var/run/cdi/nvidia*.yaml /var/run/cdi/nvidia*.json
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
sudo nvidia-ctk --debug cdi list
```

重点观察输出的 CDI kind。当前环境已经跑通的是：

```text
management.nvidia.com/gpu=GPU-...
```

也可能是 NVIDIA 默认文档里的：

```text
nvidia.com/gpu=GPU-...
```

两者都可以，关键是 **CDI spec、NVIDIA runtime default-kind、device plugin 请求的 kind 必须一致**。

如果你看到两个 kind 混在一起，先定位文件：

```bash
sudo grep -R "kind:\|management.nvidia.com/gpu\|nvidia.com/gpu" /etc/cdi /var/run/cdi 2>/dev/null
```

删除过期 spec 后重新生成：

```bash
sudo rm -f /etc/cdi/nvidia*.yaml /etc/cdi/nvidia*.json /var/run/cdi/nvidia*.yaml /var/run/cdi/nvidia*.json
sudo mkdir -p /var/run/cdi
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
sudo nvidia-ctk --debug cdi list
```

### 4.4 设置 NVIDIA runtime 为 CDI 模式

每个 GPU 节点：

```bash
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi
```

然后把 `default-kind` 设为你在 `cdi list` 里实际看到的 kind。当前环境建议：

```bash
sudo nvidia-ctk config --in-place \
  --set nvidia-container-runtime.modes.cdi.default-kind=management.nvidia.com/gpu
```

如果你的 `cdi list` 输出是 `nvidia.com/gpu=...`，则改用：

```bash
sudo nvidia-ctk config --in-place \
  --set nvidia-container-runtime.modes.cdi.default-kind=nvidia.com/gpu
```

### 4.5 特别处理 `/usr/local/nvidia/toolkit/.config`

你们环境里曾经出现过这个文件反复覆盖 `/etc` 配置：

```text
/usr/local/nvidia/toolkit/.config/nvidia-container-runtime/config.toml
```

必须一起检查：

```bash
sudo grep -R "mode\|default-kind\|management.nvidia\|skip-mode-detection" \
  /etc/nvidia-container-runtime \
  /usr/local/nvidia/toolkit/.config/nvidia-container-runtime 2>/dev/null
```

当前 runtime CDI 目标：

```text
mode = "cdi"
default-kind = "management.nvidia.com/gpu"
```

如果 `/usr/local/nvidia/toolkit/.config/nvidia-container-runtime/config.toml` 没跟上，手工修：

```bash
CFG=/usr/local/nvidia/toolkit/.config/nvidia-container-runtime/config.toml

sudo sed -i -E 's/^([[:space:]]*)mode = ".*"/\1mode = "cdi"/' "$CFG"
sudo sed -i -E 's#^([[:space:]]*)default-kind = ".*"#\1default-kind = "management.nvidia.com/gpu"#' "$CFG"
sudo sed -i -E 's/^([[:space:]]*)skip-mode-detection = .*/\1skip-mode-detection = false/' "$CFG"
```

如果 30 秒后又被改回，查是谁在写：

```bash
kubectl get pods,ds,deploy -A -o wide | egrep -i 'nvidia|gpu|toolkit|driver|hami|volcano'

kubectl get pods -A -o yaml | \
  grep -B40 -A25 -E '/usr/local/nvidia/toolkit|NVIDIA_CONTAINER_RUNTIME_CONFIG|CONTAINERD_CONFIG|management.nvidia|nvidia-container-toolkit'
```

本环境没有使用 NVIDIA GPU Operator，但可能仍有 toolkit DaemonSet 或平台脚本在写该目录。需要从源头改它的 values/YAML，而不是只改宿主机文件。

### 4.6 持久化 CDI

如果节点有 `nvidia-cdi-refresh` 服务，启用它：

```bash
sudo systemctl enable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service 2>/dev/null || true
sudo systemctl status nvidia-cdi-refresh.path nvidia-cdi-refresh.service --no-pager 2>/dev/null || true
```

如果没有该服务，至少确认节点重启后能重新执行：

```bash
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
```

不要在 `/etc/cdi` 和 `/var/run/cdi` 保留互相矛盾的 spec。运行时会扫描多个 CDI spec 目录，stale spec 会制造非常隐蔽的问题。

### 4.7 重启 K3s

按节点角色执行：

```bash
# control-plane/server 节点，例如 gpu-server-6
sudo systemctl restart k3s

# worker/agent 节点，例如 gpu-server-5
sudo systemctl restart k3s-agent
```

### 4.8 RuntimeClass 检查

```bash
kubectl get runtimeclass
```

如果没有 `nvidia`，可以创建：

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
```

应用：

```bash
kubectl apply -f runtimeclass-nvidia.yaml
```

### 4.9 CDI runtime canary

先不测 vGPU，只测 runtime + CDI：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cdi-runtime-canary
  namespace: default
spec:
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-server-6
  containers:
  - name: test
    image: ubuntu:22.04
    command: ["/bin/sh", "-lc"]
    args:
    - |
      nvidia-smi -L
      ldconfig -p | grep -E 'libcuda|libnvidia-ml' || true
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: all
```

验证：

```bash
kubectl delete pod cdi-runtime-canary --ignore-not-found
kubectl apply -f cdi-runtime-canary.yaml
kubectl logs cdi-runtime-canary
kubectl describe pod cdi-runtime-canary
```

`nvidia-smi -L` 成功即表示 CDI runtime 可用。CDI 模式下 `libcuda.so.1` 不一定在 `/usr/local/nvidia/lib64`，用 `ldconfig -p | grep libcuda` 检查更可靠。

---

## 5. 安装 Volcano

如果集群已经有 Volcano：

```bash
kubectl get pods -n volcano-system
kubectl get crd | grep volcano
kubectl get queue
```

如果需要安装，常见 Helm 安装方式：

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm upgrade --install volcano volcano-sh/volcano -n volcano-system --create-namespace
kubectl -n volcano-system rollout status deploy/volcano-scheduler
kubectl -n volcano-system rollout status deploy/volcano-controllers
```

确认 scheduler 启用了 deviceshare/vGPU 相关能力：

```bash
kubectl -n volcano-system get cm volcano-scheduler-configmap -o yaml | grep -A12 -B4 deviceshare
```

如果没有 deviceshare，需要按你安装的 Volcano 版本调整 scheduler config。HAMi/vGPU 资源调度依赖 Volcano scheduler 能识别 `volcano.sh/vgpu-*`。

---

## 6. 安装 volcano-vgpu-device-plugin

### 6.1 安装或升级插件

`volcano-vgpu-device-plugin` 的 `cdi.enabled=true` **不是必需项**。这里有两种路线：

```text
A. 推荐路线，也是当前环境已验证路线：
   NVIDIA runtime 使用 mode=cdi；
   volcano-vgpu-device-plugin 继续用普通 vGPU/envvar 注入模式；
   插件 cdi.enabled=false 或保持默认未开启。

B. 可选路线：
   volcano-vgpu-device-plugin 使用 cdi-annotations；
   这时才需要 cdi.enabled=true，并且要确认 containerd、CDI spec、插件注解三者一致。
```

路线 A 的好处是改动少：让 NVIDIA runtime 负责 CDI 注入 driver library 和设备，插件只负责 vGPU 调度与 HAMi-core 注入。你当前 `gpu-canary-cdi` 已经能跑通，说明这条路可行。

如果已有 release，先备份 values：

```bash
export PLUGIN_NS=kube-system
helm -n $PLUGIN_NS get values volcano-vgpu-device-plugin -a > ~/volcano-vgpu-values.backup.yaml
```

已有 release 时，先看当前 values：

```bash
helm repo add volcano-vgpu-device-plugin https://project-hami.github.io/volcano-vgpu-device-plugin
helm repo update

helm -n $PLUGIN_NS get values volcano-vgpu-device-plugin -a | grep -i -B3 -A6 'cdi\|device_list\|strategy\|memory'
```

如果你采用路线 A，且发现之前已经把插件设成 `cdi.enabled=true`，可以改回 false：

```bash
helm upgrade volcano-vgpu-device-plugin \
  volcano-vgpu-device-plugin/volcano-vgpu-device-plugin \
  -n $PLUGIN_NS \
  --reuse-values \
  --set cdi.enabled=false
```

如果 values 里本来就没有 `cdi.enabled=true`，通常只需要保留现状：

```bash
helm upgrade volcano-vgpu-device-plugin \
  volcano-vgpu-device-plugin/volcano-vgpu-device-plugin \
  -n $PLUGIN_NS \
  --reuse-values
```

如果是首次安装，先查看 chart values，再设置与你集群一致的 `gpu-memory-factor`。路线 A 不需要设置 `cdi.enabled=true`：

```bash
helm show values volcano-vgpu-device-plugin/volcano-vgpu-device-plugin | grep -i -B2 -A5 'cdi\|memory'

helm install volcano-vgpu-device-plugin \
  volcano-vgpu-device-plugin/volcano-vgpu-device-plugin \
  -n $PLUGIN_NS
```

本文档约定：

```text
gpu-memory-factor=1024
```

也就是 ClearML Task 的 `vgpu_memory=24` 表示 24 GiB。

### 6.2 可选：插件 cdi-annotations 路线

只有你明确要让 `volcano-vgpu-device-plugin` 自己写 CDI annotation 时，才开启：

```bash
helm upgrade volcano-vgpu-device-plugin \
  volcano-vgpu-device-plugin/volcano-vgpu-device-plugin \
  -n $PLUGIN_NS \
  --reuse-values \
  --set cdi.enabled=true
```

这条路线要额外确认：

```bash
kubectl -n $PLUGIN_NS get ds volcano-device-plugin -o yaml | \
  egrep -n 'DEVICE_LIST_STRATEGY|cdi-annotations|/var/run/cdi|management.nvidia|nvidia.com/gpu'
```

如果插件产生的是 `nvidia.com/gpu=GPU-...`，而 runtime 的 `default-kind` 是 `management.nvidia.com/gpu`，就会再次触发 `unresolvable CDI devices`。因此路线 B 必须把插件、runtime、节点 CDI spec 的 kind 全部对齐。

### 6.3 验证 DaemonSet

```bash
kubectl -n $PLUGIN_NS get ds,pods | egrep -i 'volcano|vgpu|hami'
kubectl -n $PLUGIN_NS rollout status daemonset -l app.kubernetes.io/name=volcano-vgpu-device-plugin
```

确认插件状态。路线 A 下，不要求看到 `DEVICE_LIST_STRATEGY=cdi-annotations`；路线 B 下，才应该看到 `cdi-annotations`：

```bash
kubectl -n $PLUGIN_NS get ds volcano-device-plugin -o yaml | \
  egrep -n 'DEVICE_LIST_STRATEGY|cdi-annotations|NVIDIA_CDI|/var/run/cdi|management.nvidia|nvidia.com/gpu'
```

如果 DaemonSet 名不是 `volcano-device-plugin`，先查：

```bash
kubectl -n $PLUGIN_NS get ds | egrep -i 'volcano|vgpu|hami'
```

### 6.4 验证节点 vGPU 资源

```bash
kubectl describe node gpu-server-5 | grep -A6 -B2 volcano.sh
kubectl describe node gpu-server-6 | grep -A6 -B2 volcano.sh
```

或：

```bash
kubectl get nodes -o json | jq -r '.items[] |
  "\(.metadata.name) vgpu-number=\(.status.allocatable["volcano.sh/vgpu-number"]) vgpu-memory=\(.status.allocatable["volcano.sh/vgpu-memory"]) vgpu-cores=\(.status.allocatable["volcano.sh/vgpu-cores"])"'
```

---

## 7. vGPU Pod canary

不要用 `nodeName`。`nodeName` 会绕过 Volcano scheduler，容易在 Allocate 阶段报：

```text
UnexpectedAdmissionError: Allocate failed ... cannot get valid pod
```

使用 `nodeSelector` 约束到指定节点，同时保留：

```yaml
schedulerName: volcano
runtimeClassName: nvidia
```

示例：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-canary-cdi
  namespace: default
  annotations:
    volcano.sh/vgpu-mode: "hami-core"
spec:
  schedulerName: volcano
  runtimeClassName: nvidia
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: gpu-server-6
  containers:
  - name: cuda
    image: nvidia/cuda:12.4.1-base-ubuntu22.04
    command: ["/bin/sh", "-lc"]
    args:
    - |
      nvidia-smi -L
      ldconfig -p | grep -E 'libcuda|libnvidia-ml'
      sleep 10
    resources:
      limits:
        volcano.sh/vgpu-number: "1"
        volcano.sh/vgpu-memory: "24"
        volcano.sh/vgpu-cores: "100"
```

验证：

```bash
kubectl delete pod gpu-canary-cdi --force --grace-period=0 --ignore-not-found
kubectl apply -f gpu-canary-cdi.yaml
kubectl logs gpu-canary-cdi
kubectl describe pod gpu-canary-cdi
```

成功特征：

```text
GPU 0: ...
libcuda.so.1 => /lib/x86_64-linux-gnu/libcuda.so.1
HAMI-core ...
```

如果 `ubuntu` canary 能 `nvidia-smi`，但 `nvidia/cuda` vGPU canary 失败，优先查 `volcano-vgpu-device-plugin` 的 CDI 模式和 Allocate 日志，而不是 ClearML。

---

## 8. 部署 ClearML Server

### 8.1 安装 Server

在 Helm chart 仓库执行：

```bash
cd "$HELM_REPO"
kubectl create namespace clearml --dry-run=client -o yaml | kubectl apply -f -
helm dependency build charts/clearml
helm upgrade --install clearml charts/clearml -n clearml
```

默认 NodePort：

| 服务 | URL |
|---|---|
| API | `http://<CLEARML_HOST>:30008` |
| Web | `http://<CLEARML_HOST>:30080` |
| Files | `http://<CLEARML_HOST>:30081` |

当前 values 里使用：

```text
http://10.10.36.6:30008
http://10.10.36.6:30080
http://10.10.36.6:30081
```

### 8.2 验证 Server

```bash
kubectl -n clearml get pods,svc
kubectl -n clearml get events --sort-by=.lastTimestamp | tail -80
curl -sSf http://10.10.36.6:30008/debug.ping
```

如果 MongoDB readiness probe 偶尔 5s timeout，但服务最终 Ready，可以先观察；如果长期 Unready，再查 PVC、IO、内存。

### 8.3 创建 Agent 凭证 Secret

在 ClearML WebUI：

```text
Settings -> Workspace -> Create new credentials
```

然后创建 K8s Secret。不要把真实 key 写进 Git：

```bash
kubectl -n clearml create secret generic clearml-agent-creds \
  --from-literal=agentk8sglue_key='<ACCESS_KEY>' \
  --from-literal=agentk8sglue_secret='<SECRET_KEY>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Helm chart 示例 `custom_values.yaml`、本文档新增的单机 values、以及多机 values 都引用：

```yaml
clearml:
  existingAgentk8sglueSecret: "clearml-agent-creds"
```

### 8.4 上线环境：Agent 环境隔离与缓存策略

ClearML Agent 会按 Task 记录的代码、Python 版本和依赖重建运行环境。缓存能显著减少重复安装时间，但 **缓存不是用户隔离边界**。上线后要把“运行隔离”和“缓存复用”分开设计：

```text
运行隔离：Kubernetes Pod / namespace / queue / Agent / venv
缓存复用：pip download cache / vcs cache / venv cache
```

推荐分层如下：

| 场景 | 建议 |
|---|---|
| 内部可信团队 | 可以共享同一队列的 pip cache 和 venv cache；要求任务不要在运行中修改 site-packages |
| 多项目组 | 每个项目组独立 ClearML queue + Agent release + cache 目录/PVC |
| 多租户上线 | 每个租户独立 namespace、Agent、Secret、queue、cache PVC；只共享基础镜像或只读镜像仓库 |
| 强安全隔离 | 不共享 venv cache；最多共享只读 wheelhouse 或 pip download cache |

当前示例采用 **按队列隔离 cache 目录** 的折中方案：

```text
/data/clearml-cache/volcano-queue/clearml
/data/clearml-cache/volcano-queue/pip
/data/clearml-cache/multinode-full-gpu/clearml
/data/clearml-cache/multinode-full-gpu/pip
```

这样 `volcano-queue` 和 `multinode-full-gpu` 不会共用同一个可写 venv cache。后续如果要按项目组或租户隔离，把 queue 名和路径替换成项目/租户名即可，例如：

```text
clearml-agent-team-a -> queue team-a-gpu -> /data/clearml-cache/team-a/...
clearml-agent-team-b -> queue team-b-gpu -> /data/clearml-cache/team-b/...
```

Task Pod 内建议统一注入这些环境变量：

```yaml
- name: PIP_CACHE_DIR
  value: /root/.cache/pip
- name: CLEARML_AGENT_VENV_CACHE_PATH
  value: /root/.clearml/venvs-cache
- name: CLEARML_AGENT__AGENT__PACKAGE_MANAGER__SYSTEM_SITE_PACKAGES
  value: "false"
- name: CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__ENABLED
  value: "true"
- name: CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__PATH
  value: '"/root/.cache/pip"'
- name: CLEARML_AGENT__AGENT__VCS_CACHE__ENABLED
  value: "true"
- name: CLEARML_AGENT__AGENT__VCS_CACHE__PATH
  value: '"/root/.clearml/vcs-cache"'
```

说明：

- `CLEARML_AGENT_VENV_CACHE_PATH` 启用 venv 缓存，但不同依赖签名仍会生成不同 cache entry。
- `PIP_CACHE_DIR` / `agent.pip_download_cache` 只缓存下载包，不等于共用 Python 环境，风险比共享 venv cache 低。
- `system_site_packages=false` 避免 Task venv 继承镜像或系统 Python 包，隔离性更好。
- 不要让训练脚本在运行过程中 `pip install` / `pip uninstall` / 修改 `site-packages`。依赖应该写进 `requirements-remote.txt` 或 Task packages。
- 如果用户能提交任意代码，不要跨租户共享可写 hostPath venv cache；改用每租户 PVC 或禁用 venv cache。

本文档提供两份可直接使用的 values：

```text
$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-single-vgpu.yaml
$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-multinode-launch-vgpu.yaml
```

---

## 9. 部署单机 vGPU ClearML Agent

单机队列：

```text
volcano-queue
```

values：

```text
D:\Python\clearml\examples\volcano_vgpu\k8s\agent-values-single-vgpu.yaml
```

关键配置：

```yaml
agentk8sglue:
  queue: volcano-queue
  defaultContainerImage: nvidia/cuda:12.4.1-runtime-ubuntu22.04
  vgpuHook:
    enabled: true
    section: VGPU
    gpuMemoryFactor: 1024
    mode: hami-core
  basePodTemplate:
    schedulerName: volcano
    runtimeClassName: nvidia
    hostPID: true
    annotations:
      volcano.sh/vgpu-mode: "hami-core"
    resources:
      limits:
        volcano.sh/vgpu-number: 1
        volcano.sh/vgpu-memory: 2
        volcano.sh/vgpu-cores: 30
      requests:
        cpu: "4"
        memory: 16Gi
    env:
      - name: PIP_CACHE_DIR
        value: /root/.cache/pip
      - name: CLEARML_AGENT_VENV_CACHE_PATH
        value: /root/.clearml/venvs-cache
      - name: CLEARML_AGENT__AGENT__PACKAGE_MANAGER__SYSTEM_SITE_PACKAGES
        value: "false"
      - name: CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__ENABLED
        value: "true"
      - name: CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__PATH
        value: '"/root/.cache/pip"'
      - name: CLEARML_AGENT__AGENT__VCS_CACHE__ENABLED
        value: "true"
      - name: CLEARML_AGENT__AGENT__VCS_CACHE__PATH
        value: '"/root/.clearml/vcs-cache"'
      - name: CLEARML_AGENT__AGENT__VENVS_CACHE__MAX_ENTRIES
        value: "20"
      - name: CLEARML_AGENT__AGENT__VENVS_CACHE__FREE_SPACE_THRESHOLD_GB
        value: "10"
    volumes:
      - name: clearml-cache
        hostPath:
          path: /data/clearml-cache/volcano-queue/clearml
          type: DirectoryOrCreate
      - name: pip-cache
        hostPath:
          path: /data/clearml-cache/volcano-queue/pip
          type: DirectoryOrCreate
    volumeMounts:
      - name: clearml-cache
        mountPath: /root/.clearml
      - name: pip-cache
        mountPath: /root/.cache/pip
```

安装：

```bash
cd "$HELM_REPO"

helm upgrade --install clearml-agent charts/clearml-agent -n clearml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-single-vgpu.yaml"
```

验证：

```bash
kubectl -n clearml rollout status deploy/clearml-agent
kubectl -n clearml logs deploy/clearml-agent --tail=120
```

日志中应能看到：

```text
[vgpu-hook] K8sIntegration._kubectl_apply patched
starting k8s glue ... --queue volcano-queue
```

---

## 10. 部署双机/多机 vGPU ClearML Agent

多机队列：

```text
multinode-full-gpu
```

虽然队列名带 `full-gpu`，在 HAMi/vGPU 集群里它申请的是 `volcano.sh/vgpu-*`，不是 `nvidia.com/gpu`。

### 10.1 创建 Volcano Queue

```bash
kubectl apply -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/queue-multinode-vgpu.yaml"
kubectl get queue multinode-full-gpu
kubectl describe queue multinode-full-gpu
```

应看到：

```yaml
capability:
  cpu: "32"
  memory: 128Gi
  volcano.sh/vgpu-number: "8"
  volcano.sh/vgpu-memory: "92"
```

如果你看到 `nvidia.com/gpu`，说明误用了 `volcano_queue_multinode_full_gpu.example.yaml`，在 vGPU-only 集群会导致 PodGroup `overused nvidia.com/gpu`。

### 10.2 创建 ClearML Queue

因为当前 Agent 镜像 `allegroai/clearml-agent-k8s-base:1.24-21` 不建议依赖 `--create-queue`，手动创建队列。

WebUI：

```text
Workers & Queues -> + NEW QUEUE -> multinode-full-gpu
```

或 Python：

```bash
python - <<'PY'
from clearml.backend_api.session.client import APIClient
api = APIClient()
name = "multinode-full-gpu"
try:
    api.queues.create(name=name)
    print("created", name)
except Exception as exc:
    print("queue may already exist:", exc)
print([q.name for q in api.queues.get_all()])
PY
```

### 10.3 安装 multinode Agent

values：

```text
D:\Python\clearml\examples\volcano_vgpu\k8s\agent-values-multinode-launch-vgpu.yaml
```

关键配置：

```yaml
agentk8sglue:
  queue: multinode-full-gpu
  createQueueIfNotExists: false
  defaultContainerImage: nvidia/cuda:12.4.1-runtime-ubuntu22.04
  vgpuHook:
    enabled: true
    section: VGPU
    gpuMemoryFactor: 1024
    mode: hami-core
  basePodTemplate:
    schedulerName: volcano
    runtimeClassName: nvidia
    annotations:
      volcano.sh/vgpu-mode: "hami-core"
      volcano.sh/queue-name: multinode-full-gpu
    resources:
      requests:
        cpu: "8"
        memory: 32Gi
    env:
      - name: CLEARML_MULTI_NODE_MASTER_DEF_ADDR
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
      - name: NCCL_DEBUG
        value: "INFO"
      - name: NCCL_IB_DISABLE
        value: "1"
      - name: NCCL_SOCKET_IFNAME
        value: "eth0"
      - name: PIP_CACHE_DIR
        value: /root/.cache/pip
      - name: CLEARML_AGENT_VENV_CACHE_PATH
        value: /root/.clearml/venvs-cache
      - name: CLEARML_AGENT__AGENT__PACKAGE_MANAGER__SYSTEM_SITE_PACKAGES
        value: "false"
      - name: CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__ENABLED
        value: "true"
      - name: CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__PATH
        value: '"/root/.cache/pip"'
      - name: CLEARML_AGENT__AGENT__VCS_CACHE__ENABLED
        value: "true"
      - name: CLEARML_AGENT__AGENT__VCS_CACHE__PATH
        value: '"/root/.clearml/vcs-cache"'
      - name: CLEARML_AGENT__AGENT__VENVS_CACHE__MAX_ENTRIES
        value: "20"
      - name: CLEARML_AGENT__AGENT__VENVS_CACHE__FREE_SPACE_THRESHOLD_GB
        value: "10"
    nodeSelector:
      gpu.present: "true"
    volumes:
      - name: clearml-cache
        hostPath:
          path: /data/clearml-cache/multinode-full-gpu/clearml
          type: DirectoryOrCreate
      - name: pip-cache
        hostPath:
          path: /data/clearml-cache/multinode-full-gpu/pip
          type: DirectoryOrCreate
    volumeMounts:
      - name: clearml-cache
        mountPath: /root/.clearml
      - name: pip-cache
        mountPath: /root/.cache/pip
```

部署：

```bash
cd "$HELM_REPO"

helm upgrade --install clearml-agent-multinode charts/clearml-agent -n clearml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-multinode-launch-vgpu.yaml"
```

验证：

```bash
kubectl -n clearml rollout status deploy/clearml-agent-multinode
kubectl -n clearml logs deploy/clearml-agent-multinode --tail=120
```

### 10.4 不要把单机 values 和多机 values 随意叠加

单机 values：

```text
$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-single-vgpu.yaml
```

多机 values：

```text
$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-multinode-launch-vgpu.yaml
```

两套可以分别部署成两个 release：

```text
clearml-agent            -> queue volcano-queue
clearml-agent-multinode  -> queue multinode-full-gpu
```

不要用一个 `helm upgrade` 同时叠加两套 values，避免深合并把不该带的字段带进去。

---

## 11. 训练脚本侧约定

算法脚本只需要标准 ClearML API，再加一个平台约定段：

```python
from clearml import Task

Task.force_requirements_env_freeze(
    force=True,
    requirements_file="requirements-remote.txt",
)

task = Task.init(project_name="volcano-vgpu", task_name="my-training")

task.connect(
    {"vgpu_number": 1, "vgpu_memory": 4, "vgpu_cores": 30},
    name="VGPU",
)

task.execute_remotely(queue_name="volcano-queue")
```

`vgpuHook` 会读取 `VGPU` 段并注入到 Pod：

```yaml
resources:
  limits:
    volcano.sh/vgpu-number: "1"
    volcano.sh/vgpu-memory: "4"
    volcano.sh/vgpu-cores: "30"
```

多机整卡示例：

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"

python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --vgpu-number 1 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

切片双机示例：

```bash
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --vgpu-number 1 \
  --vgpu-memory 4 \
  --vgpu-cores 30
```

---

## 12. 验证顺序

不要直接上完整双机训练。按下面顺序验证，每一层通过再进入下一层。

### 12.1 节点 CDI

每个 GPU 节点：

```bash
nvidia-smi -L
sudo nvidia-ctk --debug cdi list
sudo grep -R "mode\|default-kind" \
  /etc/nvidia-container-runtime \
  /usr/local/nvidia/toolkit/.config/nvidia-container-runtime 2>/dev/null
```

### 12.2 K8s runtime CDI

```bash
kubectl apply -f cdi-runtime-canary.yaml
kubectl logs cdi-runtime-canary
```

预期：

```text
GPU 0: ...
GPU 1: ...
```

### 12.3 Volcano vGPU

```bash
kubectl apply -f gpu-canary-cdi.yaml
kubectl logs gpu-canary-cdi
kubectl describe pod gpu-canary-cdi
```

预期：

```text
nvidia-smi -L 成功
ldconfig 能看到 libcuda.so.1
没有 unresolvable CDI devices
```

### 12.4 ClearML 单机任务

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python train/single_vgpu_minimal.py
```

检查：

```bash
kubectl -n clearml get pods -o wide --sort-by=.metadata.creationTimestamp
kubectl -n clearml logs <task-pod-name> --tail=80
```

### 12.5 ClearML 多机任务

```bash
python train/multinode_launch_smoke.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

训练模板验证：

```bash
python train/multinode_launch_ddp_train.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --epochs 5 \
  --batch-size 128 \
  --vgpu-memory 24 \
  --vgpu-cores 100
```

检查：

```bash
kubectl -n clearml get pods -o wide --sort-by=.metadata.creationTimestamp
kubectl -n clearml get events --sort-by=.lastTimestamp | tail -120
```

ClearML WebUI 中看：

```text
SCALARS -> smoke/allreduce_sum = 3.0
CONFIGURATION -> allreduce_ok = 1
```

---

## 13. 历史 PodGroup / Volcano Job 可选方案

### 13.1 PodGroup gang（历史方案，不推荐）

旧的 PodGroup 方案已归档到 `deprecated/k8s/`，当前不再作为主线部署路径。需要对照历史配置时只查看文件，不建议在当前主线中 apply：

```bash
less "$CLEARML_REPO/examples/volcano_vgpu/deprecated/k8s/podgroup-clearml-gang-full-2-vgpu.yaml"
```

注意：

- vGPU-only 集群必须用 `volcano.sh/vgpu-number` 作为 `minResources`
- 不要用 `nvidia.com/gpu` 版 PodGroup
- Pod 模板还需要 `scheduling.k8s.io/group-name` 注解

### 13.2 Volcano Job

如果想绕开 ClearML k8s-glue 的多 Task 模式，让 Volcano 原生创建多 Pod Job：

```bash
cd "$CLEARML_REPO/examples/volcano_vgpu"
python submit/submit_volcano_job.py --num-nodes 2 --apply
```

在 HAMi/vGPU 集群里，脚本应使用 vGPU 模式渲染 `volcano.sh/vgpu-*`。如果手工改 YAML，不要保留 `nvidia.com/gpu`。

---

## 14. 常见问题排查

### 14.1 `unresolvable CDI devices management.nvidia.com/gpu=GPU-...`

说明 runtime/pod 请求了 CDI，但当前节点 CDI spec 里没有对应设备。

排查：

```bash
kubectl get pod <pod> -n <ns> -o yaml | egrep -i 'cdi|management.nvidia|nvidia.com/gpu|NVIDIA_VISIBLE_DEVICES|runtimeClassName'
sudo nvidia-ctk --debug cdi list
sudo grep -R "mode\|default-kind\|management.nvidia\|nvidia.com/gpu" \
  /etc/nvidia-container-runtime \
  /usr/local/nvidia/toolkit/.config/nvidia-container-runtime \
  /etc/cdi /var/run/cdi 2>/dev/null
```

修复原则：

```text
pod 请求的 CDI kind = runtime default-kind = cdi list 输出的 kind
```

当前环境建议统一：

```text
management.nvidia.com/gpu
```

### 14.2 `libcuda.so.1: cannot open shared object file`

legacy 模式下常见，原因是 HAMi-core 注入了 `LD_PRELOAD`，但 NVIDIA driver library 没被 runtime 挂入或没在搜索路径内。

当前推荐解法是 runtime CDI。CDI 通过 NVIDIA runtime 层注入设备、driver library、binary，避免每个 ClearML pod 手写 `LD_LIBRARY_PATH`。这不要求 `volcano-vgpu-device-plugin` 必须开启 `cdi.enabled=true`。

验证：

```bash
ldconfig -p | grep -E 'libcuda|libnvidia-ml'
```

不要只查：

```bash
/usr/local/nvidia/lib64/libcuda.so.1
```

CDI 模式下它可能在：

```text
/lib/x86_64-linux-gnu/libcuda.so.1
```

### 14.3 `nvidia-smi: not found`

先确认 CDI runtime canary 是否成功。如果 `nvidia-smi -L` 在 `ubuntu:22.04` canary 中成功，说明 runtime CDI 可用。

如果某个 CUDA 镜像里找不到：

```bash
command -v nvidia-smi || true
ls -l /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi 2>/dev/null || true
```

不要只因为 `nvidia-smi` PATH 问题判断 GPU 不可用。训练框架最终看的是 CUDA driver/library。

### 14.4 `Allocate failed ... cannot get valid pod`

通常是测试 Pod 用了：

```yaml
nodeName: gpu-server-6
```

这会绕过 Volcano scheduler。改成：

```yaml
schedulerName: volcano
nodeSelector:
  kubernetes.io/hostname: gpu-server-6
```

### 14.5 PodGroup `overused nvidia.com/gpu`

vGPU-only 集群没有 `nvidia.com/gpu`，但队列或 PodGroup 还在用它。

检查：

```bash
kubectl describe queue multinode-full-gpu
```

修复：

```bash
kubectl apply -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/queue-multinode-vgpu.yaml"
```

### 14.6 Agent CrashLoop: `unrecognized arguments: --create-queue`

当前 `allegroai/clearml-agent-k8s-base:1.24-21` 不应依赖 `createQueueIfNotExists=true`。

修复：

```bash
helm upgrade clearml-agent-multinode "$HELM_REPO/charts/clearml-agent" -n clearml \
  -f "$CLEARML_REPO/examples/volcano_vgpu/k8s/agent-values-multinode-launch-vgpu.yaml" \
  --set agentk8sglue.createQueueIfNotExists=false
```

然后在 ClearML WebUI 或 API 手动创建队列。

### 14.7 ClearML Pod 没有 vGPU limits

检查 Agent 日志：

```bash
kubectl -n clearml logs deploy/clearml-agent --tail=160 | grep -i vgpu
kubectl -n clearml logs deploy/clearml-agent-multinode --tail=160 | grep -i vgpu
```

检查 Task 是否有 `VGPU` 段：

```python
task.connect({"vgpu_number": 1, "vgpu_memory": 4, "vgpu_cores": 30}, name="VGPU")
```

检查生成的 Pod：

```bash
kubectl -n clearml get pod <task-pod> -o yaml | egrep -i 'volcano.sh/vgpu|runtimeClassName|schedulerName'
```

### 14.8 NCCL 跨节点失败

当前 values 里默认：

```yaml
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```

如果节点实际网卡不是 `eth0`，修改多机 values：

```yaml
- name: NCCL_SOCKET_IFNAME
  value: "<实际网卡名>"
```

查看 Pod 内网卡：

```bash
kubectl -n clearml exec <pod> -- ip addr
kubectl -n clearml exec <pod> -- ip route
```

---

## 15. 日常维护

### 15.1 驱动或 GPU 拓扑变化后

每个 GPU 节点重新生成 CDI：

```bash
sudo rm -f /var/run/cdi/nvidia*.yaml /var/run/cdi/nvidia*.json
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
sudo nvidia-ctk --debug cdi list
```

然后重启节点 K3s 服务与插件：

```bash
sudo systemctl restart k3s-agent   # worker
sudo systemctl restart k3s         # server

kubectl -n kube-system rollout restart ds/volcano-device-plugin
```

### 15.2 升级 Agent values 后

```bash
helm -n clearml get values clearml-agent -a > ~/clearml-agent-values.backup.yaml
helm -n clearml get values clearml-agent-multinode -a > ~/clearml-agent-multinode-values.backup.yaml

kubectl -n clearml rollout restart deploy/clearml-agent
kubectl -n clearml rollout restart deploy/clearml-agent-multinode
```

### 15.3 统一状态快照

```bash
kubectl get nodes -o wide
kubectl get pods -A | egrep -i 'clearml|volcano|vgpu|hami|nvidia'
kubectl get queue
kubectl get runtimeclass
kubectl -n clearml get deploy,po,svc
kubectl -n kube-system get ds | egrep -i 'volcano|vgpu|hami|nvidia'
kubectl -n volcano-system get pods
kubectl get events -A --sort-by=.lastTimestamp | tail -120
```

---

## 16. 回滚到 legacy 模式

仅在 CDI 无法稳定时使用。不要只删 CDI 文件，必须整栈回滚。

插件回滚仅针对你启用过插件 `cdi.enabled=true` 的情况；如果一直走路线 A，插件侧通常无需额外回滚：

```bash
helm upgrade volcano-vgpu-device-plugin \
  volcano-vgpu-device-plugin/volcano-vgpu-device-plugin \
  -n $PLUGIN_NS \
  --reuse-values \
  --set cdi.enabled=false
```

每个 GPU 节点：

```bash
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=legacy

CFG=/usr/local/nvidia/toolkit/.config/nvidia-container-runtime/config.toml
[ -f "$CFG" ] && sudo sed -i -E 's/^([[:space:]]*)mode = ".*"/\1mode = "legacy"/' "$CFG"

sudo rm -f /etc/cdi/nvidia*.yaml /etc/cdi/nvidia*.json /var/run/cdi/nvidia*.yaml /var/run/cdi/nvidia*.json

sudo systemctl restart k3s-agent   # worker
sudo systemctl restart k3s         # server
```

ClearML Agent：

```bash
kubectl -n clearml rollout restart deploy/clearml-agent
kubectl -n clearml rollout restart deploy/clearml-agent-multinode
```

legacy 模式若再次出现 `libcuda.so.1`，不要让算法脚本逐个补 env。应通过 ClearML pod template、Kyverno/MutatingWebhook、或修插件统一处理。当前更推荐继续用 CDI。

---

## 17. 最终验收清单

```text
[ ] gpu-server-5 / gpu-server-6: nvidia-smi -L 正常
[ ] gpu-server-5 / gpu-server-6: nvidia-ctk --debug cdi list 能看到本机 GPU UUID
[ ] runtime mode=cdi，default-kind 与 cdi list 一致
[ ] /usr/local/nvidia/toolkit/.config 没有把 mode 改回 legacy 或错误 kind
[ ] K3s 已按节点角色重启
[ ] RuntimeClass nvidia 存在
[ ] volcano-scheduler / volcano-controllers Running
[ ] volcano-vgpu-device-plugin Running；路线 A 下未启用 cdi-annotations，路线 B 下 cdi.enabled=true 且 kind 对齐
[ ] 节点 allocatable 有 volcano.sh/vgpu-*
[ ] cdi-runtime-canary 成功
[ ] gpu-canary-cdi 成功，无 unresolvable CDI events
[ ] ClearML Server API/Web/Files 可访问
[ ] clearml-agent 监听 volcano-queue
[ ] clearml-agent-multinode 监听 multinode-full-gpu
[ ] 单机/多机 Agent 使用不同 cache 路径或 PVC
[ ] Task Pod 内 `CLEARML_AGENT_VENV_CACHE_PATH=/root/.clearml/venvs-cache`
[ ] Task Pod 内 `PIP_CACHE_DIR=/root/.cache/pip`
[ ] Task Pod 内未启用 `system_site_packages`
[ ] train/single_vgpu_minimal.py 能跑通
[ ] train/multinode_launch_smoke.py 双机 allreduce_sum=3.0
[ ] train/multinode_launch_ddp_train.py 双机 train/loss 与 train/accuracy 正常上报
```

---

## 18. 参考

- NVIDIA CDI 文档：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html>
- K3s advanced runtime 文档：<https://docs.k3s.io/advanced>
- Volcano GPU virtualization 文档：<https://volcano.sh/en/docs/gpu_virtualization/>
- volcano-vgpu-device-plugin：<https://github.com/Project-HAMi/volcano-vgpu-device-plugin>
- ClearML Agent 文档：<https://clear.ml/docs/latest/docs/clearml_agent/>
- ClearML Agent 环境变量：<https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_env_var/>
- ClearML 配置文件参考：<https://clear.ml/docs/latest/docs/configs/clearml_conf/>
- ClearML Helm chart：`D:\Python\clearml-helm-charts`
- 当前 ClearML vGPU 示例：`D:\Python\clearml\examples\volcano_vgpu`
