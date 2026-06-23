# 全栈启用 CDI（K3s + NVIDIA + volcano-vgpu-device-plugin）

面向与 **未来 K3s / NVIDIA Container Toolkit 默认路线一致** 的集群迁移。  
**必须整栈同步**：两台 GPU 节点 + plugin `cdi.enabled=true` + 同一 `runtimeClassName`，禁止「半 CDI」。

与 HAMi / ClearML 关系：

- `volcano.sh/vgpu-*` + `hami-core` **不变**（vgpuHook、`task.connect(VGPU)` 不变）
- CDI 管 **物理 GPU 如何进容器**（`management.nvidia.com/gpu=GPU-<uuid>`）
- plugin 非 CDI 版（`cdi.enabled=false`）与节点 CDI 混搭会导致 gpu-5 行、gpu-6 挂

---

## 0. 前置

- 两台 GPU 节点：`gpu-server-5`（agent）、`gpu-server-6`（常为 server+worker）
- `gpu-memory-factor=1024`（GiB）与现网 plugin / ClearML 一致
- 备份：

```bash
# 每台 GPU 节点
sudo mkdir -p ~/cdi-migrate-bak
sudo cp -a /etc/cdi ~/cdi-migrate-bak/ 2>/dev/null || true
sudo cp /etc/nvidia-container-runtime/config.toml ~/cdi-migrate-bak/
helm get values volcano-vgpu-device-plugin -n <PLUGIN_NS> -a > ~/cdi-migrate-bak/vgpu-plugin-values.yaml
```

---

## 1. 两台 GPU 节点：CDI + mode=cdi

在 **gpu-server-5** 和 **gpu-server-6** 各执行：

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 必须为 UUID 格式（不能只有 "0","1","2"）
sudo grep -c "management.nvidia.com" /etc/cdi/nvidia.yaml
nvidia-smi -L

sudo nvidia-ctk config --set mode=cdi
grep '^mode' /etc/nvidia-container-runtime/config.toml
# 期望: mode = "cdi"
```

**持久化 nvidia runtime（可选，推荐）** — 路径按节点角色选 `agent` 或 `server` 目录：

```bash
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd
# control-plane 若用 server 路径：
# sudo mkdir -p /var/lib/rancher/k3s/etc/containerd

sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl <<'EOF'
{{ template "base" . }}
EOF
```

> 全 CDI 栈下以 K3s 自动发现的 `nvidia` / `nvidia-cdi` 为准；若 containerd 启动失败，按 [K3s #13678](https://github.com/k3s-io/k3s/issues/13678) 提供完整 tmpl，勿重复 append 冲突段。

**重启 K3s（角色别搞错）：**

```bash
# gpu-server-5（worker agent）
sudo systemctl restart k3s-agent

# gpu-server-6（control-plane）
sudo systemctl restart k3s
```

**验证两台对称：**

```bash
ls -la /etc/cdi/
grep '^mode' /etc/nvidia-container-runtime/config.toml
for f in /var/lib/rancher/k3s/agent/etc/containerd/config.toml \
         /var/lib/rancher/k3s/etc/containerd/config.toml; do
  [ -f "$f" ] && echo "=== $f ===" && grep -E 'cdi|nvidia' "$f" | head -12
done
```

---

## 2. volcano-vgpu-device-plugin：cdi.enabled=true

**保留现有 vgpu 定制**（`gpu-memory-factor`、自改字段等）：

```bash
export PLUGIN_NS=kube-system   # 按实际修改

helm get values volcano-vgpu-device-plugin -n $PLUGIN_NS -a

helm upgrade volcano-vgpu-device-plugin volcano-vgpu-device-plugin/volcano-vgpu-device-plugin \
  -n $PLUGIN_NS \
  --reuse-values \
  --set cdi.enabled=true

kubectl rollout status daemonset -n $PLUGIN_NS -l app.kubernetes.io/name=volcano-vgpu-device-plugin
kubectl get pods -n $PLUGIN_NS | grep -i vgpu
```

查 chart 是否还要求其它 CDI 项：

```bash
helm show values volcano-vgpu-device-plugin/volcano-vgpu-device-plugin | grep -i cdi -B1 -A3
```

确认节点资源仍为 `volcano.sh/vgpu-*`：

```bash
kubectl describe node gpu-server-5 | grep -A3 volcano.sh
kubectl describe node gpu-server-6 | grep -A3 volcano.sh
```

---

## 3. runtimeClassName（ClearML / 训练 Pod）

CDI 全栈下通常保留：

```yaml
runtimeClassName: nvidia
```

若 chart / K3s 文档要求 `nvidia-cdi`，则 **单机与多机 Agent 模板统一改**，不要 5 用 nvidia、6 用别的。

**ClearML Helm：**

```bash
# 单机 volcano-queue
helm upgrade clearml-agent charts/clearml-agent -n clearml \
  -f examples/volcano-vgpu/custom_values.yaml

# 多机（单独 values；可与单机同样保留 runtimeClassName: nvidia）
helm upgrade clearml-agent-multinode charts/clearml-agent -n clearml \
  -f examples/volcano_vgpu/k8s/values-multinode-vgpu.standalone.example.yaml

kubectl scale deploy clearml-agent-multinode -n clearml --replicas=1
kubectl rollout restart deploy/clearml-agent -n clearml
kubectl rollout restart deploy/clearml-agent-multinode -n clearml
```

`basePodTemplate` 仍须：

- `schedulerName: volcano`
- `volcano.sh/vgpu-mode: hami-core`
- **不要** `nvidia.com/gpu`
- vGPU 由 `vgpuHook` + Task `VGPU` 段注入

---

## 4. 节点冒烟（再上 ClearML）

每台 GPU 节点各测一次（改 `nodeSelector` 与 pod 名）：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cdi-smoke-gpu5
  namespace: clearml
spec:
  restartPolicy: Never
  runtimeClassName: nvidia
  nodeSelector:
    kubernetes.io/hostname: gpu-server-5
  containers:
    - name: t
      image: nvidia/cuda:12.4.1-runtime-ubuntu22.04
      command: ["nvidia-smi", "-L"]
```

```bash
kubectl apply -f cdi-smoke-gpu5.yaml
# gpu-server-6 同理 cdi-smoke-gpu6.yaml
kubectl describe pod -n clearml cdi-smoke-gpu6 | tail -15
kubectl delete pod -n clearml cdi-smoke-gpu5 cdi-smoke-gpu6
```

**不得出现** `unresolvable CDI devices management.nvidia.com/gpu=GPU-...`。

---

## 5. ClearML 多机回归

```bash
python train_launch_multinode_wholecard.py \
  --num-nodes 2 \
  --queue multinode-full-gpu \
  --vgpu-memory 4 \
  --vgpu-cores 30
```

检查：

- master / worker Pod 均 `Started`，可落在不同节点
- 日志：`HAMI-core Initializing`、`volcano.sh/vgpu-*` limits
- 无 CDI Events

---

## 6. 回滚（整栈）

```bash
helm upgrade volcano-vgpu-device-plugin -n $PLUGIN_NS --reuse-values --set cdi.enabled=false

# 每台 GPU 节点
sudo nvidia-ctk config --set mode=legacy
sudo rm -f /etc/cdi/nvidia.yaml
sudo systemctl restart k3s-agent   # 或 k3s

kubectl rollout restart deploy/clearml-agent -n clearml
kubectl rollout restart deploy/clearml-agent-multinode -n clearml
```

---

## 7. 勾选清单

```text
[ ] gpu-5/6: nvidia-ctk cdi generate，management.nvidia.com > 0
[ ] gpu-5/6: mode=cdi，restart k3s-agent / k3s
[ ] 两台 containerd config 结构接近
[ ] helm: cdi.enabled=true --reuse-values
[ ] plugin DS Ready；节点仍有 volcano.sh/vgpu-*
[ ] cdi-smoke 在 5、6 均 nvidia-smi 成功
[ ] clearml-agent + multinode-agent 已 rollout restart
[ ] multinode 2 Pod Running，无 CDI Events
```

---

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| `MULTINODE_schemes_zh.md` | 多机方案 1 操作 |
| `custom_values.yaml` | 单机 Agent + vgpuHook |
| `values-multinode-vgpu.standalone.example.yaml` | 多机 Agent |

K3s 勿手改 `config.toml`；plugin 与节点 CDI **必须同时开或同时关**。
