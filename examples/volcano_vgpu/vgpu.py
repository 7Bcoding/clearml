"""SDK helpers for per-task Volcano vGPU on ClearML k8s glue.

Platform convention (this repo):
    volcano-vgpu-device-plugin --gpu-memory-factor=1024
    -> vgpu_memory is GiB integers (2 = 2GB, passed to volcano.sh/vgpu-memory as "2")
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Union

from clearml import Task

VGPU_SECTION = "VGPU"
# Must match volcano-vgpu-device-plugin --gpu-memory-factor on the GPU nodes.
GPU_MEMORY_FACTOR = 1024


def connect_vgpu(
    task: Task,
    *,
    vgpu_number: int = 1,
    vgpu_memory: int,
    vgpu_cores: int,
    section: str = VGPU_SECTION,
    memory_factor: int = GPU_MEMORY_FACTOR,
) -> dict[str, Any]:
    """Register per-task vGPU limits.

    Requires platform ``agentk8sglue.vgpuHook.enabled=true``.

    When ``memory_factor=1024`` (default here), ``vgpu_memory`` is **GiB**:
        connect_vgpu(task, vgpu_memory=2, vgpu_cores=30)  # 2GB, 30% cores

    When ``memory_factor=1``, ``vgpu_memory`` is **MiB**:
        connect_vgpu(task, vgpu_memory=2048, vgpu_cores=30, memory_factor=1)
    """
    if vgpu_number < 1:
        raise ValueError("vgpu_number must be >= 1")
    if vgpu_memory < 1:
        unit = "GiB" if memory_factor > 1 else "MiB"
        raise ValueError("vgpu_memory must be >= 1 (%s)" % unit)
    if not 0 < vgpu_cores <= 100:
        raise ValueError("vgpu_cores must be in (0, 100]")

    if memory_factor > 1 and vgpu_memory > 64:
        warnings.warn(
            "vgpu_memory=%s looks like legacy MiB on a factor=%s cluster; use GiB (e.g. 2 for 2GB)"
            % (vgpu_memory, memory_factor),
            stacklevel=2,
        )

    return task.connect(
        {
            "vgpu_number": int(vgpu_number),
            "vgpu_memory": int(vgpu_memory),
            "vgpu_cores": int(vgpu_cores),
            "gpu_memory_factor": int(memory_factor),
        },
        name=section,
    )


def connect_vgpu_from_dict(
    task: Task,
    config: Mapping[str, Union[int, str]],
    section: str = VGPU_SECTION,
    memory_factor: int = GPU_MEMORY_FACTOR,
) -> dict[str, Any]:
    """Same as :func:`connect_vgpu` but accepts a mapping."""
    return connect_vgpu(
        task,
        vgpu_number=int(config.get("vgpu_number", 1)),
        vgpu_memory=int(config["vgpu_memory"]),
        vgpu_cores=int(config["vgpu_cores"]),
        section=section,
        memory_factor=int(config.get("gpu_memory_factor", memory_factor)),
    )


def expected_memory_mib(vgpu_memory: int, memory_factor: int = GPU_MEMORY_FACTOR) -> int:
    """Convert SDK vgpu_memory to expected nvidia-smi memory.total (MiB)."""
    return int(vgpu_memory) * int(memory_factor)
