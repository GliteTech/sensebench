"""Collect hardware details for self-hosted runs."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from sensebench.runs.models import MachineGpuInfo, MachineInfo

SUBPROCESS_TIMEOUT_SECONDS: float = 10.0
NVIDIA_SMI_COMMAND: str = "nvidia-smi"
NVIDIA_SMI_GPU_QUERY_ARGS: list[str] = [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]
GPU_QUERY_FIELD_COUNT: int = 3
GPU_NAME_JOIN_SEPARATOR: str = " + "
CUDA_VERSION_PATTERN: re.Pattern[str] = re.compile(r"CUDA Version:\s*([0-9.]+)")
CPU_MODEL_PATTERN: re.Pattern[str] = re.compile(r"^model name\s*:\s*(.+)$", re.MULTILINE)
MEM_TOTAL_PATTERN: re.Pattern[str] = re.compile(r"^MemTotal:\s*(\d+)\s*kB$", re.MULTILINE)
PROC_CPUINFO_PATH: Path = Path("/proc/cpuinfo")
PROC_MEMINFO_PATH: Path = Path("/proc/meminfo")
MACOS_CPU_MODEL_COMMAND: list[str] = ["sysctl", "-n", "machdep.cpu.brand_string"]
MACOS_MEM_BYTES_COMMAND: list[str] = ["sysctl", "-n", "hw.memsize"]
KIB_PER_GIB: float = 1024.0 * 1024.0
BYTES_PER_GIB: float = 1024.0**3
RAM_GIB_DECIMALS: int = 1


def _run_command(*, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args=args,
            capture_output=True,
            check=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    return result.stdout


def _read_text(*, path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_int(*, value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_gpu_query_output(*, text: str) -> MachineGpuInfo | None:
    rows: list[list[str]] = [
        [field.strip() for field in line.split(",")]
        for line in text.splitlines()
        if len(line.strip()) > 0
    ]
    rows = [row for row in rows if len(row) == GPU_QUERY_FIELD_COUNT and len(row[0]) > 0]
    if len(rows) == 0:
        return None
    unique_names: list[str] = sorted({row[0] for row in rows})
    vram_values: list[int] = [
        vram for row in rows if (vram := _parse_int(value=row[1])) is not None
    ]
    driver_values: list[str] = sorted({row[2] for row in rows if len(row[2]) > 0})
    return MachineGpuInfo(
        name=GPU_NAME_JOIN_SEPARATOR.join(unique_names),
        count=len(rows),
        vram_mib_per_gpu=max(vram_values) if len(vram_values) > 0 else None,
        driver_version=driver_values[0] if len(driver_values) == 1 else None,
    )


def parse_cuda_version(*, banner: str) -> str | None:
    match = CUDA_VERSION_PATTERN.search(banner)
    if match is None:
        return None
    return match.group(1)


def parse_cpu_model(*, cpuinfo: str) -> str | None:
    match = CPU_MODEL_PATTERN.search(cpuinfo)
    if match is None:
        return None
    return match.group(1).strip()


def parse_mem_total_gib(*, meminfo: str) -> float | None:
    match = MEM_TOTAL_PATTERN.search(meminfo)
    if match is None:
        return None
    return round(int(match.group(1)) / KIB_PER_GIB, RAM_GIB_DECIMALS)


def _collect_gpu_info() -> MachineGpuInfo | None:
    query_output = _run_command(args=[NVIDIA_SMI_COMMAND, *NVIDIA_SMI_GPU_QUERY_ARGS])
    if query_output is None:
        return None
    gpu = parse_gpu_query_output(text=query_output)
    if gpu is None:
        return None
    banner = _run_command(args=[NVIDIA_SMI_COMMAND])
    if banner is None:
        return gpu
    return gpu.model_copy(update={"cuda_version": parse_cuda_version(banner=banner)})


def _collect_cpu_model() -> str | None:
    cpuinfo = _read_text(path=PROC_CPUINFO_PATH)
    if cpuinfo is not None:
        model = parse_cpu_model(cpuinfo=cpuinfo)
        if model is not None:
            return model
    sysctl_output = _run_command(args=MACOS_CPU_MODEL_COMMAND)
    if sysctl_output is None:
        return None
    stripped = sysctl_output.strip()
    return stripped if len(stripped) > 0 else None


def _collect_ram_gib() -> float | None:
    meminfo = _read_text(path=PROC_MEMINFO_PATH)
    if meminfo is not None:
        ram_gib = parse_mem_total_gib(meminfo=meminfo)
        if ram_gib is not None:
            return ram_gib
    sysctl_output = _run_command(args=MACOS_MEM_BYTES_COMMAND)
    if sysctl_output is None:
        return None
    mem_bytes = _parse_int(value=sysctl_output)
    if mem_bytes is None:
        return None
    return round(mem_bytes / BYTES_PER_GIB, RAM_GIB_DECIMALS)


def collect_machine_info(
    *,
    provider: str | None = None,
    instance_id: str | None = None,
    hourly_rate_usd: float | None = None,
) -> MachineInfo:
    return MachineInfo(
        platform=platform.platform(),
        cpu_model=_collect_cpu_model(),
        cpu_cores=os.cpu_count(),
        ram_gib=_collect_ram_gib(),
        gpu=_collect_gpu_info(),
        provider=provider,
        instance_id=instance_id,
        hourly_rate_usd=hourly_rate_usd,
    )
