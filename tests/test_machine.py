from __future__ import annotations

import subprocess

from pytest import MonkeyPatch

from sensebench.runner.machine import (
    collect_machine_info,
    parse_cpu_model,
    parse_cuda_version,
    parse_gpu_query_output,
    parse_mem_total_gib,
)
from sensebench.runs.models import MachineInfo

SINGLE_GPU_QUERY_OUTPUT: str = "NVIDIA H100 80GB HBM3, 81559, 565.57.01\n"
MULTI_GPU_QUERY_OUTPUT: str = (
    "NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
    "NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
    "NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
    "NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
)
HETEROGENEOUS_GPU_QUERY_OUTPUT: str = (
    "NVIDIA H100 80GB HBM3, 81559, 565.57.01\nNVIDIA A100-SXM4-80GB, 81920, 565.57.01\n"
)
NVIDIA_SMI_BANNER: str = (
    "Thu Jun 12 10:00:00 2026\n"
    "+-----------------------------------------------------------------------------+\n"
    "| NVIDIA-SMI 565.57.01    Driver Version: 565.57.01    CUDA Version: 12.7     |\n"
)
PROC_CPUINFO: str = (
    "processor\t: 0\n"
    "vendor_id\t: AuthenticAMD\n"
    "model name\t: AMD EPYC 9554 64-Core Processor\n"
    "processor\t: 1\n"
    "model name\t: AMD EPYC 9554 64-Core Processor\n"
)
PROC_MEMINFO: str = "MemTotal:       263856332 kB\nMemFree:        123456 kB\n"
EXPECTED_RAM_GIB: float = 251.6
SINGLE_GPU_NAME: str = "NVIDIA H100 80GB HBM3"
SINGLE_GPU_VRAM_MIB: int = 81559
SINGLE_GPU_DRIVER: str = "565.57.01"
MULTI_GPU_COUNT: int = 4
EXPECTED_CUDA_VERSION: str = "12.7"
EXPECTED_CPU_MODEL: str = "AMD EPYC 9554 64-Core Processor"
SUBPROCESS_RUN_ATTR: str = "run"


def test_parse_gpu_query_output_single_gpu() -> None:
    gpu = parse_gpu_query_output(text=SINGLE_GPU_QUERY_OUTPUT)

    assert gpu is not None
    assert gpu.name == SINGLE_GPU_NAME
    assert gpu.count == 1
    assert gpu.vram_mib_per_gpu == SINGLE_GPU_VRAM_MIB
    assert gpu.driver_version == SINGLE_GPU_DRIVER


def test_parse_gpu_query_output_counts_homogeneous_gpus() -> None:
    gpu = parse_gpu_query_output(text=MULTI_GPU_QUERY_OUTPUT)

    assert gpu is not None
    assert gpu.count == MULTI_GPU_COUNT
    assert gpu.name == "NVIDIA A100-SXM4-80GB"


def test_parse_gpu_query_output_joins_heterogeneous_names() -> None:
    gpu = parse_gpu_query_output(text=HETEROGENEOUS_GPU_QUERY_OUTPUT)

    assert gpu is not None
    assert gpu.count == 2
    assert "NVIDIA H100 80GB HBM3" in gpu.name
    assert "NVIDIA A100-SXM4-80GB" in gpu.name


def test_parse_gpu_query_output_empty_text() -> None:
    assert parse_gpu_query_output(text="") is None


def test_parse_cuda_version_from_banner() -> None:
    assert parse_cuda_version(banner=NVIDIA_SMI_BANNER) == EXPECTED_CUDA_VERSION


def test_parse_cuda_version_missing() -> None:
    assert parse_cuda_version(banner="no gpus here") is None


def test_parse_cpu_model_from_proc_cpuinfo() -> None:
    assert parse_cpu_model(cpuinfo=PROC_CPUINFO) == EXPECTED_CPU_MODEL


def test_parse_mem_total_gib_from_proc_meminfo() -> None:
    assert parse_mem_total_gib(meminfo=PROC_MEMINFO) == EXPECTED_RAM_GIB


def test_collect_machine_info_degrades_without_probes(monkeypatch: MonkeyPatch) -> None:
    def raising_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("command not found")

    monkeypatch.setattr(target=subprocess, name=SUBPROCESS_RUN_ATTR, value=raising_run)

    machine = collect_machine_info(provider="vast.ai", instance_id="123", hourly_rate_usd=2.5)

    assert machine.gpu is None
    assert machine.provider == "vast.ai"
    assert machine.instance_id == "123"
    assert machine.hourly_rate_usd == 2.5
    assert machine.platform is not None
    assert machine.cpu_cores is not None


def test_machine_info_json_round_trip() -> None:
    machine = collect_machine_info(provider="vast.ai")

    restored = MachineInfo.model_validate_json(machine.model_dump_json())

    assert restored == machine
