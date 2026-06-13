#!/usr/bin/env python3
"""Provision a vast.ai GPU instance for a SenseBench self-hosted benchmark batch.

Searches vast.ai offers for the requested GPU preset from the manifest, rents the
cheapest acceptable offer, waits for the instance to start, SSH-verifies the GPU,
and writes an instance.json consumed by the other tools in this directory.

Runs on the operator's local machine. See docs/self_hosted.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from sensebench.paths import (
    DEFAULT_SSH_KEY_PATH,
    INSTANCE_FILENAME,
    SELF_HOSTED_MANIFEST_PATH,
    WORK_ROOT,
)
from tools.self_hosted.models import (
    GpuPresetConfig,
    GpuPresetKey,
    InstanceRecord,
    SelfHostedManifest,
)

DEFAULT_VASTAI_COMMAND: str = "uvx vastai@0.5.0"
DEFAULT_MAX_ATTEMPTS: int = 3
LABEL_PREFIX: str = "sensebench"
PROVIDER_NAME: str = "vast.ai"
SSH_USER: str = "root"
SSH_CONNECT_TIMEOUT_SECONDS: float = 20.0
SSH_CONNECT_TIMEOUT_TEXT: str = f"{SSH_CONNECT_TIMEOUT_SECONDS:g}"
GPU_NAME_QUERY: list[str] = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
OFFER_ORDER_FIELD: str = "dph"
OFFER_SEARCH_LIMIT: int = 20
RUNNING_STATUS: str = "running"
INSTANCE_POLL_INTERVAL_SECONDS: float = 30.0
INSTANCE_POLL_TIMEOUT_SECONDS: float = 600.0
SSH_VERIFY_ATTEMPTS: int = 5
SSH_VERIFY_RETRY_SECONDS: float = 30.0
SSH_AUTH_FAILURE_MARKERS: list[str] = ["Permission denied", "publickey"]
PROVISION_PROGRESS_DESCRIPTION: str = "Provisioning offers"
PROVISION_PROGRESS_UNIT: str = "offer"


@dataclass(frozen=True, slots=True)
class ProvisionPlan:
    preset_key: GpuPresetKey
    preset: GpuPresetConfig
    image: str
    label: str


@dataclass(frozen=True, slots=True)
class Offer:
    offer_id: int
    dph_total: float


@dataclass(frozen=True, slots=True)
class RunningInstance:
    instance_id: int
    ssh_host: str
    ssh_port: int
    hourly_rate_usd: float


def _log(*, message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _vastai_json(*, vastai_command: list[str], arguments: list[str]) -> object:
    command: list[str] = vastai_command + arguments
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _load_plan(
    *,
    manifest_path: Path,
    gpu: GpuPresetKey,
    label: str,
) -> ProvisionPlan:
    manifest = SelfHostedManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    preset = manifest.gpu_presets.get(gpu)
    if preset is None:
        available = ", ".join(sorted(manifest.gpu_presets))
        raise SystemExit(f"unknown gpu preset {gpu!r}; available presets: {available}")
    return ProvisionPlan(
        preset_key=gpu,
        preset=preset,
        image=manifest.default_image,
        label=label,
    )


def _search_offers(*, vastai_command: list[str], preset: GpuPresetConfig) -> list[Offer]:
    arguments: list[str] = [
        "search",
        "offers",
        preset.search,
        "--order",
        OFFER_ORDER_FIELD,
        "--limit",
        str(OFFER_SEARCH_LIMIT),
        "--raw",
    ]
    payload = _vastai_json(vastai_command=vastai_command, arguments=arguments)
    assert isinstance(payload, list), "search offers --raw returns a JSON list"
    offers: list[Offer] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        offer_id: object = entry.get("id")
        dph_total: object = entry.get("dph_total")
        if not isinstance(offer_id, int) or not isinstance(dph_total, int | float):
            continue
        if float(dph_total) <= preset.max_hourly_usd:
            offers.append(Offer(offer_id=offer_id, dph_total=float(dph_total)))
    return sorted(offers, key=lambda offer: offer.dph_total)


def _create_instance(*, vastai_command: list[str], plan: ProvisionPlan, offer: Offer) -> int:
    arguments: list[str] = [
        "create",
        "instance",
        str(offer.offer_id),
        "--image",
        plan.image,
        "--ssh",
        "--disk",
        str(plan.preset.disk_gb),
        "--raw",
    ]
    payload = _vastai_json(vastai_command=vastai_command, arguments=arguments)
    assert isinstance(payload, dict), "create instance --raw returns a JSON object"
    new_contract: object = payload.get("new_contract")
    if not isinstance(new_contract, int):
        raise RuntimeError(f"create instance returned no new_contract: {payload}")
    return new_contract


def _label_instance(*, vastai_command: list[str], instance_id: int, label: str) -> None:
    command: list[str] = vastai_command + ["label", "instance", str(instance_id), label]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        _log(message=f"warning: labeling instance {instance_id} failed: {result.stderr.strip()}")


def _show_instance(*, vastai_command: list[str], instance_id: int) -> dict[str, object] | None:
    command: list[str] = vastai_command + ["show", "instance", str(instance_id), "--raw"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        payload: object = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list):
        payload = payload[0] if len(payload) > 0 else None
    if not isinstance(payload, dict):
        return None
    return payload


def _destroy_instance(*, vastai_command: list[str], instance_id: int) -> None:
    command: list[str] = vastai_command + ["destroy", "instance", str(instance_id)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        _log(
            message=(
                f"warning: destroying instance {instance_id} failed: {result.stderr.strip()}; "
                f"destroy it manually with: {' '.join(command)}"
            )
        )


def _wait_for_running(
    *,
    vastai_command: list[str],
    instance_id: int,
) -> RunningInstance | None:
    deadline: float = time.monotonic() + INSTANCE_POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        info = _show_instance(vastai_command=vastai_command, instance_id=instance_id)
        if info is not None and info.get("actual_status") == RUNNING_STATUS:
            ssh_host: object = info.get("ssh_host")
            ssh_port: object = info.get("ssh_port")
            dph_total: object = info.get("dph_total")
            if (
                isinstance(ssh_host, str)
                and isinstance(ssh_port, int)
                and isinstance(dph_total, int | float)
            ):
                return RunningInstance(
                    instance_id=instance_id,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    hourly_rate_usd=float(dph_total),
                )
        status = "(no response)" if info is None else str(info.get("actual_status"))
        _log(message=f"instance {instance_id}: actual_status={status}; waiting...")
        time.sleep(INSTANCE_POLL_INTERVAL_SECONDS)
    return None


def _ssh_command(*, instance: RunningInstance) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_TEXT}",
        "-i",
        str(DEFAULT_SSH_KEY_PATH),
        "-p",
        str(instance.ssh_port),
        f"{SSH_USER}@{instance.ssh_host}",
    ]


def _verify_gpu_over_ssh(*, vastai_command: list[str], instance: RunningInstance) -> bool:
    command: list[str] = _ssh_command(instance=instance) + GPU_NAME_QUERY
    for attempt in range(1, SSH_VERIFY_ATTEMPTS + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        gpu_name = result.stdout.strip()
        if result.returncode == 0 and len(gpu_name) > 0:
            _log(message=f"instance {instance.instance_id}: GPU verified: {gpu_name}")
            return True
        _log(
            message=(
                f"ssh GPU check {attempt}/{SSH_VERIFY_ATTEMPTS} failed "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        )
        if any(marker in result.stderr for marker in SSH_AUTH_FAILURE_MARKERS):
            _log(
                message=(
                    "hint: the instance may not have your SSH key; attach it with: "
                    f"{' '.join(vastai_command)} attach ssh {instance.instance_id} "
                    f'"$(cat {DEFAULT_SSH_KEY_PATH}.pub)"'
                )
            )
        if attempt < SSH_VERIFY_ATTEMPTS:
            time.sleep(SSH_VERIFY_RETRY_SECONDS)
    return False


def _provision_offer(
    *,
    vastai_command: list[str],
    plan: ProvisionPlan,
    offer: Offer,
) -> RunningInstance | None:
    _log(message=f"creating instance from offer {offer.offer_id} (${offer.dph_total:.3f}/h)")
    try:
        instance_id = _create_instance(vastai_command=vastai_command, plan=plan, offer=offer)
    except (RuntimeError, json.JSONDecodeError) as error:
        _log(message=f"create instance failed for offer {offer.offer_id}: {error}")
        return None
    _log(message=f"created instance {instance_id}; waiting for it to start")
    _label_instance(vastai_command=vastai_command, instance_id=instance_id, label=plan.label)
    instance = _wait_for_running(vastai_command=vastai_command, instance_id=instance_id)
    if instance is None:
        _log(
            message=(
                f"instance {instance_id} did not reach {RUNNING_STATUS!r} within "
                f"{INSTANCE_POLL_TIMEOUT_SECONDS:.0f}s; destroying it"
            )
        )
        _destroy_instance(vastai_command=vastai_command, instance_id=instance_id)
        return None
    if not _verify_gpu_over_ssh(vastai_command=vastai_command, instance=instance):
        _log(message=f"SSH GPU verification failed for instance {instance_id}; destroying it")
        _destroy_instance(vastai_command=vastai_command, instance_id=instance_id)
        return None
    return instance


def _write_instance_file(
    *,
    out_path: Path,
    plan: ProvisionPlan,
    offer: Offer,
    instance: RunningInstance,
) -> None:
    record = InstanceRecord(
        provider=PROVIDER_NAME,
        instance_id=instance.instance_id,
        gpu_preset=plan.preset_key,
        gpu_label=plan.preset.gpu_label,
        offer_id=offer.offer_id,
        image=plan.image,
        disk_gb=plan.preset.disk_gb,
        ssh_host=instance.ssh_host,
        ssh_port=instance.ssh_port,
        ssh_user=SSH_USER,
        hourly_rate_usd=instance.hourly_rate_usd,
        label=plan.label,
        created_at=datetime.now(tz=UTC),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        record.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rent and verify a vast.ai GPU instance for a SenseBench batch."
    )
    parser.add_argument("--gpu", required=True, help="GPU preset key from the manifest.")
    parser.add_argument("--manifest", default=str(SELF_HOSTED_MANIFEST_PATH))
    parser.add_argument(
        "--out",
        default=None,
        help=f"Output path (default {WORK_ROOT}/<gpu>/{INSTANCE_FILENAME}).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help=f"Instance label (default {LABEL_PREFIX}/<gpu>-batch).",
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--vastai-cmd",
        default=DEFAULT_VASTAI_COMMAND,
        help="vastai CLI invocation, split on spaces.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gpu = str(args.gpu)
    max_attempts = int(args.max_attempts)
    if max_attempts < 1:
        _log(message="--max-attempts must be a positive integer")
        return 2
    vastai_command: list[str] = str(args.vastai_cmd).split()
    label = str(args.label) if args.label is not None else f"{LABEL_PREFIX}/{gpu}-batch"
    out_path = Path(str(args.out)) if args.out is not None else WORK_ROOT / gpu / INSTANCE_FILENAME
    plan = _load_plan(manifest_path=Path(str(args.manifest)), gpu=gpu, label=label)
    _log(message=f"searching offers for {plan.preset.gpu_label}: {plan.preset.search}")
    try:
        offers = _search_offers(vastai_command=vastai_command, preset=plan.preset)
    except (RuntimeError, json.JSONDecodeError, OSError) as error:
        _log(message=f"offer search failed: {error}")
        return 1
    if len(offers) == 0:
        _log(
            message=(
                f"no rentable offers at or under ${plan.preset.max_hourly_usd}/h for "
                f"preset {gpu!r}; loosen the search query or raise max_hourly_usd"
            )
        )
        return 1
    candidates: list[Offer] = offers[:max_attempts]
    _log(message=f"found {len(offers)} acceptable offers; trying up to {len(candidates)}")
    for offer in tqdm(
        candidates,
        desc=PROVISION_PROGRESS_DESCRIPTION,
        unit=PROVISION_PROGRESS_UNIT,
    ):
        instance = _provision_offer(vastai_command=vastai_command, plan=plan, offer=offer)
        if instance is None:
            continue
        _write_instance_file(out_path=out_path, plan=plan, offer=offer, instance=instance)
        _log(message=f"instance {instance.instance_id} ready; wrote {out_path}")
        print(" ".join(_ssh_command(instance=instance)))
        return 0
    _log(
        message=(
            f"all {len(candidates)} provisioning attempts failed for preset {gpu!r}; "
            "check vast.ai account status and offer availability, then retry"
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
