#!/usr/bin/env python3
"""Destroy the vast.ai instance recorded in an instance.json file.

Confirms the instance is actually gone (you keep paying until it is) and stamps
destroyed_at into the instance.json on success. Stdlib only; runs locally.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_VASTAI_COMMAND: str = "uvx vastai@0.5.0"
DESTROY_POLL_ATTEMPTS: int = 3
DESTROY_POLL_INTERVAL_SECONDS: float = 10.0
GONE_STATUSES: frozenset[str] = frozenset({"destroyed", "terminated", "deleted"})
INSTANCE_ID_FIELD: str = "instance_id"
DESTROYED_AT_FIELD: str = "destroyed_at"


def _log(*, message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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


def _instance_is_gone(*, vastai_command: list[str], instance_id: int) -> bool:
    info = _show_instance(vastai_command=vastai_command, instance_id=instance_id)
    if info is None or len(info) == 0:
        return True
    actual_status: object = info.get("actual_status")
    if isinstance(actual_status, str) and actual_status.lower() in GONE_STATUSES:
        return True
    intended_status: object = info.get("intended_status")
    return isinstance(intended_status, str) and intended_status.lower() in GONE_STATUSES


def _stamp_destroyed(*, instance_path: Path, record: dict[str, object]) -> None:
    record[DESTROYED_AT_FIELD] = datetime.now(tz=UTC).isoformat()
    instance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Destroy the vast.ai instance recorded in an instance.json file."
    )
    parser.add_argument("instance_json", help="instance.json written by provision.py.")
    parser.add_argument(
        "--vastai-cmd",
        default=DEFAULT_VASTAI_COMMAND,
        help="vastai CLI invocation, split on spaces.",
    )
    args = parser.parse_args(argv)
    vastai_command: list[str] = str(args.vastai_cmd).split()
    instance_path = Path(str(args.instance_json))
    record: object = json.loads(instance_path.read_text(encoding="utf-8"))
    assert isinstance(record, dict), "instance.json holds a JSON object"
    instance_id: object = record.get(INSTANCE_ID_FIELD)
    if not isinstance(instance_id, int):
        _log(message=f"{instance_path}: missing integer {INSTANCE_ID_FIELD} field")
        return 2

    destroy: list[str] = vastai_command + ["destroy", "instance", str(instance_id)]
    _log(message=f"destroying instance {instance_id}")
    result = subprocess.run(destroy, capture_output=True, text=True)
    if result.returncode != 0:
        # The instance may already be gone; the poll below decides.
        _log(message=f"destroy command failed: {result.stderr.strip()}")

    for attempt in range(1, DESTROY_POLL_ATTEMPTS + 1):
        if _instance_is_gone(vastai_command=vastai_command, instance_id=instance_id):
            _stamp_destroyed(instance_path=instance_path, record=record)
            _log(message=f"instance {instance_id} destroyed; stamped {instance_path}")
            return 0
        _log(
            message=(
                f"instance {instance_id} still reported alive "
                f"(check {attempt}/{DESTROY_POLL_ATTEMPTS})"
            )
        )
        if attempt < DESTROY_POLL_ATTEMPTS:
            time.sleep(DESTROY_POLL_INTERVAL_SECONDS)

    _log(
        message=(
            f"!!! instance {instance_id} STILL APPEARS ALIVE — you are still being billed.\n"
            f"!!! retry:  {' '.join(vastai_command)} destroy instance {instance_id}\n"
            f"!!! verify: {' '.join(vastai_command)} show instances --raw"
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
