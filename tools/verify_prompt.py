#!/usr/bin/env python3
"""Validate SenseBench prompt JSON files from a source checkout."""

from __future__ import annotations

import sys

from sensebench.verify.prompts import main


def _main() -> int:
    return main()


if __name__ == "__main__":
    sys.exit(_main())
