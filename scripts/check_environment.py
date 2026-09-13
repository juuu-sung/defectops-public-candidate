#!/usr/bin/env python3
"""Read local environment details without installing ML dependencies."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    versions = {}
    for package in ("anomalib", "torch", "torchvision", "lightning", "numpy", "fastapi", "streamlit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
              "architecture": platform.machine(), "processor": platform.processor(),
              "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
              "packages": versions,
              "note": "Package presence is not proof of GPU/backend usability. Verify with a model smoke test."}
    if shutil.which("nvidia-smi"):
        result["nvidia"] = command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                            "--format=csv,noheader"])
    else:
        result["nvidia"] = {"unavailable": "nvidia-smi not found; this does not establish GPU absence"}
    if platform.system() == "Darwin":
        result["memory_bytes"] = command_output(["sysctl", "-n", "hw.memsize"])
        result["chip"] = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
