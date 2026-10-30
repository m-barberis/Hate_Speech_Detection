#!/usr/bin/env bash
# Read-only inventory. Run on an allocated GPU node, in the intended work directory.
set -u
printf '\n=== Operating system and Python ===\n'
uname -a
command -v python3 || true
python3 --version || true
for interpreter in python3.10 python3.11 python3.12; do
    if command -v "$interpreter" >/dev/null 2>&1; then "$interpreter" --version; fi
done
printf '\n=== GPU, driver, free GPU memory and other GPU users ===\n'
nvidia-smi || true
printf '\n=== Available host memory ===\n'
free -h || true
printf '\n=== Free space in current directory and home ===\n'
df -h . "$HOME" || true
printf '\n=== User storage quota (if configured) ===\n'
if command -v quota >/dev/null 2>&1; then quota -s || true; fi
printf '\n=== Existing package versions; no imports or installations ===\n'
python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for package in ('torch', 'torchvision', 'transformers', 'accelerate', 'bitsandbytes', 'Pillow'):
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, 'not installed in this Python')
PY
