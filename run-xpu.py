"""Run a pipeline script with the native Intel XPU PyTorch build."""

import runpy
import sys

import torch


def main() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit("Intel XPU is not available in this Python environment")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run-xpu.py <script.py> [args ...]")
    if sys.argv[1].startswith("-"):
        raise SystemExit("usage: python run-xpu.py <script.py> [args ...]")
    print(f"Intel XPU: {torch.xpu.get_device_name(0)}")
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()