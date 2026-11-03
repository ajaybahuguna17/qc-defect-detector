#!/usr/bin/env python3
"""
ExecuTorch Environment & Hardware Backend Validation Script.

Validates the execution environment for Phase 2:
  1. Python runtime version (3.10 - 3.12 recommended for ExecuTorch).
  2. PyTorch installation & EXIR compatibility.
  3. ExecuTorch core package & extension status.
  4. Hardware backend acceleration: XNNPACK (Arm CPU), TorchAO, MPS/Metal, NEON/AVX.
  5. HuggingFace Transformers & tokenizers for Llama-3.2-1B-Instruct access.

Provides actionable, platform-specific setup commands for macOS (Student 2), Linux, and Windows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Tuple


# Compatibility Matrix
RECOMMENDED_PYTHON_MIN = (3, 10)
RECOMMENDED_PYTHON_MAX = (3, 12)
RECOMMENDED_TORCH_VERSION = "2.4.0"


class EnvironmentChecker:
    """Performs deep introspection of the PyTorch and ExecuTorch edge toolchain."""

    def __init__(self) -> None:
        self.os_name = platform.system()
        self.os_release = platform.release()
        self.arch = platform.machine()
        self.is_arm = self.arch.lower() in ("arm64", "aarch64")
        self.is_macos = self.os_name == "Darwin"
        self.is_windows = self.os_name == "Windows"
        self.is_linux = self.os_name == "Linux"

    def check_python(self) -> Dict[str, Any]:
        """Validate Python runtime version."""
        ver = sys.version_info
        ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
        is_supported = (
            (ver.major == 3)
            and (RECOMMENDED_PYTHON_MIN[1] <= ver.minor <= RECOMMENDED_PYTHON_MAX[1])
        )
        warning = None
        if ver.minor > RECOMMENDED_PYTHON_MAX[1]:
            warning = f"Python {ver_str} is newer than official ExecuTorch 0.3/0.4 prebuilt wheels (target 3.10-3.12)."
        elif ver.minor < RECOMMENDED_PYTHON_MIN[1]:
            warning = f"Python {ver_str} is deprecated. ExecuTorch requires Python >= 3.10."

        return {
            "version": ver_str,
            "major": ver.major,
            "minor": ver.minor,
            "executable": sys.executable,
            "supported": is_supported,
            "warning": warning,
        }

    def check_pytorch(self) -> Dict[str, Any]:
        """Check PyTorch core and dynamo / export availability."""
        try:
            import torch  # type: ignore

            torch_ver = torch.__version__
            has_dynamo = hasattr(torch, "_dynamo")
            has_export = hasattr(torch, "export") and hasattr(torch.export, "export")
            has_xnnpack_backend = hasattr(torch.backends, "xnnpack") and torch.backends.xnnpack.enabled

            # Check accelerator availability
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

            return {
                "installed": True,
                "version": torch_ver,
                "has_dynamo": has_dynamo,
                "has_export": has_export,
                "torch_xnnpack_enabled": has_xnnpack_backend,
                "cuda_available": has_cuda,
                "mps_available": has_mps,
                "warning": None,
            }
        except ImportError:
            return {
                "installed": False,
                "version": None,
                "has_dynamo": False,
                "has_export": False,
                "torch_xnnpack_enabled": False,
                "cuda_available": False,
                "mps_available": False,
                "warning": "PyTorch is not installed in the active environment.",
            }

    def check_executorch(self) -> Dict[str, Any]:
        """Check ExecuTorch core, EXIR compiler passes, and extensions."""
        status: Dict[str, Any] = {
            "installed": False,
            "version": None,
            "has_exir": False,
            "has_to_edge": False,
            "has_memory_planning": False,
            "has_xnnpack_partitioner": False,
            "warning": None,
        }

        try:
            import executorch  # type: ignore

            status["installed"] = True
            status["version"] = getattr(executorch, "__version__", "unknown")

            # Check EXIR
            try:
                import executorch.exir as exir  # type: ignore

                status["has_exir"] = True
                status["has_to_edge"] = hasattr(exir, "to_edge")
                status["has_memory_planning"] = hasattr(exir, "MemoryPlanningPass") or hasattr(
                    exir, "memory_planning"
                )
            except ImportError:
                pass

            # Check XNNPACK partitioner
            try:
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # type: ignore
                    XnnpackPartitioner,
                )

                status["has_xnnpack_partitioner"] = True
            except ImportError:
                pass

        except ImportError:
            status["warning"] = (
                "ExecuTorch is not installed. Pipeline B requires ExecuTorch SDK "
                "for lowering Llama 3.2 to mobile .pte format."
            )

        return status

    def check_quantization(self) -> Dict[str, Any]:
        """Check TorchAO and quantizer backends."""
        torchao_installed = False
        torchao_ver = None
        xnnpack_quantizer_available = False

        try:
            import torchao  # type: ignore

            torchao_installed = True
            torchao_ver = getattr(torchao, "__version__", "available")
        except ImportError:
            pass

        try:
            from executorch.backends.xnnpack.quantization.xnnpack_quantizer import (  # type: ignore
                XnnpackQuantizer,
            )

            xnnpack_quantizer_available = True
        except ImportError:
            pass

        return {
            "torchao_installed": torchao_installed,
            "torchao_version": torchao_ver,
            "xnnpack_quantizer_available": xnnpack_quantizer_available,
        }

    def check_huggingface(self) -> Dict[str, Any]:
        """Check HuggingFace transformers, huggingface_hub, and accelerate."""
        results: Dict[str, Any] = {}
        for pkg in ["transformers", "huggingface_hub", "accelerate", "sentencepiece", "tiktoken"]:
            try:
                mod = importlib.import_module(pkg)
                results[pkg] = getattr(mod, "__version__", "installed")
            except ImportError:
                results[pkg] = None
        return results

    def get_full_report(self) -> Dict[str, Any]:
        """Compile complete system status report."""
        py_status = self.check_python()
        torch_status = self.check_pytorch()
        et_status = self.check_executorch()
        quant_status = self.check_quantization()
        hf_status = self.check_huggingface()

        # Overall readiness evaluation
        ready_for_export = (
            py_status["supported"]
            and torch_status["installed"]
            and torch_status["has_export"]
            and et_status["installed"]
            and et_status["has_exir"]
        )

        return {
            "platform": {
                "system": self.os_name,
                "release": self.os_release,
                "architecture": self.arch,
                "is_arm": self.is_arm,
                "is_macos": self.is_macos,
            },
            "python": py_status,
            "pytorch": torch_status,
            "executorch": et_status,
            "quantization": quant_status,
            "huggingface": hf_status,
            "ready_for_export": ready_for_export,
        }

    def print_diagnostics(self) -> None:
        """Render diagnostic report to terminal."""
        report = self.get_full_report()
        p = report["platform"]
        py = report["python"]
        th = report["pytorch"]
        et = report["executorch"]
        q = report["quantization"]
        hf = report["huggingface"]

        print("=" * 72)
        print("EXECUTORCH & LLAMA 3.2 EDGE PIPELINE ENVIRONMENT CHECK")
        print("=" * 72)
        print(f"Host OS:             {p['system']} {p['release']} ({p['architecture']})")
        print(f"Target Architecture: {'Arm (NEON-accelerated target)' if p['is_arm'] else 'x86_64 / Host Workstation'}")
        print(f"Python Executable:   {py['executable']}")
        print(f"Python Version:      {py['version']} [{'OK' if py['supported'] else 'WARNING'}]")
        if py["warning"]:
            print(f"  -> Notice: {py['warning']}")

        print("-" * 72)
        print("PYTORCH STATUS:")
        if th["installed"]:
            cuda_mps = "CUDA" if th["cuda_available"] else ("MPS" if th["mps_available"] else "CPU")
            print(f"  PyTorch Core:      Installed (v{th['version']}) [Backend: {cuda_mps}]")
            print(f"  torch.export:      {'Available' if th['has_export'] else 'Missing'}")
            print(f"  XNNPACK in Torch:  {'Enabled' if th['torch_xnnpack_enabled'] else 'Disabled'}")
        else:
            print("  PyTorch Core:      NOT INSTALLED")

        print("-" * 72)
        print("EXECUTORCH TOOLCHAIN:")
        if et["installed"]:
            print(f"  ExecuTorch Core:   Installed (v{et['version']})")
            print(f"  EXIR Dialect:      {'Available' if et['has_exir'] else 'Missing'}")
            print(f"  to_edge Lowering:  {'Available' if et['has_to_edge'] else 'Missing'}")
            print(f"  Memory Planning:   {'Available' if et['has_memory_planning'] else 'Missing'}")
            print(f"  XNNPACK Delegate:  {'Available' if et['has_xnnpack_partitioner'] else 'Missing'}")
        else:
            print("  ExecuTorch Core:   NOT INSTALLED")
            print("  -> Phase 2 lowering scripts will run in dry-run/mock mode on this host.")

        print("-" * 72)
        print("QUANTIZATION & LLM ASSETS:")
        print(f"  TorchAO (4/8-bit): {'Installed (v' + q['torchao_version'] + ')' if q['torchao_installed'] else 'Missing'}")
        print(f"  XnnpackQuantizer:  {'Available' if q['xnnpack_quantizer_available'] else 'Not found'}")
        print(f"  Transformers:      {'Installed (v' + hf['transformers'] + ')' if hf['transformers'] else 'Missing'}")
        print(f"  HuggingFace Hub:   {'Installed' if hf['huggingface_hub'] else 'Missing'}")
        print(f"  TikToken:          {'Installed' if hf['tiktoken'] else 'Missing'}")

        print("=" * 72)
        if report["ready_for_export"]:
            print("STATUS: ALL SYSTEMS READY for full Llama 3.2 ExecuTorch compilation.")
        else:
            print("STATUS: ENVIRONMENT READY FOR CODE VERIFICATION & DRY-RUN EXPORTS.")
            print("To enable full hardware lowering to .pte, follow the setup instructions below.")
        print("=" * 72)

    def print_setup_guide(self) -> None:
        """Display exact setup instructions for each platform."""
        print("\n" + "=" * 72)
        print("STUDENT 2 OPERATIONAL SETUP GUIDE (ExecuTorch on macOS Apple Silicon)")
        print("=" * 72)
        print("""
# 1. Create a dedicated virtual environment with Python 3.11:
conda create -n executorch-env python=3.11 -y
conda activate executorch-env

# 2. Install nightly/pre-release PyTorch, TorchAO, and ExecuTorch:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install --pre torchao --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install --pre executorch --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 3. Clone ExecuTorch source repository for Llama examples & XNNPACK backend:
git clone --depth=1 --recursive https://github.com/pytorch/executorch.git
cd executorch
./install_requirements.sh

# 4. Install HuggingFace dependencies & authentication for Llama-3.2-1B:
pip install transformers huggingface_hub accelerate tiktoken sentencepiece
huggingface-cli login
""")
        print("=" * 72)
        print("LINUX / WINDOWS WORKSTATION SETUP GUIDE")
        print("=" * 72)
        print("""
# On Linux / Windows with CPU or CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --pre executorch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers huggingface_hub tiktoken
""")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ExecuTorch environment diagnostics for QC Reasoning Pipeline."
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON status.")
    parser.add_argument(
        "--guide", "--install-guide", action="store_true", help="Display full installation guide."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Return exit code 1 if ExecuTorch is not installed."
    )
    args = parser.parse_args()

    checker = EnvironmentChecker()

    if args.json:
        report = checker.get_full_report()
        print(json.dumps(report, indent=2))
        return 0 if (report["ready_for_export"] or not args.strict) else 1

    checker.print_diagnostics()

    if args.guide:
        checker.print_setup_guide()

    report = checker.get_full_report()
    if args.strict and not report["ready_for_export"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
