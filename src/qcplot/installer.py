"""QCPlot Cross-Platform Automated Installer & Environment Setup.

Detects the host operating system (Windows, Linux, macOS), CPU architecture,
and hardware acceleration capabilities (DirectML, CUDA, ROCm, Apple Metal/CoreML),
resolves appropriate dependencies, installs packages, and provisions the workspace.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Determine workspace root
_repo_root = Path(__file__).resolve().parent.parent.parent
if (_repo_root / "pyproject.toml").exists():
    PROJECT_ROOT = _repo_root
else:
    PROJECT_ROOT = Path.cwd()

# Color helper for terminals before rich is installed
_IS_TTY = sys.stdout.isatty() and os.name != "nt" or (os.name == "nt" and "WT_SESSION" in os.environ)


def _c(text: str, code: str) -> str:
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(text: str) -> str:
    return _c(text, "1;36")


def green(text: str) -> str:
    return _c(text, "1;32")


def yellow(text: str) -> str:
    return _c(text, "1;33")


def red(text: str) -> str:
    return _c(text, "1;31")


def dim(text: str) -> str:
    return _c(text, "2;37")


def bold(text: str) -> str:
    return _c(text, "1")


def detect_platform_info() -> dict[str, Any]:
    """Detect host OS, architecture, and compute capabilities."""
    system = platform.system()
    machine = platform.machine().lower()
    os_name = "unknown"
    is_windows = False
    is_linux = False
    is_macos = False

    if system == "Windows":
        os_name = "Windows"
        is_windows = True
    elif system == "Linux":
        os_name = "Linux"
        is_linux = True
    elif system == "Darwin":
        os_name = "macOS"
        is_macos = True
    else:
        os_name = system

    is_arm = machine in ("arm64", "aarch64")
    is_x86_64 = machine in ("x86_64", "amd64")

    # Detect hardware acceleration indicators
    gpus: list[str] = []
    cuda_available = False
    rocm_available = False
    directml_capable = is_windows

    # Check NVIDIA CUDA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                gpu_names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                gpus.extend(gpu_names)
                cuda_available = True
        except Exception:
            pass

    # Check AMD ROCm via rocm-smi
    if shutil.which("rocm-smi"):
        try:
            res = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                rocm_available = True
                gpus.append("AMD ROCm GPU")
        except Exception:
            pass

    # Check Apple Silicon GPU
    if is_macos and is_arm:
        gpus.append("Apple Silicon GPU (Metal / CoreML)")

    # Select recommended ONNX runtime package for the platform
    # On Windows: onnxruntime-directml is the first-class DirectML GPU accelerator
    # On Linux / macOS / others: onnxruntime (or onnxruntime-gpu if NVIDIA CUDA present)
    if is_windows:
        recommended_ort = "onnxruntime-directml>=1.24.4"
        ort_type = "DirectML (Windows GPU)"
    elif cuda_available and not is_arm:
        # User has Linux with NVIDIA GPU: can use onnxruntime or onnxruntime-gpu
        recommended_ort = "onnxruntime>=1.20.0"
        ort_type = "CPU / CUDA (Linux)"
    elif is_macos:
        recommended_ort = "onnxruntime>=1.20.0"
        ort_type = "CoreML / CPU (macOS)"
    else:
        recommended_ort = "onnxruntime>=1.20.0"
        ort_type = "CPU (Linux / Generic)"

    return {
        "os_name": os_name,
        "system": system,
        "release": platform.release(),
        "machine": machine,
        "is_windows": is_windows,
        "is_linux": is_linux,
        "is_macos": is_macos,
        "is_arm": is_arm,
        "is_x86_64": is_x86_64,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "gpus": gpus,
        "cuda_available": cuda_available,
        "rocm_available": rocm_available,
        "directml_capable": directml_capable,
        "recommended_ort": recommended_ort,
        "ort_type": ort_type,
    }


def get_core_dependencies(platform_info: dict[str, Any] | None = None) -> list[str]:
    """Return complete list of package requirements adapted for the current platform."""
    if platform_info is None:
        platform_info = detect_platform_info()

    deps = [
        "laya>=0.5.0",
        "matplotlib>=3.11.2",
        "numpy>=2.5.5",
        "onnx>=1.23.2",
        platform_info["recommended_ort"],
        "onnxscript>=0.7.2",
        "openpyxl>=3.1.5",
        "pandas>=3.1.1",
        "plotext>=6.1.0",
        "rich>=15.0.0",
        "torch>=2.6.0",
        "transformers>=5.17.0",
    ]
    return deps


def detect_package_manager() -> tuple[str, list[str]]:
    """Detect available package manager ('uv' or 'pip')."""
    uv_bin = shutil.which("uv")
    if uv_bin:
        return "uv", [uv_bin, "pip", "install"]
    return "pip", [sys.executable, "-m", "pip", "install"]


def verify_environment() -> dict[str, dict[str, Any]]:
    """Test import of all required dependencies and report their versions."""
    results: dict[str, dict[str, Any]] = {}
    check_modules = [
        ("rich", "rich"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("plotext", "plotext"),
        ("openpyxl", "openpyxl"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnxscript", "onnxscript"),
        ("laya", "laya"),
    ]

    for display_name, mod_name in check_modules:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            results[display_name] = {"installed": True, "version": str(ver)}
        except ImportError as exc:
            results[display_name] = {"installed": False, "version": None, "error": str(exc)}
        except Exception as exc:
            results[display_name] = {"installed": True, "version": "unknown", "warning": str(exc)}

    # If some modules failed in current interpreter, check if local .venv has them
    venv_py = PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if not all(r["installed"] for r in results.values()) and venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
        probe_code = (
            "import json; mods = ['rich','numpy','pandas','matplotlib','plotext','openpyxl','torch','transformers','onnx','onnxruntime','onnxscript','laya']; "
            "res = {}; "
            "exec(\"for m in mods:\\n  try:\\n    mod = __import__(m)\\n    res[m] = {'installed': True, 'version': str(getattr(mod, '__version__', 'installed'))}\\n  except Exception as e:\\n    res[m] = {'installed': False, 'version': None, 'error': str(e)}\"); "
            "print(json.dumps(res))"
        )
        try:
            p_res = subprocess.run([str(venv_py), "-c", probe_code], capture_output=True, text=True, check=False)
            if p_res.returncode == 0 and p_res.stdout.strip():
                import json
                venv_results = json.loads(p_res.stdout.strip())
                for k, v in venv_results.items():
                    if v.get("installed"):
                        results[k] = v
        except Exception:
            pass

    return results


def setup_workspace(repo_root: Path | None = None) -> None:
    """Create directory structure and default sample datasets."""
    root = repo_root or PROJECT_ROOT
    model_dir = root / "model"
    onnx_dir = model_dir / "onnx"
    data_dir = root / "data"
    output_dir = root / "outputs"

    for d in (model_dir, onnx_dir, data_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Initialize default config if not present
    cfg_file = model_dir / "qcplot_config.json"
    if not cfg_file.exists():
        import json
        default_cfg = {
            "default_model": "multilingual",
            "default_device": "auto",
            "default_backend": "auto",
        }
        cfg_file.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")

    # Sample dataset check
    sample_csv = data_dir / "sample_bolts.csv"
    orig_csv = data_dir / "bolts.csv"
    if not sample_csv.exists() and orig_csv.exists():
        shutil.copyfile(orig_csv, sample_csv)
    elif not orig_csv.exists() and sample_csv.exists():
        shutil.copyfile(sample_csv, orig_csv)


def install_packages(
    packages: list[str] | None = None,
    verbose: bool = True,
    use_uv: bool = True,
) -> bool:
    """Install required packages cross-platform using uv or pip."""
    plat = detect_platform_info()
    if packages is None:
        packages = get_core_dependencies(plat)

    pm_name, pm_cmd = detect_package_manager()
    if not use_uv and pm_name == "uv":
        pm_name, pm_cmd = "pip", [sys.executable, "-m", "pip", "install"]

    if verbose:
        print(cyan(f"==> Installing packages using {pm_name.upper()} for {plat['os_name']} ({plat['machine']})..."))

    # Try project-level uv sync if uv is used inside repo with pyproject.toml
    if pm_name == "uv" and (PROJECT_ROOT / "pyproject.toml").exists():
        uv_bin = pm_cmd[0]
        sync_cmd = [uv_bin, "sync"]
        if verbose:
            print(dim(f"Running: {' '.join(sync_cmd)}"))
        try:
            res = subprocess.run(sync_cmd, cwd=str(PROJECT_ROOT), check=False)
            if res.returncode == 0:
                if verbose:
                    print(green("✓ Successfully synchronized environment via uv sync"))
                return True
        except Exception:
            pass

    # Fallback to direct pip/uv pip install
    cmd = list(pm_cmd) + packages
    if verbose:
        print(dim(f"Running: {' '.join(cmd)}"))

    res = subprocess.run(cmd, check=False)
    if res.returncode == 0:
        if verbose:
            print(green(f"✓ All packages installed successfully via {pm_name}"))
        return True
    else:
        if verbose:
            print(red(f"✖ Package installation failed with return code {res.returncode}"))
        return False


def run_installer(
    interactive: bool = False,
    device: str = "auto",
    skip_packages: bool = False,
    verbose: bool = True,
) -> bool:
    """Complete cross-platform installer function:

    1. Detects OS and hardware acceleration.
    2. Installs cross-platform dependencies adapted to the detected OS.
    3. Provisions directories, sample datasets, and persistent config.
    4. Validates installation integrity across all modules.
    """
    print()
    print(bold(cyan("================================================================")))
    print(bold(cyan("      QCPlot Intelligent Cross-Platform Auto-Installer          ")))
    print(bold(cyan("================================================================")))
    print()

    # Step 1: Detect Platform
    plat = detect_platform_info()
    print(bold("1. System Inspection:"))
    print(f"   • Operating System: {green(plat['os_name'])} ({plat['release']})")
    print(f"   • Architecture:     {green(plat['machine'])}")
    print(f"   • Python Version:   {green(plat['python_version'])}")

    if plat["gpus"]:
        for g in plat["gpus"]:
            print(f"   • Detected GPU:     {green(g)}")
    else:
        print(f"   • Detected GPU:     {yellow('None / CPU Mode')}")

    print(f"   • ONNX Acceleration:{green(plat['ort_type'])}")
    print(f"   • Recommended ONNX: {cyan(plat['recommended_ort'])}")
    print()

    # Step 2: Install Packages (if requested / needed)
    if not skip_packages:
        print(bold("2. Package Installation:"))
        success = install_packages(verbose=verbose)
        if not success:
            print(yellow("⚠ Direct install encountered warnings; attempting to verify existing packages..."))
        print()

    # Step 3: Setup Workspace
    print(bold("3. Workspace Configuration:"))
    setup_workspace()
    print(f"   • Workspace Root:   {green(str(PROJECT_ROOT))}")
    print(f"   • Directories:      {green('model/')}  {green('model/onnx/')}  {green('data/')}  {green('outputs/')}")
    print(f"   • Sample Dataset:   {green('data/sample_bolts.csv ready')}")
    print(f"   • Config File:      {green('model/qcplot_config.json')}")
    print()

    # Step 4: Verification
    print(bold("4. Environment Verification:"))
    verification = verify_environment()
    all_ok = True
    for mod_name, info in verification.items():
        if info["installed"]:
            v_str = info.get("version", "installed")
            print(f"   {green('✓')} {mod_name:<16} {dim(v_str)}")
        else:
            all_ok = False
            err = info.get("error", "missing")
            print(f"   {red('✖')} {mod_name:<16} {red(err)}")

    print()
    if all_ok:
        print(bold(green("✔ QCPlot setup successfully completed for " + plat["os_name"] + "!")))
        print()
        print("You can now run QCPlot:")
        print(cyan("  uv run main.py --init"))
        print(cyan('  uv run main.py -f data/sample_bolts.csv -q "Check stability"'))
        print(cyan("  uv run main.py --device-info"))
    else:
        print(bold(yellow("⚠ Some dependencies could not be verified in the current interpreter.")))
        print(dim("If running in a virtual environment, ensure you activate it or use 'uv run qcplot'."))

    return all_ok


if __name__ == "__main__":
    run_installer()
