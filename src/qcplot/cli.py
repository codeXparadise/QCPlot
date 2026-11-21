from __future__ import annotations

import time

SCRIPT_START = time.perf_counter()

import argparse
import contextlib
import functools
import inspect
import io
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Detect workspace root or fallback to working directory
_repo_root = Path(__file__).resolve().parent.parent.parent
if (_repo_root / "pyproject.toml").exists():
    PROJECT_ROOT = _repo_root
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-relaunch inside local .venv if invoked with system python missing dependencies
_venv_python = PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
if _venv_python.exists() and sys.executable.lower() != str(_venv_python).lower():
    try:
        import rich
    except ImportError:
        import subprocess
        _launcher = PROJECT_ROOT / "main.py"
        if not _launcher.exists():
            _launcher = Path(__file__).resolve()
        _res = subprocess.run([str(_venv_python), str(_launcher)] + sys.argv[1:])
        sys.exit(_res.returncode)

# ---------------------------------------------------------------------------
# 1. Directories, Persistent Config & Model Catalog (100% Cross-Platform)
# ---------------------------------------------------------------------------
os.system("")  # Enable ANSI color escape sequences on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
MODEL_DIR = PROJECT_ROOT / "model"
ONNX_DIR = MODEL_DIR / "onnx"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_FILE = MODEL_DIR / "qcplot_config.json"
LEGACY_CONFIG_FILE = MODEL_DIR / "qcc_config.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_DEFAULT_MODEL = "multilingual"
WEIGHT_EXTS = ("*.safetensors", "*.bin", "*.pt")


@dataclass(frozen=True)
class ModelMeta:
    key: str
    repo: str
    params: str
    size_hint: str
    description: str
    subfolder: str | None = None


AVAILABLE_MODELS: dict[str, ModelMeta] = {
    "multilingual": ModelMeta(
        key="multilingual",
        repo="convaiinnovations/laya-multilingual",
        params="322M",
        size_hint="~650 MB",
        description="100+ languages, Hinglish, 8k ctx (Recommended)",
        subfolder="multilingual",
    ),
    "english": ModelMeta(
        key="english",
        repo="convaiinnovations/laya",
        params="421M",
        size_hint="~843 MB",
        description="English-only short prompts (512 ctx)",
        subfolder=None,
    ),
    "typed-decisions": ModelMeta(
        key="typed-decisions",
        repo="convaiinnovations/laya-typed-decisions",
        params="421M",
        size_hint="~846 MB",
        description="Structured multi-field workflows (1024 ctx)",
        subfolder="typed-decisions",
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "laya": "english",
    "convaiinnovations/laya": "english",
    "laya-multilingual": "multilingual",
    "convaiinnovations/laya-multilingual": "multilingual",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "convaiinnovations/laya-typed-decisions": "typed-decisions",
}

ONNX_GPU_PROVIDERS: list[tuple[str, str, str]] = [
    ("CUDAExecutionProvider", "cuda", "ONNX CUDA GPU"),
    ("ROCMExecutionProvider", "rocm", "ONNX ROCm GPU"),
    ("DmlExecutionProvider", "directml", "ONNX DirectML GPU"),
    ("CoreMLExecutionProvider", "coreml", "ONNX Apple CoreML"),
    ("OpenVINOExecutionProvider", "openvino", "ONNX Intel OpenVINO"),
]


def load_config() -> dict[str, str]:
    defaults = {
        "default_model": FALLBACK_DEFAULT_MODEL,
        "default_device": "auto",
        "default_backend": "auto",
    }
    cfg_file = CONFIG_FILE if CONFIG_FILE.exists() else LEGACY_CONFIG_FILE
    if cfg_file.exists():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                defaults.update({k: str(v) for k, v in data.items() if v})
        except Exception:
            pass
    return defaults


def save_config(cfg: dict[str, str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def normalize_model_key(name: str) -> str:
    cleaned = name.strip().lower().replace("_", "-")
    if cleaned in AVAILABLE_MODELS or cleaned == "auto":
        return cleaned
    return MODEL_ALIASES.get(cleaned, cleaned)


def get_onnx_model_path(model_key: str) -> Path:
    key = normalize_model_key(model_key)
    if key == "auto":
        key = FALLBACK_DEFAULT_MODEL
    return ONNX_DIR / key / "model.onnx"


def is_onnx_exported(model_key: str) -> bool:
    p = get_onnx_model_path(model_key)
    return p.exists() and p.stat().st_size > 1024


def get_onnx_size_mb(model_key: str) -> float:
    folder = get_onnx_model_path(model_key).parent
    if not folder.exists():
        return 0.0
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _find_model_weight_files(model_key: str) -> list[Path]:
    key = normalize_model_key(model_key)
    if key == "auto":
        return [p for ext in WEIGHT_EXTS for p in MODEL_DIR.rglob(ext)]

    meta = AVAILABLE_MODELS.get(key)
    if meta is None:
        return []

    found: list[Path] = []
    dedicated_slug = "models--" + meta.repo.replace("/", "--")
    for repo_dir in MODEL_DIR.rglob(dedicated_slug):
        if repo_dir.is_dir():
            for ext in WEIGHT_EXTS:
                found.extend(repo_dir.rglob(ext))

    if not found:
        for repo_dir in MODEL_DIR.rglob("models--convaiinnovations--laya"):
            if not repo_dir.is_dir():
                continue
            if meta.subfolder:
                for sub in repo_dir.glob(f"snapshots/*/{meta.subfolder}"):
                    for ext in WEIGHT_EXTS:
                        found.extend(sub.rglob(ext))
            else:
                for snap in repo_dir.glob("snapshots/*"):
                    for ext in WEIGHT_EXTS:
                        found.extend(snap.glob(ext))

    unique: dict[Path, Path] = {}
    for p in found:
        try:
            unique[p.resolve()] = p
        except OSError:
            unique[p] = p
    return list(unique.values())


def locate_checkpoint_dir(model_key: str) -> Path | None:
    """Finds the cached directory containing rl_agent_config.json for ONNXAgent."""
    key = normalize_model_key(model_key)
    meta = AVAILABLE_MODELS.get(key)
    if meta is None:
        return None

    dedicated_slug = "models--" + meta.repo.replace("/", "--")
    for repo_dir in MODEL_DIR.rglob(dedicated_slug):
        for cfg in repo_dir.rglob("rl_agent_config.json"):
            return cfg.parent

    for repo_dir in MODEL_DIR.rglob("models--convaiinnovations--laya"):
        if meta.subfolder:
            for sub in repo_dir.glob(f"snapshots/*/{meta.subfolder}"):
                if (sub / "rl_agent_config.json").exists():
                    return sub
        else:
            for snap in repo_dir.glob("snapshots/*"):
                if (snap / "rl_agent_config.json").exists():
                    return snap
    return None


def is_model_cached(model_choice: str) -> bool:
    return len(_find_model_weight_files(model_choice)) > 0


def get_cached_size_mb(model_choice: str) -> float:
    files = _find_model_weight_files(model_choice)
    total_bytes = sum(f.stat().st_size for f in files if f.exists())
    return total_bytes / (1024 * 1024)


def configure_environment(model_choice: str, force_online: bool = False) -> bool:
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODEL_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cached = is_model_cached(model_choice)
    if force_online:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    elif cached and os.environ.get("HF_HUB_OFFLINE") != "0":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    warnings.filterwarnings("ignore")
    for log_name in (
        "transformers",
        "huggingface_hub",
        "torch",
        "laya",
        "onnxruntime",
        "onnxscript",
        "urllib3",
        "httpx",
        "filelock",
        "matplotlib",
    ):
        logging.getLogger(log_name).setLevel(logging.ERROR)
    return cached


from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(file=sys.__stdout__, highlight=False)
ACCENT = "#D97757"


@contextlib.contextmanager
def quiet_backend():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


# Lazily imported so --help / --models / invalid flags respond instantly
plt: Any = None
np: Any = None
pd: Any = None
torch: Any = None
ort: Any = None
RUNTIME_INIT_SECS: float = 0.0


def _parse_ver_tuple(ver_str: str) -> tuple[int, int]:
    nums = re.findall(r"\d+", ver_str)
    major = int(nums[0]) if len(nums) >= 1 else 0
    minor = int(nums[1]) if len(nums) >= 2 else 0
    return major, minor


def init_backends_only() -> tuple[str, str, str]:
    """Fast lightweight import for --device-info without matplotlib/pandas."""
    global torch, ort
    with console.status("[dim]Inspecting compute devices & ONNX providers...[/dim]", spinner="dots", spinner_style="cyan"):
        with quiet_backend():
            import torch as _torch

            try:
                import onnxruntime as _ort
                ort_ver = getattr(_ort, "__version__", "installed")
            except ImportError:
                _ort = None
                ort_ver = "not installed"

            try:
                import transformers as _tf
                tf_ver = getattr(_tf, "__version__", "not installed")
            except Exception:
                tf_ver = "not installed"

    torch, ort = _torch, _ort
    return getattr(torch, "__version__", "unknown"), str(tf_ver), str(ort_ver)


def init_runtime() -> None:
    global plt, np, pd, torch, ort, RUNTIME_INIT_SECS
    t0 = time.perf_counter()
    with console.status("[dim]Initializing runtime...[/dim]", spinner="dots", spinner_style="cyan"):
        with quiet_backend():
            import numpy as _np
            import pandas as _pd
            import torch as _torch

            try:
                import onnxruntime as _ort
            except ImportError:
                _ort = None

            try:
                import transformers as _tf
                tf_ver = getattr(_tf, "__version__", "0.0")
            except Exception:
                tf_ver = "0.0"

    np, pd, torch, ort = _np, _pd, _torch, _ort
    RUNTIME_INIT_SECS = time.perf_counter() - t0

    torch_v = _parse_ver_tuple(getattr(torch, "__version__", "0.0"))
    tf_v = _parse_ver_tuple(tf_ver)
    if torch_v < (2, 6) and tf_v >= (5, 0):
        msg = Text()
        msg.append("Incompatible dependency versions detected:\n", style="bold red")
        msg.append(f"  torch == {torch.__version__}  ×  transformers == {tf_ver}\n\n", style="white")
        msg.append("Run this to fix:\n", style="dim")
        msg.append('  uv add "torch>=2.6.0" "transformers>=5.17.0"', style="bold cyan")
        console.print(
            Panel(
                msg,
                title="[bold red]Dependency Fix Required[/bold red]",
                box=box.ROUNDED,
                border_style="red",
                expand=False,
            )
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Custom Rich Help Menu + Auto-Help on Any Wrong Flag
# ---------------------------------------------------------------------------
def render_rich_help(error_msg: str | None = None) -> None:
    cfg = load_config()
    def_model = cfg.get("default_model", FALLBACK_DEFAULT_MODEL)
    def_dev = cfg.get("default_device", "auto")
    def_backend = cfg.get("default_backend", "auto")

    if error_msg:
        console.print(
            Panel(
                f"[bold red]✖ {error_msg}[/bold red]",
                box=box.ROUNDED,
                border_style="red",
                padding=(0, 2),
                expand=False,
            )
        )

    title = Text()
    title.append("Usage: ", style="bold cyan")
    title.append("uv run qcplot [OPTIONS]", style="bold white")
    title.append("  (or .\\qcplot [OPTIONS])\n", style="dim")
    title.append(f"Defaults: Model={def_model} · Backend={def_backend} · Device={def_dev}", style="dim")

    console.print(Panel(title, box=box.ROUNDED, border_style="bright_black", padding=(0, 2), expand=False))

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Core Execution",
            [
                ("-f, --file PATH", "Path to input CSV, TSV, or Excel (.xlsx/.xls) tabular data file"),
                ("-q, --question TEXT", "Natural language QC question (English, Hinglish, or 100+ languages)"),
                ("--interactive", "Keep model loaded in memory for instant back-to-back queries"),
                ("-o, --output PATH", "Custom output file path for the high-res PNG chart"),
            ],
        ),
        (
            "Model Management",
            [
                ("--model NAME", f"Select checkpoint: multilingual | english | typed-decisions | auto (default: {def_model})"),
                ("--models", "List all models, PyTorch cache, ONNX export status, disk size, and default"),
                ("--pull NAME", "Download & cache a Laya model into ./model directly from terminal"),
                ("--export-onnx NAME", "Export a cached model to ONNX for cross-platform GPU acceleration"),
                ("--set-default NAME", "Persistently change the default model used on every run"),
                ("-r, --remove NAME", "Delete a downloaded model or 'all' to free up disk space"),
            ],
        ),
        (
            "Setup & Hardware",
            [
                ("--install", "Auto-detect OS, GPU & install dependencies cross-platform"),
                ("--init", "First-time setup (folders, config, multilingual pull, ONNX GPU export, sample CSV)"),
                ("--backend MODE", "Inference engine: auto | onnx | torch (auto picks GPU first)"),
                ("--device MODE", "Compute target: auto | cuda | rocm | directml | mps | coreml | xpu | cpu"),
                ("--device-info", "Inspect CPU, PyTorch GPUs, ONNX Runtime providers, and active default"),
            ],
        ),
        (
            "Display & Chart Options",
            [
                ("--max-rows N", "Maximum table rows shown before smart head/tail truncation (default: 12)"),
                ("--full", "Display every subgroup row in the terminal table without truncation"),
                ("--popup", "Also open the interactive Matplotlib window in addition to saving PNG"),
                ("-h, --help", "Show this formatted help guide and demo commands"),
            ],
        ),
    ]

    for sec_title, rows in sections:
        tbl = Table(
            title=f"[bold {ACCENT}]{sec_title}[/bold {ACCENT}]",
            title_justify="left",
            box=box.ROUNDED,
            border_style="bright_black",
            show_header=False,
            padding=(0, 1),
            expand=True,
        )
        tbl.add_column("Flag", style="bold cyan", width=22, no_wrap=True)
        tbl.add_column("Description", style="white")
        for flag, desc in rows:
            tbl.add_row(flag, desc)
        console.print(tbl)

    demo = Text()
    demo.append("# 1. First-time setup (pulls multilingual + auto-exports ONNX if GPU provider found)\n", style="dim")
    demo.append("uv run qcplot --init\n\n", style="bold green")
    demo.append("# 2. Check GPU/ONNX providers & installed models\n", style="dim")
    demo.append("uv run qcplot --device-info\n", style="cyan")
    demo.append("uv run qcplot --models\n\n", style="cyan")
    demo.append("# 3. Pull or export another model / remove or switch default\n", style="dim")
    demo.append("uv run qcplot --pull english\n", style="cyan")
    demo.append("uv run qcplot --export-onnx multilingual\n", style="cyan")
    demo.append("uv run qcplot --set-default multilingual\n", style="cyan")
    demo.append("uv run qcplot --remove english  # (or: uv run qcplot --remove all)\n\n", style="cyan")
    demo.append("# 4. Run QCPlot analysis (auto GPU via ONNX DirectML / CUDA / ROCm / MPS)\n", style="dim")
    demo.append('uv run qcplot -f data/sample_bolts.csv -q "Bolt diameter stability check karo"\n', style="bold white")
    demo.append("uv run qcplot --interactive", style="bold white")

    console.print(
        Panel(
            demo,
            title="[bold]Demo Commands[/bold]",
            title_align="left",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )


class RichHelpParser(argparse.ArgumentParser):
    def print_help(self, file: Any = None) -> None:
        render_rich_help()

    def error(self, message: str) -> None:
        render_rich_help(error_msg=message)
        self.exit(2)


# ---------------------------------------------------------------------------
# 3. SPC Constants & Data Structures
# ---------------------------------------------------------------------------
SPC_CONSTANTS: dict[int, tuple[float, float, float, float]] = {
    2: (1.880, 0.000, 3.267, 1.128),
    3: (1.023, 0.000, 2.574, 1.693),
    4: (0.729, 0.000, 2.282, 2.059),
    5: (0.577, 0.000, 2.114, 2.326),
    6: (0.483, 0.000, 2.004, 2.534),
    7: (0.419, 0.076, 1.924, 2.704),
    8: (0.373, 0.136, 1.864, 2.847),
    9: (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
    11: (0.285, 0.256, 1.744, 3.173),
    12: (0.266, 0.283, 1.717, 3.258),
    13: (0.249, 0.307, 1.693, 3.336),
    14: (0.235, 0.328, 1.672, 3.407),
    15: (0.223, 0.347, 1.653, 3.472),
}


def c4_constant(n: int) -> float:
    if n < 2:
        return 1.0
    log_c4 = (
        0.5 * math.log(2.0 / (n - 1))
        + math.lgamma(n / 2.0)
        - math.lgamma((n - 1) / 2.0)
    )
    return math.exp(log_c4)


@dataclass
class DeviceInfo:
    runtime: str              # "onnx" or "torch"
    backend: str              # "directml", "cuda", "rocm", "mps", "coreml", "xpu", "openvino", "cpu"
    torch_device: Any         # "cuda", "mps", "xpu", "cpu"
    ort_provider: str | None
    hardware_name: str
    note: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"

    @property
    def label(self) -> str:
        rt = "ONNX" if self.runtime == "onnx" else "TORCH"
        if self.is_gpu:
            return f"{rt} · {self.backend.upper()} ({self.hardware_name})"
        return f"{rt} · CPU ({self.hardware_name})"


@dataclass
class RunStats:
    model_used: str
    device_label: str
    is_gpu: bool
    runtime_init_s: float
    model_load_s: float
    data_load_s: float
    laya_infer_s: float
    math_solve_s: float
    chart_render_s: float
    query_total_s: float
    script_total_s: float
    subgroups: int


@dataclass
class PanelSpec:
    ylabel: str
    x: Any
    y: Any
    cl: float
    ucl: Any
    lcl: Any
    cl_label: str


@dataclass
class ReportSpec:
    chart_name: str
    problem_category: str
    table_headers: list[str]
    table_rows: list[list[str]]
    constants: list[tuple[str, str]]
    formulas_and_math: list[tuple[str, str, str, str]]
    ooc_summary: list[tuple[str, bool, str]]


@dataclass
class ChartSpec:
    title: str
    subtitle: str = "Investigate any out-of-control subgroups."
    panels: list[PanelSpec] = field(default_factory=list)
    footer_stats: list[str] = field(default_factory=list)
    footer_note: str = ""
    report: ReportSpec | None = None


def _find_ooc_indices(panel: PanelSpec) -> list[Any]:
    y = np.asarray(panel.y, dtype=float)
    ucl = np.broadcast_to(np.asarray(panel.ucl, dtype=float), y.shape)
    lcl = np.broadcast_to(np.asarray(panel.lcl, dtype=float), y.shape)
    valid = ~np.isnan(y)
    mask = valid & ((y > ucl) | (y < lcl))
    ooc_pts: list[Any] = []
    for i in np.where(mask)[0]:
        val = panel.x[i]
        try:
            f = float(val)
            ooc_pts.append(int(f) if f.is_integer() else round(f, 2))
        except (ValueError, TypeError):
            ooc_pts.append(str(val))
    return ooc_pts


# ---------------------------------------------------------------------------
# 4. Cross-Platform GPU-First Resolver (ONNX Runtime + PyTorch)
# ---------------------------------------------------------------------------
def get_available_ort_providers() -> list[str]:
    if ort is None:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def configure_torch_performance(dev: DeviceInfo) -> None:
    if dev.runtime != "torch":
        return
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if dev.backend in ("cuda", "rocm"):
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def resolve_device(requested_device: str = "auto", requested_backend: str = "auto") -> DeviceInfo:
    cpu_name = platform.processor() or platform.machine() or "CPU"
    ort_provs = get_available_ort_providers()

    def _pick_onnx_gpu() -> DeviceInfo | None:
        if not ort_provs:
            return None
        for prov, short_b, disp in ONNX_GPU_PROVIDERS:
            if prov in ort_provs and requested_device in ("auto", short_b):
                return DeviceInfo(
                    runtime="onnx",
                    backend=short_b,
                    torch_device="cpu",
                    ort_provider=prov,
                    hardware_name=disp,
                )
        return None

    def _pick_torch_gpu() -> DeviceInfo | None:
        if requested_device in ("auto", "cuda", "rocm") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            is_rocm = bool(getattr(torch.version, "hip", None))
            b = "rocm" if is_rocm else "cuda"
            if requested_device in ("auto", b, "cuda"):
                dev = DeviceInfo(
                    runtime="torch",
                    backend=b,
                    torch_device="cuda",
                    ort_provider=None,
                    hardware_name=name,
                )
                configure_torch_performance(dev)
                return dev
        if requested_device in ("auto", "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = DeviceInfo(
                runtime="torch",
                backend="mps",
                torch_device="mps",
                ort_provider=None,
                hardware_name="Apple Silicon GPU",
            )
            configure_torch_performance(dev)
            return dev
        if requested_device in ("auto", "xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "Intel XPU"
            dev = DeviceInfo(
                runtime="torch",
                backend="xpu",
                torch_device="xpu",
                ort_provider=None,
                hardware_name=name,
            )
            configure_torch_performance(dev)
            return dev
        return None

    if requested_backend == "onnx":
        if ort is None:
            return DeviceInfo(
                runtime="torch",
                backend="cpu",
                torch_device="cpu",
                ort_provider=None,
                hardware_name=cpu_name,
                note="onnxruntime not installed; fell back to PyTorch CPU",
            )
        if requested_device != "cpu":
            gpu_dev = _pick_onnx_gpu()
            if gpu_dev:
                return gpu_dev
        return DeviceInfo(
            runtime="onnx",
            backend="cpu",
            torch_device="cpu",
            ort_provider="CPUExecutionProvider",
            hardware_name=cpu_name,
        )

    if requested_backend == "torch":
        if requested_device != "cpu":
            gpu_dev = _pick_torch_gpu()
            if gpu_dev:
                return gpu_dev
        note = ""
        if requested_device not in ("auto", "cpu"):
            note = f"PyTorch '{requested_device}' unavailable; falling back to CPU"
        return DeviceInfo(
            runtime="torch",
            backend="cpu",
            torch_device="cpu",
            ort_provider=None,
            hardware_name=cpu_name,
            note=note,
        )

    # AUTO backend: GPU-first across PyTorch native GPUs and ONNX GPU providers
    if requested_device != "cpu":
        if requested_device in ("directml", "coreml", "openvino"):
            onnx_gpu = _pick_onnx_gpu()
            if onnx_gpu:
                return onnx_gpu
        else:
            torch_gpu = _pick_torch_gpu()
            if torch_gpu:
                return torch_gpu
            onnx_gpu = _pick_onnx_gpu()
            if onnx_gpu:
                return onnx_gpu

    note = ""
    if requested_device not in ("auto", "cpu"):
        note = f"Requested device '{requested_device}' unavailable; falling back to CPU"
    return DeviceInfo(
        runtime="torch",
        backend="cpu",
        torch_device="cpu",
        ort_provider=None,
        hardware_name=cpu_name,
        note=note,
    )


def cmd_device_info() -> None:
    torch_ver, tf_ver, ort_ver = init_backends_only()
    cfg = load_config()
    selected = resolve_device(
        requested_device=cfg.get("default_device", "auto"),
        requested_backend=cfg.get("default_backend", "auto"),
    )

    cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
    cpu_cores = os.cpu_count() or 1
    ort_provs = get_available_ort_providers()

    cuda_ok = bool(torch.cuda.is_available())
    cuda_detail = "-"
    if cuda_ok:
        try:
            props = torch.cuda.get_device_properties(0)
            cuda_detail = f"{props.name} ({props.total_memory / (1024**3):.1f} GB VRAM)"
        except Exception:
            cuda_detail = torch.cuda.get_device_name(0)

    mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    xpu_ok = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    xpu_detail = torch.xpu.get_device_name(0) if xpu_ok and hasattr(torch.xpu, "get_device_name") else "Not available"

    tbl = Table(
        title="[bold cyan]Compute Backends (ONNX Runtime + PyTorch)[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold white",
        padding=(0, 1),
    )
    tbl.add_column("Engine / Backend", style="bold cyan", no_wrap=True)
    tbl.add_column("Status", justify="center", no_wrap=True)
    tbl.add_column("Hardware / Provider Details", style="white")
    tbl.add_column("Default", justify="center", no_wrap=True)

    rows: list[tuple[str, bool, str, bool]] = []
    for prov, short_b, disp in ONNX_GPU_PROVIDERS:
        ok = prov in ort_provs
        is_sel = selected.runtime == "onnx" and selected.ort_provider == prov
        detail = prov if ok else ("onnxruntime not installed" if ort is None else "Provider not available")
        rows.append((disp, ok, detail, is_sel))

    rows.extend([
        ("PyTorch CUDA / ROCm", cuda_ok, cuda_detail if cuda_ok else "Not available", selected.runtime == "torch" and selected.backend in ("cuda", "rocm")),
        ("PyTorch Apple MPS", mps_ok, "Apple Silicon GPU" if mps_ok else "Not available", selected.runtime == "torch" and selected.backend == "mps"),
        ("PyTorch Intel XPU", xpu_ok, xpu_detail, selected.runtime == "torch" and selected.backend == "xpu"),
        ("ONNX CPU", "CPUExecutionProvider" in ort_provs, "CPUExecutionProvider" if "CPUExecutionProvider" in ort_provs else "Not installed", selected.runtime == "onnx" and selected.backend == "cpu"),
        ("PyTorch CPU", True, f"{cpu_name} ({cpu_cores} logical cores)", selected.runtime == "torch" and selected.backend == "cpu"),
    ])

    for name, ok, detail, is_sel in rows:
        status_txt = "[green]✓ Available[/green]" if ok else "[dim]○ Unavailable[/dim]"
        sel_txt = f"[bold {ACCENT}]★ SELECTED[/bold {ACCENT}]" if is_sel else "[dim]—[/dim]"
        tbl.add_row(name, status_txt, detail, sel_txt)

    env_summary = Text()
    env_summary.append("OS / Python   : ", style="dim")
    env_summary.append(f"{platform.system()} {platform.release()} · Python {platform.python_version()}\n", style="white")
    env_summary.append("ML Runtime    : ", style="dim")
    env_summary.append(f"torch {torch_ver} · transformers {tf_ver} · onnxruntime {ort_ver}\n", style="cyan")
    env_summary.append("Active Default: ", style="dim")
    env_summary.append(selected.label, style="bold green" if selected.is_gpu else "bold yellow")
    if not selected.is_gpu:
        env_summary.append(
            "\nGPU Tip       : Windows AMD/Intel → uv add onnx onnxscript onnxruntime-directml  |  "
            "NVIDIA → uv add onnx onnxscript onnxruntime-gpu  |  Linux AMD → PyTorch ROCm",
            style="dim yellow",
        )

    console.print(Panel(env_summary, title="[bold]System & Default Device[/bold]", title_align="left", box=box.ROUNDED, border_style="bright_black", padding=(0, 2), expand=False))
    console.print(tbl)


# ---------------------------------------------------------------------------
# 5. ONNX Exporter (Auto-Matches ONNXAgent Expected Output Names)
# ---------------------------------------------------------------------------
def _find_loaded_agents(router: Any) -> list[Any]:
    found: list[Any] = []
    seen: set[int] = set()
    queue: list[Any] = [router]
    for val in vars(router).values():
        if isinstance(val, dict):
            queue.extend(val.values())
        elif isinstance(val, (list, tuple, set)):
            queue.extend(val)
        else:
            queue.append(val)
    for obj in queue:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        mod = getattr(obj, "model", None)
        if mod is not None and isinstance(mod, torch.nn.Module):
            found.append(obj)
    return found


@functools.lru_cache(maxsize=1)
def _detect_onnx_agent_output_names() -> list[str]:
    """Reads the exact output tensor names requested by laya.onnx_agent.ONNXAgent."""
    try:
        import laya.onnx_agent as oa

        for attr in ("OUTPUT_NAMES", "ONNX_OUTPUT_NAMES", "_OUTPUT_NAMES"):
            val = getattr(oa, attr, None)
            if isinstance(val, (list, tuple)) and val and all(isinstance(x, str) for x in val):
                return list(val)

        src = inspect.getsource(oa)
        for match in re.finditer(r"\.run\(\s*\[([^\]]+)\]", src):
            names = re.findall(r"""['"]([A-Za-z0-9_]+)['"]""", match.group(1))
            if names:
                return names
    except Exception:
        pass
    return ["logits", "act_logits"]


def sanitize_onnx_model_for_hardware(onnx_path: Path) -> bool:
    """Sanitizes an ONNX model for DirectML and cross-platform hardware providers.

    DirectML's DmlExecutionProvider has an issue with opset 14+ Reshape operators
    where allowzero=1 combined with dynamic/negative dimensions (-1) triggers
    DirectX error 80070057 ('The parameter is incorrect').
    Converting allowzero to 0 resolves the issue across all execution providers
    without altering numerical semantics or model precision.
    """
    if not onnx_path.exists():
        return False
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
        modified = False
        for node in model.graph.node:
            if node.op_type == "Reshape":
                for attr in node.attribute:
                    if attr.name == "allowzero" and attr.i == 1:
                        attr.i = 0
                        modified = True
        if modified:
            onnx.save(model, str(onnx_path))
            return True
    except Exception as exc:
        log.debug(f"DirectML ONNX graph sanitization skipped/failed: {exc}")
    return False


def export_model_to_onnx(model_key: str, force: bool = False) -> Path:
    """Exports a cached Laya PyTorch checkpoint to ./model/onnx/<key>/model.onnx."""
    key = normalize_model_key(model_key)
    if key == "auto":
        key = FALLBACK_DEFAULT_MODEL
    if key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {', '.join(AVAILABLE_MODELS)}")

    out_path = get_onnx_model_path(key)
    if out_path.exists() and not force:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        for old_file in out_path.parent.glob("model.onnx*"):
            try:
                old_file.unlink()
            except OSError:
                pass

    t0 = time.perf_counter()

    with console.status(
        f"[dim]Exporting [bold cyan]{key}[/bold cyan] to ONNX (one-time GPU optimization)...[/dim]",
        spinner="dots",
        spinner_style=ACCENT,
    ):
        with quiet_backend():
            from laya import Router

            sig = inspect.signature(Router.__init__).parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
            kwargs: dict[str, Any] = {}
            if "preload" in sig or has_var_kw:
                kwargs["preload"] = False
            if "device" in sig or has_var_kw:
                kwargs["device"] = "cpu"
            if "cache_dir" in sig or has_var_kw:
                kwargs["cache_dir"] = str(MODEL_DIR)

            router = Router(**kwargs)
            if hasattr(router, "load"):
                router.load(key)

            agents = _find_loaded_agents(router)
            if not agents:
                raise RuntimeError(f"Could not locate PyTorch model weights inside Laya Router for '{key}'.")

            py_model = agents[0].model.to("cpu").eval()
            expected_names = _detect_onnx_agent_output_names()

            # Use batch=2, seq=32, options=4 so dynamic axes never specialize to 1
            dummy_inputs = (
                torch.full((2, 32), 10, dtype=torch.long),
                torch.ones((2, 32), dtype=torch.long),
                torch.tensor([[2, 6, 10, 14], [2, 6, 10, 14]], dtype=torch.long),
                torch.ones((2, 4), dtype=torch.bool),
                torch.zeros((2,), dtype=torch.long),
            )
            input_names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]

            with torch.no_grad():
                probe_out = py_model(*dummy_inputs)

            is_dict_out = isinstance(probe_out, dict)
            if is_dict_out:
                if all(k in probe_out for k in expected_names):
                    source_keys = list(expected_names)
                    output_names = list(expected_names)
                else:
                    source_keys = list(probe_out.keys())
                    output_names = (
                        list(expected_names)
                        if len(source_keys) == len(expected_names)
                        else list(source_keys)
                    )
                sample_outs = tuple(probe_out[k] for k in source_keys)
            elif isinstance(probe_out, (tuple, list)):
                source_keys = []
                if len(probe_out) >= len(expected_names):
                    output_names = list(expected_names)
                else:
                    output_names = [f"out_{i}" for i in range(len(probe_out))]
                sample_outs = tuple(probe_out[: len(output_names)])
            else:
                source_keys = []
                output_names = [expected_names[0]] if expected_names else ["logits"]
                sample_outs = (probe_out,)

            class _LayaOnnxWrapper(torch.nn.Module):
                def __init__(self, inner: Any, keys: list[str], n_out: int, dict_mode: bool):
                    super().__init__()
                    self.inner = inner
                    self.keys = keys
                    self.n_out = n_out
                    self.dict_mode = dict_mode

                def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
                    out = self.inner(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                    if self.dict_mode:
                        return tuple(out[k] for k in self.keys)
                    if isinstance(out, (tuple, list)):
                        return tuple(out[: self.n_out])
                    return (out,)

            wrapper = _LayaOnnxWrapper(py_model, source_keys, len(output_names), is_dict_out).eval()
            exported = False
            last_err: Exception | None = None

            # 1. Try modern dynamo ONNX exporter (opset 18)
            try:
                from torch.export import Dim

                b_dim = Dim("batch", min=1, max=64)
                s_dim = Dim("seq", min=4, max=8192)
                k_dim = Dim("options", min=1, max=64)
                dyn_shapes = {
                    "input_ids": {0: b_dim, 1: s_dim},
                    "attention_mask": {0: b_dim, 1: s_dim},
                    "marker_pos": {0: b_dim, 1: k_dim},
                    "marker_mask": {0: b_dim, 1: k_dim},
                    "qtype": {0: b_dim},
                }
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper,
                        dummy_inputs,
                        str(out_path),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=dyn_shapes,
                        opset_version=18,
                        dynamo=True,
                    )
                exported = out_path.exists()
            except Exception as exc:
                last_err = exc

            # 2. Fallback to TorchScript ONNX exporter
            if not exported:
                dyn_axes: dict[str, dict[int, str]] = {
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "marker_pos": {0: "batch", 1: "options"},
                    "marker_mask": {0: "batch", 1: "options"},
                    "qtype": {0: "batch"},
                }
                for name, tensor in zip(output_names, sample_outs):
                    axes = {0: "batch"}
                    if getattr(tensor, "ndim", 1) >= 2 and tensor.shape[1] == 4:
                        axes[1] = "options"
                    dyn_axes[name] = axes

                try:
                    with torch.no_grad():
                        torch.onnx.export(
                            wrapper,
                            dummy_inputs,
                            str(out_path),
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dyn_axes,
                            opset_version=17,
                            dynamo=False,
                        )
                    exported = out_path.exists()
                except Exception as exc2:
                    raise RuntimeError(f"ONNX export failed (dynamo: {last_err}; legacy: {exc2})") from exc2

            if exported:
                sanitize_onnx_model_for_hardware(out_path)

    elapsed = time.perf_counter() - t0
    size_mb = get_onnx_size_mb(key)
    console.print(
        f"[green]✓[/green] ONNX exported [bold cyan]{key}[/bold cyan] "
        f"[dim](outputs: {', '.join(output_names)}) → {out_path.relative_to(PROJECT_ROOT)} ({size_mb:.0f} MB · {elapsed:.1f}s)[/dim]"
    )
    return out_path


# ---------------------------------------------------------------------------
# 6. Model Management Commands (--models, --pull, --export-onnx, --set-default, --init)
# ---------------------------------------------------------------------------
def cmd_list_models() -> None:
    cfg = load_config()
    default_key = normalize_model_key(cfg.get("default_model", FALLBACK_DEFAULT_MODEL))

    tbl = Table(
        title="[bold cyan]Laya Models Catalog[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold white",
        padding=(0, 1),
    )
    tbl.add_column("Model", style="bold cyan", no_wrap=True)
    tbl.add_column("Params", justify="right", no_wrap=True)
    tbl.add_column("PyTorch Cache", justify="center", no_wrap=True)
    tbl.add_column("ONNX GPU Build", justify="center", no_wrap=True)
    tbl.add_column("Role", justify="center", no_wrap=True)
    tbl.add_column("Best For", style="white")

    installed_count = 0
    total_disk_mb = 0.0
    for key, meta in AVAILABLE_MODELS.items():
        cached = is_model_cached(key)
        if cached:
            installed_count += 1
            mb = get_cached_size_mb(key)
            total_disk_mb += mb
            inst_str = f"[green]✓ Local ({mb:.0f} MB)[/green]"
        else:
            inst_str = "[dim]○ Not pulled[/dim]"

        if is_onnx_exported(key):
            omb = get_onnx_size_mb(key)
            total_disk_mb += omb
            onnx_str = f"[green]✓ Ready ({omb:.0f} MB)[/green]"
        else:
            onnx_str = "[dim]○ Not exported[/dim]"

        role_str = f"[bold {ACCENT}]★ Default[/bold {ACCENT}]" if key == default_key else "[dim]—[/dim]"
        tbl.add_row(key, meta.params, inst_str, onnx_str, role_str, meta.description)

    console.print(tbl)
    console.print(
        f"[dim]Summary:[/dim] [bold white]{installed_count}/{len(AVAILABLE_MODELS)}[/bold white] installed "
        f"[dim]({total_disk_mb:.0f} MB total in ./model)  ·  Default:[/dim] [bold cyan]{default_key}[/bold cyan]"
    )
    console.print(
        "[dim]Commands:[/dim] [cyan]qcplot --pull <name>[/cyan]  ·  "
        "[cyan]qcplot --export-onnx <name>[/cyan]  ·  "
        "[cyan]qcplot --set-default <name>[/cyan]  ·  "
        "[cyan]qcplot --remove <name>[/cyan]\n"
    )


def cmd_set_default(model_name: str) -> None:
    key = normalize_model_key(model_name)
    if key not in AVAILABLE_MODELS and key != "auto":
        render_rich_help(error_msg=f"Unknown model '{model_name}'. Choose: {', '.join(AVAILABLE_MODELS)} or auto.")
        sys.exit(2)
    cfg = load_config()
    cfg["default_model"] = key
    save_config(cfg)
    console.print(f"[green]✓[/green] Default model set to [bold cyan]{key}[/bold cyan] [dim](saved in ./model/qcplot_config.json)[/dim]")
    if key != "auto" and not is_model_cached(key):
        console.print(f"  [dim yellow]ℹ Not downloaded yet. Run: [cyan]qcplot --pull {key}[/cyan][/dim yellow]")


def cmd_pull_model(model_name: str, device_req: str = "auto", backend_req: str = "auto") -> None:
    key = normalize_model_key(model_name)
    if key not in AVAILABLE_MODELS:
        render_rich_help(error_msg=f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}")
        sys.exit(2)

    configure_environment(key, force_online=True)
    init_runtime()
    dev = resolve_device(device_req, backend_req)

    meta = AVAILABLE_MODELS[key]
    t0 = time.perf_counter()
    with console.status(
        f"[dim]Pulling [bold cyan]{meta.repo}[/bold cyan] ({meta.size_hint}) into ./model...[/dim]",
        spinner="dots",
        spinner_style=ACCENT,
    ):
        with quiet_backend():
            from laya import Router

            sig = inspect.signature(Router.__init__).parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
            kwargs: dict[str, Any] = {}
            if "preload" in sig or has_var_kw:
                kwargs["preload"] = False
            if "cache_dir" in sig or has_var_kw:
                kwargs["cache_dir"] = str(MODEL_DIR)
            if "device" in sig or has_var_kw:
                kwargs["device"] = "cpu"
            r = Router(**kwargs)
            if hasattr(r, "load"):
                r.load(key)

    elapsed = time.perf_counter() - t0
    size_mb = get_cached_size_mb(key)
    console.print(
        f"[green]✓[/green] Pulled [bold cyan]{key}[/bold cyan] ({meta.repo}) "
        f"[dim]· {size_mb:.0f} MB cached · {elapsed:.1f}s[/dim]"
    )

    if dev.runtime == "onnx" and not is_onnx_exported(key):
        try:
            export_model_to_onnx(key)
        except Exception as exc:
            console.print(f"  [dim yellow]ℹ Skipped auto ONNX export ({exc}); PyTorch fallback remains ready.[/dim yellow]")

    dev_style = "bold green" if dev.is_gpu else "yellow"
    console.print(f"  [dim]Active compute target:[/dim] [{dev_style}]{dev.label}[/{dev_style}]\n")


SAMPLE_BOLTS_CSV = """Subgroup,M1,M2,M3,M4,M5
1,98.1,99.4,97.2,101.0,94.3
2,99.5,100.2,98.8,102.1,96.9
3,100.8,99.6,101.4,100.1,99.9
4,108.5,95.2,98.4,101.1,88.4
5,100.2,101.1,99.8,100.9,101.3
6,97.6,98.4,99.1,96.8,98.2
7,98.9,100.4,97.8,99.5,98.6
8,96.5,98.1,97.2,98.8,96.1
9,103.8,104.5,103.2,104.9,103.9
10,98.4,100.6,96.2,101.5,98.3
11,97.1,98.5,97.9,99.2,96.4
12,100.1,101.2,100.6,99.8,100.9
"""


def ensure_workspace_setup() -> None:
    """Auto-sets up directories and sample dataset on fresh clone or first run."""
    for folder in (MODEL_DIR, ONNX_DIR, DATA_DIR, OUTPUT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    sample_bolts = DATA_DIR / "sample_bolts.csv"
    orig_bolts = DATA_DIR / "bolts.csv"

    if not sample_bolts.exists():
        if orig_bolts.exists():
            try:
                shutil.copyfile(orig_bolts, sample_bolts)
            except Exception:
                sample_bolts.write_text(SAMPLE_BOLTS_CSV, encoding="utf-8")
        else:
            sample_bolts.write_text(SAMPLE_BOLTS_CSV, encoding="utf-8")

    if not orig_bolts.exists() and sample_bolts.exists():
        try:
            shutil.copyfile(sample_bolts, orig_bolts)
        except Exception:
            orig_bolts.write_text(SAMPLE_BOLTS_CSV, encoding="utf-8")


def _safe_rmtree(path: Path) -> int:
    """Recursively deletes a directory or file and returns total bytes freed."""
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            sz = path.stat().st_size
            path.unlink(missing_ok=True)
            return sz
        except Exception:
            return 0

    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                pass

    def _on_error(func: Any, p: str, exc_info: Any) -> None:
        try:
            os.chmod(p, 0o777)
            func(p)
        except Exception:
            pass

    try:
        shutil.rmtree(path, onerror=_on_error)
    except Exception:
        pass
    return total_bytes


def cmd_remove_model(model_target: str) -> None:
    target = model_target.strip().lower()
    if target in ("all", "purge", "*"):
        with console.status("[dim]Purging all downloaded models and ONNX builds...[/dim]", spinner="dots", spinner_style=ACCENT):
            freed_bytes = 0
            if ONNX_DIR.exists():
                freed_bytes += _safe_rmtree(ONNX_DIR)
                ONNX_DIR.mkdir(parents=True, exist_ok=True)

            for d in list(MODEL_DIR.glob("models--*")):
                freed_bytes += _safe_rmtree(d)

            for folder_name in ("hub", ".locks", "xet"):
                sub = MODEL_DIR / folder_name
                if sub.exists():
                    freed_bytes += _safe_rmtree(sub)

            harness_file = MODEL_DIR / ".agent_harnesses.json"
            if harness_file.exists():
                freed_bytes += _safe_rmtree(harness_file)

        mb = freed_bytes / (1024 * 1024)
        gb = mb / 1024
        size_lbl = f"{gb:.2f} GB" if gb >= 1.0 else f"{mb:.1f} MB"
        console.print(f"[green]✓[/green] Purged all cached models and ONNX builds [dim](freed {size_lbl})[/dim]")
        return

    key = normalize_model_key(target)
    if key not in AVAILABLE_MODELS:
        render_rich_help(
            error_msg=f"Unknown model '{model_target}'. Choose from: {', '.join(AVAILABLE_MODELS)}, or 'all'."
        )
        sys.exit(2)

    meta = AVAILABLE_MODELS[key]
    freed_bytes = 0

    with console.status(
        f"[dim]Removing model [bold cyan]{key}[/bold cyan] ({meta.repo})...[/dim]",
        spinner="dots",
        spinner_style=ACCENT,
    ):
        # 1. Remove ONNX build
        onnx_folder = ONNX_DIR / key
        if onnx_folder.exists():
            freed_bytes += _safe_rmtree(onnx_folder)

        # 2. Remove dedicated huggingface model cache if present
        dedicated_slug = "models--" + meta.repo.replace("/", "--")
        for repo_dir in list(MODEL_DIR.rglob(dedicated_slug)):
            if repo_dir.exists():
                freed_bytes += _safe_rmtree(repo_dir)

        # 3. If model was inside models--convaiinnovations--laya
        for repo_dir in list(MODEL_DIR.rglob("models--convaiinnovations--laya")):
            if not repo_dir.exists():
                continue
            if meta.subfolder:
                for sub in list(repo_dir.glob(f"snapshots/*/{meta.subfolder}")):
                    if sub.exists():
                        freed_bytes += _safe_rmtree(sub)
            else:
                # english / root model
                freed_bytes += _safe_rmtree(repo_dir)

        # 4. Clean matching lock directories
        for lock_dir in list(MODEL_DIR.glob(f".locks/{dedicated_slug}*")):
            freed_bytes += _safe_rmtree(lock_dir)

    mb = freed_bytes / (1024 * 1024)
    gb = mb / 1024
    size_lbl = f"{gb:.2f} GB" if gb >= 1.0 else f"{mb:.1f} MB"

    if freed_bytes > 0:
        console.print(
            f"[green]✓[/green] Removed model [bold cyan]{key}[/bold cyan] ({meta.repo}) [dim](freed {size_lbl})[/dim]"
        )
    else:
        console.print(
            f"[yellow]ℹ[/yellow] Model [bold cyan]{key}[/bold cyan] was not downloaded or already removed."
        )

    cfg = load_config()
    if cfg.get("default_model") == key:
        console.print(
            f"  [dim yellow]ℹ Removed model was set as default. Run: [cyan]qcplot --set-default <model>[/cyan][/dim yellow]"
        )


def cmd_init(device_req: str = "auto", backend_req: str = "auto") -> None:
    ensure_workspace_setup()
    console.print(
        Panel(
            f"[bold {ACCENT}]◆ QCPlot · First-Time Setup Wizard[/bold {ACCENT}]\n"
            "[dim]Creating workspace, checking GPU/ONNX backends, pulling default multilingual model & optimizing[/dim]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(0, 2),
            expand=False,
        )
    )

    console.print("[green]✓[/green] Workspace ready  [dim](model/, model/onnx/, data/, outputs/)[/dim]")

    cfg = load_config()
    cfg.setdefault("default_model", FALLBACK_DEFAULT_MODEL)
    cfg["default_device"] = device_req
    cfg["default_backend"] = backend_req
    save_config(cfg)
    default_model = cfg["default_model"]
    console.print(f"[green]✓[/green] Config saved     [dim](model={default_model}, backend={backend_req}, device={device_req})[/dim]")
    console.print("[green]✓[/green] Sample dataset   [dim](data/sample_bolts.csv ready)[/dim]")

    cached = is_model_cached(default_model)
    configure_environment(default_model, force_online=not cached)
    init_runtime()
    dev = resolve_device(device_req, backend_req)
    dev_style = "bold green" if dev.is_gpu else "yellow"
    console.print(f"[green]✓[/green] Active compute   [{dev_style}]{dev.label}[/{dev_style}]")

    if cached:
        mb = get_cached_size_mb(default_model)
        console.print(f"[green]✓[/green] Default model    [cyan]{default_model}[/cyan] [dim](cached · {mb:.0f} MB)[/dim]")
        if dev.runtime == "onnx":
            needs_export = not is_onnx_exported(default_model)
            if not needs_export and ort is not None:
                try:
                    sess, _ = _build_ort_session(get_onnx_model_path(default_model), "CPUExecutionProvider")
                    expected_outs = set(_detect_onnx_agent_output_names())
                    actual_outs = {o.name for o in sess.get_outputs()}
                    if not expected_outs.issubset(actual_outs):
                        needs_export = True
                except Exception:
                    needs_export = True
            if needs_export:
                try:
                    export_model_to_onnx(default_model, force=True)
                except Exception as exc:
                    console.print(f"  [dim yellow]ℹ ONNX export skipped ({exc})[/dim yellow]")
            else:
                console.print(f"[green]✓[/green] ONNX GPU build   [cyan]{default_model}[/cyan] [dim](ready · {get_onnx_size_mb(default_model):.0f} MB)[/dim]")
    else:
        cmd_pull_model(default_model, device_req=device_req, backend_req=backend_req)

    next_steps = Text()
    next_steps.append("Setup complete! Try running:\n", style="bold green")
    next_steps.append('  qcplot -f data/sample_bolts.csv -q "Bolt diameter stability check karo"\n', style="white")
    next_steps.append("  qcplot --interactive\n", style="cyan")
    next_steps.append("  qcplot --device-info\n", style="dim")
    next_steps.append("  qcplot --models", style="dim")
    console.print()
    console.print(Panel(next_steps, box=box.ROUNDED, border_style="green", padding=(0, 2), expand=False))


# ---------------------------------------------------------------------------
# 7. QCPlot Modern Branding & Rich Braille Inline Chart
# ---------------------------------------------------------------------------
def clear_screen() -> None:
    """Smoothly clears the terminal screen across Windows, macOS, and Linux."""
    try:
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[H\033[2J\033[3J")
            sys.stdout.flush()
    except Exception:
        pass


def _model_display_label(key: str) -> str:
    norm = normalize_model_key(key)
    if norm == "auto":
        return "auto-router"
    meta = AVAILABLE_MODELS.get(norm)
    return f"{meta.repo} ({meta.key} · {meta.params})" if meta else key


def get_qcplot_logo_lines(width: int | None = None) -> list[str]:
    if width is None:
        try:
            width = shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:
            width = 80

    R = "\033[0m"
    FG = "\033[1;39m"
    DIM = "\033[38;2;100;116;139m"
    LINE = "\033[1;38;2;56;189;248m"

    COMP_SOFT = "\033[38;2;245;158;11m"
    COMP_BOLD = "\033[1;38;2;251;146;60m"

    B1 = "\033[38;2;147;197;253m"
    B2 = "\033[38;2;96;165;250m"
    B3 = "\033[38;2;59;130;246m"
    B4 = "\033[38;2;37;99;235m"

    PLOT_GRAD = [
        "\033[1;38;2;96;165;250m",
        "\033[1;38;2;75;148;248m",
        "\033[1;38;2;59;130;246m",
        "\033[1;38;2;47;114;240m",
        "\033[1;38;2;37;99;235m",
        "\033[1;38;2;29;78;216m",
    ]

    logo = [
        f"{FG}      ▄▄██████████▄▄      {R}",
        f"{FG}   ▄████▀▀      ▀▀████▄   {R}",
        f"{FG}  ███▀              ▀███  {R}",
        f"{FG} ███             {LINE}●{FG}    ███ {R}",
        f"{FG} ██           {LINE}●─╱{FG}      ██ {R}",
        f"{FG} ██        {LINE}●─╱{FG}   {B1}██{FG}    ██ {R}",
        f"{FG} ██     {LINE}●─╱{FG}   {B2}██{FG} {B2}██{FG}    ██ {R}",
        f"{FG} ███   {LINE}╱{FG}   {B3}██{FG} {B3}██{FG} {B3}██{FG}   ███ {R}",
        f"{FG}  ███▄  {B4}██{FG} {B4}██{FG} {B4}██{FG} {B4}██{FG} ▄███  {R}",
        f"{FG}   ▀████▄▄      ▄▄▀ ▀███▄ {R}",
        f"{FG}      ▀▀█████████▀   ▀████{R}",
    ]

    qc = [
        " ▄████▄   ▄████▄",
        "██▀  ▀██ ██▀  ▀▀",
        "██    ██ ██     ",
        "██    ██ ██     ",
        "██▄ ▀▄██ ██▄  ▄▄",
        " ▀████▀▄  ▀████▀",
    ]

    plot = [
        "██████▄ ██          ██  ",
        "██   ██ ██          ██  ",
        "██▄▄▄█▀ ██  ▄███▄ ██████",
        "██      ██ ██▀ ▀██  ██  ",
        "██      ██ ██▄ ▄██  ██▄ ",
        "██      ██  ▀███▀    ▀██",
    ]

    tagline = f"{FG}Intelligent {B3}Quality Control{FG} Charting.{R}"
    author = f"{COMP_SOFT}Author : {COMP_BOLD}Vishal Prajapati{R}"

    lines: list[str] = [""]

    # Wide / Standard layout (>= 76 cols): exact match to logo.py
    if width >= 76:
        right = [
            "",
            *[f"{FG}{q} {c}{p}{R}" for q, p, c in zip(qc, plot, PLOT_GRAD)],
            "",
            tagline,
            author,
            "",
        ]
        pad = "  "
        for i, (left_row, right_row) in enumerate(zip(logo, right)):
            sep = f"   {DIM}│{R}   " if 1 <= i <= 9 else "       "
            lines.append(f"{pad}{left_row}{sep}{right_row}")

    elif width >= 46:
        # Stacked layout for compact / split terminal windows (46 to 75 cols)
        logo_pad = " " * max(0, (width - 26) // 2)
        for row in logo:
            lines.append(f"{logo_pad}{row}")
        lines.append("")
        block_pad = " " * max(0, (width - 41) // 2)
        for q, p, c in zip(qc, plot, PLOT_GRAD):
            lines.append(f"{block_pad}{FG}{q} {c}{p}{R}")
        lines.append("")
        tag_pad = " " * max(0, (width - 36) // 2)
        auth_pad = " " * max(0, (width - 26) // 2)
        lines.append(f"{tag_pad}{tagline}")
        lines.append(f"{auth_pad}{author}")

    else:
        # Ultra-compact layout for very small screens (< 46 cols)
        logo_pad = " " * max(0, (width - 26) // 2)
        for row in logo:
            lines.append(f"{logo_pad}{row}")
        lines.append("")
        lines.append(f"  {FG}QCPlot{R}")
        lines.append(f"  {tagline}")
        lines.append(f"  {author}")

    lines.append("")
    return lines


def print_qcplot_logo(width: int | None = None) -> None:
    """Prints the modern QCPlot branding banner directly to stdout without wrapping."""
    for line in get_qcplot_logo_lines(width=width):
        print(line)


def print_qcplot_info_table(
    dev: DeviceInfo | None = None,
    question: str | None = None,
    chart_name: str | None = None,
    chart_category: str | None = None,
    decision_reason: str | None = None,
    data_info: str | None = None,
    model_choice: str | None = None,
) -> None:
    tbl = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        padding=(0, 2),
        show_header=False,
        expand=False,
    )
    tbl.add_column("Field", style="dim", no_wrap=True)
    tbl.add_column("Value", style="white")

    if dev is not None:
        if dev.is_gpu:
            dev_str = f"[bold green]GPU · {dev.backend.upper()}[/bold green] [dim]({dev.label})[/dim]"
        else:
            dev_str = f"[bold yellow]CPU[/bold yellow] [dim]({dev.label})[/dim]"
        tbl.add_row("Compute Device", dev_str)

    if model_choice is not None:
        tbl.add_row("Active Model", f"[cyan]{_model_display_label(model_choice)}[/cyan]")

    if question:
        tbl.add_row("Question", f'[bold white]"{question}"[/bold white]')

    if chart_name:
        cat_str = f" [dim]({chart_category})[/dim]" if chart_category else ""
        tbl.add_row("Question Type", f"[bold cyan]{chart_name}[/bold cyan]{cat_str}")

    if decision_reason:
        tbl.add_row("Decision Route", f"[dim]{decision_reason}[/dim]")

    if data_info:
        tbl.add_row("Data & Groups", f"[white]{data_info}[/white]")

    console.print(tbl)
    console.print()


print_qcplot_brand = print_qcplot_info_table


def render_terminal_chart(spec: ChartSpec) -> None:
    console.print()
    console.print(Rule("[dim]Inline Control Chart[/dim]", style="bright_black", align="left"))

    braille_bits = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
    prio_styles = {1: "green", 2: "red", 3: "bold cyan"}

    cols = max(44, min(console.size.width - 18, 84))
    rows = 12 if len(spec.panels) == 2 else 14
    px_w, px_h = cols * 2, rows * 4

    for panel in spec.panels:
        x = np.asarray(panel.x, dtype=float)
        y = np.asarray(panel.y, dtype=float)
        ucl_arr = np.broadcast_to(np.asarray(panel.ucl, dtype=float), y.shape)
        lcl_arr = np.broadcast_to(np.asarray(panel.lcl, dtype=float), y.shape)
        valid = ~np.isnan(y)

        xv, yv = x[valid], y[valid]
        ucl_v, lcl_v = ucl_arr[valid], lcl_arr[valid]
        is_var = np.ndim(panel.ucl) > 0 and not np.allclose(ucl_v, ucl_v[0], equal_nan=True)

        xmin, xmax = float(np.min(xv)), float(np.max(xv))
        if xmax == xmin:
            xmin, xmax = xmin - 1.0, xmax + 1.0

        all_vals = np.concatenate([yv, ucl_v, lcl_v, [panel.cl]])
        dmin, dmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = max((dmax - dmin) * 0.14, 1e-6)
        ymin, ymax = dmin - pad, dmax + pad

        bits = [[0] * cols for _ in range(rows)]
        prio = [[0] * cols for _ in range(rows)]
        ooc_cells = [[False] * cols for _ in range(rows)]

        def to_px(val_x: float, val_y: float) -> tuple[int, int]:
            px = int(round((val_x - xmin) / (xmax - xmin) * (px_w - 1)))
            py = int(round((ymax - val_y) / (ymax - ymin) * (px_h - 1)))
            return max(0, min(px_w - 1, px)), max(0, min(px_h - 1, py))

        def set_px(px: int, py: int, p: int) -> None:
            cx, dx = divmod(px, 2)
            cy, dy = divmod(py, 4)
            bits[cy][cx] |= braille_bits[dy][dx]
            if p >= prio[cy][cx]:
                prio[cy][cx] = p

        def draw_line(x0: int, y0: int, x1: int, y1: int, p: int, dashed: bool = False) -> None:
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                if dashed and (s % 4 >= 2):
                    continue
                px = int(round(x0 + (x1 - x0) * s / steps))
                py = int(round(y0 + (y1 - y0) * s / steps))
                set_px(px, py, p)

        _, cl_py = to_px(xmin, panel.cl)
        draw_line(0, cl_py, px_w - 1, cl_py, p=1)

        if is_var:
            for arr in (ucl_v, lcl_v):
                pts = [to_px(float(a), float(b)) for a, b in zip(xv, arr)]
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    mid_x = (x0 + x1) // 2
                    draw_line(x0, y0, mid_x, y0, p=2, dashed=True)
                    draw_line(mid_x, y0, mid_x, y1, p=2, dashed=True)
                    draw_line(mid_x, y1, x1, y1, p=2, dashed=True)
        else:
            _, ucl_py = to_px(xmin, float(ucl_v[-1]))
            _, lcl_py = to_px(xmin, float(lcl_v[-1]))
            draw_line(0, ucl_py, px_w - 1, ucl_py, p=2, dashed=True)
            draw_line(0, lcl_py, px_w - 1, lcl_py, p=2, dashed=True)

        pts = [to_px(float(a), float(b)) for a, b in zip(xv, yv)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            draw_line(x0, y0, x1, y1, p=3)
        for i, (px, py) in enumerate(pts):
            set_px(px, py, p=3)
            if yv[i] > ucl_v[i] or yv[i] < lcl_v[i]:
                ooc_cells[py // 4][px // 2] = True

        tick_rows = {0, rows // 4, rows // 2, (3 * rows) // 4, rows - 1}
        tick_vals = {r: ymax - ((r + 0.5) / rows) * (ymax - ymin) for r in tick_rows}
        y_w = max(len(f"{v:.2f}") for v in tick_vals.values())

        canvas = Text()
        for r in range(rows):
            lbl = f"{tick_vals[r]:>{y_w}.2f}" if r in tick_rows else " " * y_w
            canvas.append(f"{lbl} │", style="dim")
            for c in range(cols):
                if ooc_cells[r][c]:
                    canvas.append("◆", style="bold red")
                elif bits[r][c]:
                    canvas.append(chr(0x2800 | bits[r][c]), style=prio_styles[prio[r][c]])
                else:
                    canvas.append(" ")
            canvas.append("\n")

        canvas.append(f"{' ' * y_w} └{'─' * cols}\n", style="dim")
        x_bar = [" "] * cols
        tick_idxs = np.unique(np.linspace(0, len(xv) - 1, min(6, len(xv)), dtype=int))
        for ti in tick_idxs:
            tag = str(int(xv[ti]))
            cx = to_px(float(xv[ti]), ymin)[0] // 2
            start = max(0, min(cols - len(tag), cx - len(tag) // 2))
            if all(ch == " " for ch in x_bar[start : start + len(tag)]):
                x_bar[start : start + len(tag)] = list(tag)
        canvas.append(f"{' ' * (y_w + 2)}{''.join(x_bar)}", style="dim")

        ucl_end, lcl_end = float(ucl_v[-1]), float(lcl_v[-1])
        title = (
            f"[bold cyan]{panel.ylabel}[/bold cyan]  ·  "
            f"[red]UCL={ucl_end:.2f}[/red]  ·  "
            f"[green]{panel.cl_label}={panel.cl:.2f}[/green]  ·  "
            f"[red]LCL={lcl_end:.2f}[/red]"
        )
        console.print(
            Panel(
                canvas,
                title=title,
                title_align="left",
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(0, 1),
                expand=False,
            )
        )


def render_tui_report(
    spec: ChartSpec,
    decision_reason: str,
    model_used: str,
    file_path: Path,
    max_rows: int = 12,
    show_full: bool = False,
) -> None:
    if spec.report is None:
        return
    r = spec.report

    total_rows = len(r.table_rows)
    truncated = (not show_full) and (total_rows > max_rows)

    caption = None
    if truncated:
        head_n = max(1, max_rows // 2)
        tail_n = max(1, max_rows - head_n)
        hidden = total_rows - (head_n + tail_n)
        display_rows = r.table_rows[:head_n] + [["···"] * len(r.table_headers)] + r.table_rows[-tail_n:]
        caption = f"Showing {head_n} first & {tail_n} last of {total_rows} rows ({hidden} hidden · pass --full to expand)"
    else:
        display_rows = r.table_rows

    data_tbl = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        caption=caption,
        caption_style="dim italic",
        pad_edge=True,
    )

    for idx, col_name in enumerate(r.table_headers):
        if col_name.lower() == "status":
            data_tbl.add_column(col_name, justify="center")
        elif idx == 0:
            data_tbl.add_column(col_name, justify="right", style="dim")
        else:
            data_tbl.add_column(col_name, justify="right")

    for row in display_rows:
        if row[0] == "···":
            data_tbl.add_row(*[f"[dim]{c}[/dim]" for c in row])
            continue
        is_ooc_row = "OOC" in row[-1]
        styled_cells = []
        for idx, cell in enumerate(row):
            if idx == len(row) - 1:
                styled_cells.append(f"[bold red]▲ {cell}[/bold red]" if is_ooc_row else "[green]✓ OK[/green]")
            elif is_ooc_row:
                styled_cells.append(f"[red]{cell}[/red]")
            else:
                styled_cells.append(cell)
        data_tbl.add_row(*styled_cells)

    console.print()
    console.print(Rule("[dim]Data & Subgroup Table[/dim]", style="bright_black", align="left"))
    console.print(data_tbl)

    const_tbl = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold white",
        padding=(0, 2),
        expand=False,
    )
    const_tbl.add_column("Parameter / Constant", style="dim")
    const_tbl.add_column("Value", justify="right", style="bold cyan")
    for k, v in r.constants:
        const_tbl.add_row(k, v)

    console.print(Rule("[dim]Parameters & Constants[/dim]", style="bright_black", align="left"))
    console.print(const_tbl)

    formula_tbl = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        header_style=f"bold {ACCENT}",
        show_lines=True,
        padding=(0, 1),
    )
    formula_tbl.add_column("Metric", style="bold white", no_wrap=True)
    formula_tbl.add_column("Formula", style="cyan")
    formula_tbl.add_column("Step-by-Step Substitution", style="dim white")
    formula_tbl.add_column("Result", justify="right", style="bold green", no_wrap=True)
    for metric, formula, substitution, result in r.formulas_and_math:
        formula_tbl.add_row(metric, formula, substitution, result)

    console.print(Rule("[dim]Formulas & Step-by-Step Math[/dim]", style="bright_black", align="left"))
    console.print(formula_tbl)

    render_terminal_chart(spec)

    verdict_lines = Text()
    all_ok = True
    for i, (label, is_ok, detail) in enumerate(r.ooc_summary):
        if i > 0:
            verdict_lines.append("\n")
        if is_ok:
            verdict_lines.append("✓ ", style="bold green")
            verdict_lines.append(f"{label:<11}", style="bold white")
            verdict_lines.append(detail, style="green")
        else:
            all_ok = False
            verdict_lines.append("▲ ", style="bold red")
            verdict_lines.append(f"{label:<11}", style="bold white")
            verdict_lines.append(detail, style="bold red")

    console.print()
    console.print(
        Panel(
            verdict_lines,
            title="[bold]Verdict[/bold]",
            title_align="left",
            box=box.ROUNDED,
            border_style="green" if all_ok else "red",
            padding=(0, 2),
            expand=False,
        )
    )


def render_run_stats(stats: RunStats, out_file: Path) -> None:
    query_exec_s = stats.laya_infer_s + stats.math_solve_s
    if query_exec_s < 1.0:
        query_str = f"[bold cyan]{query_exec_s * 1000:.1f} ms[/bold cyan]"
    else:
        query_str = f"[bold cyan]{query_exec_s:.2f} s[/bold cyan]"

    if stats.chart_render_s < 1.0:
        plot_str = f"[white]{stats.chart_render_s * 1000:.1f} ms[/white]"
    else:
        plot_str = f"[white]{stats.chart_render_s:.2f} s[/white]"

    try:
        rel_out = out_file.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_out = out_file

    tbl = Table(box=box.ROUNDED, border_style="bright_black", padding=(0, 2), show_header=False, expand=False)
    tbl.add_column("Metric", style="dim", no_wrap=True)
    tbl.add_column("Value", style="white")

    tbl.add_row("Query Execution", query_str)
    tbl.add_row("Plotting Time", plot_str)
    tbl.add_row("Output PNG", f"[bold green]{rel_out}[/bold green]")

    console.print()
    console.print(
        Panel(
            tbl,
            title="[bold cyan]Run Stats[/bold cyan]",
            title_align="left",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(0, 1),
            expand=False,
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# 8. Modern Colorful Matplotlib Plotter (Dynamic Anti-Overlap Spacing)
# ---------------------------------------------------------------------------
def _resolve_label_y_fracs(
    values: list[float], vmin: float, vmax: float, min_gap: float = 0.17
) -> list[float]:
    span = max(vmax - vmin, 1e-9)
    indexed = sorted(enumerate([(v - vmin) / span for v in values]), key=lambda t: t[1])
    pos = [max(0.08, min(0.92, p)) for _, p in indexed]

    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < min_gap:
            pos[i] = pos[i - 1] + min_gap

    if pos[-1] > 0.92:
        shift = pos[-1] - 0.92
        pos = [p - shift for p in pos]
        for i in range(len(pos) - 2, -1, -1):
            if pos[i + 1] - pos[i] < min_gap:
                pos[i] = pos[i + 1] - min_gap

    pos = [max(0.06, min(0.94, p)) for p in pos]
    resolved = [0.0] * len(values)
    for (orig_idx, _), p in zip(indexed, pos):
        resolved[orig_idx] = p
    return resolved


def plot_control_chart(
    spec: ChartSpec,
    save_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    global plt
    if plt is None:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt

    n_panels = len(spec.panels)
    fig_h = 7.4 if n_panels == 2 else 4.9
    fig, axes = plt.subplots(n_panels, 1, figsize=(12.2, fig_h), sharex=True, squeeze=False)
    axes_flat = axes.ravel()

    BG_FIG, BG_AX = "#F8FAFC", "#FFFFFF"
    CLR_ZONE, CLR_CL, CLR_LIMIT = "#ECFDF5", "#059669", "#E11D48"
    CLR_LINE, CLR_PT, CLR_OOC = "#2563EB", "#1D4ED8", "#DC2626"
    CLR_GRID, CLR_TEXT, CLR_MUTED = "#E2E8F0", "#0F172A", "#475569"

    fig.patch.set_facecolor(BG_FIG)

    for idx, (ax, panel) in enumerate(zip(axes_flat, spec.panels)):
        ax.set_facecolor(BG_AX)
        x = np.asarray(panel.x, dtype=float)
        y = np.asarray(panel.y, dtype=float)
        ucl_arr = np.broadcast_to(np.asarray(panel.ucl, dtype=float), y.shape)
        lcl_arr = np.broadcast_to(np.asarray(panel.lcl, dtype=float), y.shape)
        valid = ~np.isnan(y)
        is_variable = np.ndim(panel.ucl) > 0 and not np.allclose(ucl_arr[valid], ucl_arr[valid][0], equal_nan=True)

        all_vals = np.concatenate([y[valid], ucl_arr[valid], lcl_arr[valid], [panel.cl]])
        dmin, dmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        dspan = max(dmax - dmin, 1e-6)
        ymin, ymax = dmin - dspan * 0.22, dmax + dspan * 0.24
        ax.set_ylim(ymin, ymax)

        x_pad = max((x[-1] - x[0]) * 0.035, 0.5) if len(x) > 1 else 0.5
        ax.set_xlim(x[0] - x_pad, x[-1] + x_pad)

        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color=CLR_GRID, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)

        if is_variable:
            ax.fill_between(x, lcl_arr, ucl_arr, step="mid", color=CLR_ZONE, alpha=0.65, zorder=1)
            ax.step(x, ucl_arr, where="mid", color=CLR_LIMIT, linestyle="--", linewidth=1.6, zorder=2)
            ax.step(x, lcl_arr, where="mid", color=CLR_LIMIT, linestyle="--", linewidth=1.6, zorder=2)
        else:
            ax.axhspan(float(lcl_arr[-1]), float(ucl_arr[-1]), color=CLR_ZONE, alpha=0.65, zorder=1)
            ax.axhline(float(ucl_arr[-1]), color=CLR_LIMIT, linestyle="--", linewidth=1.6, zorder=2)
            ax.axhline(float(lcl_arr[-1]), color=CLR_LIMIT, linestyle="--", linewidth=1.6, zorder=2)

        ax.axhline(panel.cl, color=CLR_CL, linestyle="-", linewidth=1.75, zorder=2)

        ooc = valid & ((y > ucl_arr) | (y < lcl_arr))
        in_ctrl = valid & ~ooc

        ax.plot(x[valid], y[valid], color=CLR_LINE, linewidth=1.9, alpha=0.9, zorder=3)
        ax.scatter(x[in_ctrl], y[in_ctrl], color=CLR_PT, edgecolor="white", linewidth=1.2, s=54, zorder=4)

        if np.any(ooc):
            ax.scatter(x[ooc], y[ooc], color=CLR_OOC, alpha=0.22, s=190, zorder=4)
            ax.scatter(x[ooc], y[ooc], color=CLR_OOC, edgecolor="white", linewidth=1.4, marker="D", s=64, zorder=5)
            for ox, oy in zip(x[ooc], y[ooc]):
                above = oy >= panel.cl
                offset_y = 11 if above else -15
                va = "bottom" if above else "top"
                if oy > ymax - dspan * 0.14:
                    offset_y, va = -15, "top"
                elif oy < ymin + dspan * 0.14:
                    offset_y, va = 11, "bottom"
                ax.annotate(
                    f"#{int(ox)}",
                    xy=(ox, oy),
                    xytext=(0, offset_y),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=8,
                    fontweight="bold",
                    color="#991B1B",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#FCA5A5", linewidth=0.8),
                    zorder=6,
                )

        ucl_end, lcl_end = float(ucl_arr[valid][-1]), float(lcl_arr[valid][-1])
        badge_specs = [
            (ucl_end, f"UCL = {ucl_end:.2f}", "#9F1239", "#FFF1F2", "#FDA4AF"),
            (panel.cl, f"{panel.cl_label} = {panel.cl:.2f}", "#065F46", "#ECFDF5", "#6EE7B7"),
            (lcl_end, f"LCL = {lcl_end:.2f}", "#9F1239", "#FFF1F2", "#FDA4AF"),
        ]
        y_fracs = _resolve_label_y_fracs([b[0] for b in badge_specs], ymin, ymax, min_gap=0.19)

        for (_, text_lbl, fg, bg, edge), y_frac in zip(badge_specs, y_fracs):
            ax.text(
                1.018,
                y_frac,
                text_lbl,
                transform=ax.transAxes,
                va="center",
                ha="left",
                fontsize=8.8,
                fontweight="bold",
                color=fg,
                bbox=dict(boxstyle="round,pad=0.34", facecolor=bg, edgecolor=edge, linewidth=1.0),
            )

        ax.set_ylabel(panel.ylabel, fontsize=10.5, fontweight="bold", color=CLR_TEXT, labelpad=10)
        ax.tick_params(axis="both", labelsize=9, colors=CLR_MUTED, length=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)

        if idx < n_panels - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Subgroup", fontsize=10, fontweight="bold", color=CLR_TEXT, labelpad=8)
            step = max(1, math.ceil(len(x) / 15))
            ax.set_xticks(x[::step])

    fig.suptitle(spec.title, fontsize=15, fontweight="bold", y=0.96, color=CLR_TEXT)
    fig.text(0.5, 0.905 if n_panels == 2 else 0.885, spec.subtitle, ha="center", fontsize=9.8, color=CLR_MUTED)

    if spec.footer_stats:
        fig.text(
            0.5,
            0.062 if n_panels == 2 else 0.082,
            "    •    ".join(spec.footer_stats),
            ha="center",
            va="center",
            fontsize=9.1,
            fontweight="semibold",
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1.0),
        )
    if spec.footer_note:
        fig.text(0.5, 0.018 if n_panels == 2 else 0.022, spec.footer_note, ha="center", fontsize=8.3, style="italic", color="#64748B")

    plt.subplots_adjust(
        left=0.085,
        right=0.835,
        top=0.845 if n_panels == 2 else 0.81,
        bottom=0.175 if n_panels == 2 else 0.23,
        hspace=0.20,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, facecolor=BG_FIG)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 9. Tabular Data Loading & Schema Detection
# ---------------------------------------------------------------------------
ID_HINTS = ("subgroup", "sample", "batch", "lot", "day", "shift", "run", "id", "no", "index", "time", "date")
SIZE_HINTS = ("size", "inspected", "total", "units", "pieces", "area", "length", "lot_size", "sample_size", "n")
COUNT_HINTS = ("defect", "defective", "reject", "fail", "error", "count", "nonconform", "nc", "issue", "flaw", "c", "d", "np")


def get_available_datasets() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        [
            p
            for p in DATA_DIR.glob("*")
            if p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls")
        ],
        key=lambda p: p.name.lower(),
    )


def render_file_not_found_error(file_path: Path) -> None:
    msg = Text()
    msg.append("File not found: ", style="bold red")
    msg.append(f"{file_path}\n", style="bold white")

    datasets = get_available_datasets()
    if datasets:
        msg.append("\nAvailable sample datasets in ./data:\n", style="dim")
        for d in datasets:
            try:
                rel = d.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = d
            msg.append(f"  • {rel}\n", style="cyan")
    else:
        msg.append("\nNo tabular files found in ./data directory.", style="dim")

    console.print(
        Panel(
            msg,
            title="[bold red]✖ Invalid File Path[/bold red]",
            box=box.ROUNDED,
            border_style="red",
            padding=(0, 2),
            expand=False,
        )
    )


def prompt_for_valid_file(console: Console, initial_input: str | None = None) -> Path | None:
    """Prompts until a valid tabular file is entered, or user exits/cancels."""
    raw = initial_input
    while True:
        if raw is None:
            try:
                raw = Prompt.ask(
                    f"[bold {ACCENT}]❯[/bold {ACCENT}] [white]File[/white]", console=console
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Aborted by user (Ctrl+C). Exiting...[/dim]")
                return None

        cleaned = raw.strip('"').strip("'").strip()
        if cleaned.lower() in ("exit", "quit", "q"):
            return None

        if not cleaned:
            console.print("[dim yellow]ℹ File path cannot be empty. Please enter a valid path.[/dim yellow]\n")
            raw = None
            continue

        target_path = Path(cleaned)
        if target_path.exists() and target_path.is_file():
            return target_path

        # Try relative to data folder if user just entered the filename
        in_data = DATA_DIR / target_path.name
        if in_data.exists() and in_data.is_file():
            return in_data

        render_file_not_found_error(target_path)
        raw = None


def load_table(file_path: Path) -> Any:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix == ".tsv":
            return pd.read_csv(file_path, sep="\t")
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path.name}' is empty.")
    except pd.errors.ParserError as pe:
        raise ValueError(f"Could not parse '{file_path.name}'. Invalid CSV/TSV format: {pe}")
    except Exception as exc:
        raise ValueError(f"Failed to read '{file_path.name}': {exc}")
    raise ValueError(f"Unsupported tabular file format '{suffix}'. Supported formats: .csv, .tsv, .xlsx, .xls")


def _col_matches(col: str, hints: tuple[str, ...]) -> bool:
    name = re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
    return any(name == h or h in name.split("_") or (len(h) > 2 and h in name) for h in hints)


def extract_question_sample_size(question: str) -> int | None:
    patterns = [
        r"\bn\s*=\s*(\d+)\b",
        r"(?:sample\s+size|subgroup\s+size|batch\s+size|batches?\s+of|samples?\s+of|size\s+of)\s*(\d+)",
        r"(\d+)\s*(?:pieces|units|items|parts)\s*(?:each|per)",
    ]
    for pat in patterns:
        m = re.search(pat, question, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


extract_prompt_sample_size = extract_question_sample_size


def detect_explicit_chart(question: str) -> str | None:
    patterns = [
        ("xbar_s", r"\b(?:x[\s-]*bar[\s-]*(?:and|&|-)?[\s-]*s|xbar[\s-]*s)\b"),
        ("xbar_r", r"\b(?:x[\s-]*bar[\s-]*(?:and|&|-)?[\s-]*r|xbar[\s-]*r)\b"),
        ("i_mr", r"\b(?:i[\s-]*(?:and|&|-)?[\s-]*mr|x[\s-]*mr|individual[s\s-]+moving[\s-]+range)\b"),
        ("np", r"\b(?:np[\s-]*chart|npchart)\b"),
        ("p", r"\b(?:p[\s-]*chart|pchart)\b"),
        ("c", r"\b(?:c[\s-]*chart|cchart)\b"),
        ("u", r"\b(?:u[\s-]*chart|uchart)\b"),
    ]
    for chart_key, pat in patterns:
        if re.search(pat, question, flags=re.IGNORECASE):
            return chart_key
    return None


def profile_dataframe(df: Any, question: str) -> dict[str, Any]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = [c for c in df.columns if _col_matches(c, ID_HINTS) and not _col_matches(c, SIZE_HINTS)]
    size_cols = [c for c in num_cols if _col_matches(c, SIZE_HINTS)]
    count_cols = [c for c in num_cols if _col_matches(c, COUNT_HINTS) and c not in size_cols]

    data_cols = [c for c in num_cols if c not in id_cols and c not in size_cols]
    if not data_cols:
        data_cols = num_cols

    vals = df[data_cols].dropna().to_numpy()
    all_nonneg_int = bool(vals.size > 0 and np.all(vals >= 0) and np.all(np.mod(vals, 1) == 0))

    subgroup_col = None
    inferred_n = 1
    layout = "single"
    if len(data_cols) >= 2:
        inferred_n = len(data_cols)
        layout = "wide"
    elif id_cols and len(data_cols) == 1:
        grp_sizes = df.groupby(id_cols[0])[data_cols[0]].count()
        if grp_sizes.max() > 1:
            subgroup_col = id_cols[0]
            inferred_n = int(round(grp_sizes.mean()))
            layout = "long"

    size_col = size_cols[0] if size_cols else None
    sample_size_constant = True
    if size_col is not None:
        sample_size_constant = df[size_col].nunique(dropna=True) <= 1

    q_n = extract_question_sample_size(question)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_columns": num_cols,
        "id_cols": id_cols,
        "size_col": size_col,
        "count_col": count_cols[0] if count_cols else (data_cols[0] if data_cols else None),
        "data_cols": data_cols,
        "subgroup_col": subgroup_col,
        "layout": layout,
        "inferred_n": inferred_n,
        "question_n": q_n,
        "prompt_n": q_n,
        "all_nonneg_int": all_nonneg_int,
        "sample_size_constant": sample_size_constant,
        "preview": df.head(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 10. Unified Laya Engine (ONNX GPU / DirectML + PyTorch GPU / CPU)
# ---------------------------------------------------------------------------
def _build_ort_session(onnx_path: Path, target_provider: str) -> tuple[Any, str]:
    """Creates an optimized ONNX Runtime InferenceSession (DirectML-safe)."""
    if onnx_path.exists():
        sanitize_onnx_model_for_hardware(onnx_path)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if target_provider == "DmlExecutionProvider":
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = (
        [target_provider, "CPUExecutionProvider"]
        if target_provider != "CPUExecutionProvider"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    active_provs = session.get_providers()
    active = active_provs[0] if active_provs else "CPUExecutionProvider"
    return session, active


def _sync_router_to_device(router: Any, target_dev: Any) -> str:
    agents = _find_loaded_agents(router)
    actual_type = "cpu"
    for agent in agents:
        mod = agent.model
        try:
            cur_dev = next(mod.parameters()).device
        except StopIteration:
            continue
        if str(cur_dev) != str(target_dev):
            mod.to(target_dev)
            if hasattr(agent, "device"):
                agent.device = target_dev
            if hasattr(agent, "_device"):
                agent._device = target_dev
        try:
            actual_type = next(mod.parameters()).device.type
        except StopIteration:
            pass
    return actual_type


def load_laya_engine(dev: DeviceInfo, model_choice: str, is_cached: bool) -> tuple[Any, DeviceInfo, float]:
    """Loads Laya via ONNXAgent (when ONNX GPU/backend active) or PyTorch Router."""
    t0 = time.perf_counter()
    active_dev = dev
    resolved_key = FALLBACK_DEFAULT_MODEL if model_choice == "auto" else normalize_model_key(model_choice)

    # Path A: ONNX Runtime (DirectML / CUDA / ROCm / CoreML / OpenVINO / ONNX CPU)
    if dev.runtime == "onnx" and ort is not None:
        try:
            onnx_path = get_onnx_model_path(resolved_key)
            if not is_onnx_exported(resolved_key):
                onnx_path = export_model_to_onnx(resolved_key)

            target_prov = dev.ort_provider or "CPUExecutionProvider"
            expected_outs = set(_detect_onnx_agent_output_names())

            with console.status(
                f"[dim]Loading ONNX engine ({resolved_key}) on [white]{dev.backend.upper()}[/white]...[/dim]",
                spinner="dots",
                spinner_style=ACCENT,
            ):
                with quiet_backend():
                    session, actual_prov = _build_ort_session(onnx_path, target_prov)
                    actual_outs = {o.name for o in session.get_outputs()}

            # Auto re-export if cached ONNX file has stale/mismatched output names (e.g., act_probs vs act_logits)
            if not expected_outs.issubset(actual_outs):
                onnx_path = export_model_to_onnx(resolved_key, force=True)
                with console.status(
                    f"[dim]Reloading ONNX session on [white]{dev.backend.upper()}[/white]...[/dim]",
                    spinner="dots",
                    spinner_style=ACCENT,
                ):
                    with quiet_backend():
                        session, actual_prov = _build_ort_session(onnx_path, target_prov)

            ckpt_dir = locate_checkpoint_dir(resolved_key)
            model_ref = str(ckpt_dir) if ckpt_dir is not None else AVAILABLE_MODELS[resolved_key].repo

            with console.status(
                "[dim]Attaching ONNXAgent tokenizer & runtime...[/dim]",
                spinner="dots",
                spinner_style=ACCENT,
            ):
                with quiet_backend():
                    from laya.onnx_agent import ONNXAgent

                    sig = inspect.signature(ONNXAgent.__init__).parameters
                    kwargs: dict[str, Any] = {"onnx_path": str(onnx_path)}
                    if "providers" in sig:
                        kwargs["providers"] = [target_prov, "CPUExecutionProvider"]
                    agent = ONNXAgent(model_ref, **kwargs)

                    if hasattr(agent, "session"):
                        agent.session = session
                    elif hasattr(agent, "_session"):
                        agent._session = session

            if actual_prov == "CPUExecutionProvider" and dev.is_gpu:
                active_dev = DeviceInfo(
                    runtime="onnx",
                    backend="cpu",
                    torch_device="cpu",
                    ort_provider="CPUExecutionProvider",
                    hardware_name=platform.processor() or platform.machine() or "CPU",
                    note=f"{target_prov} could not initialize; running ONNX on CPU",
                )
            elapsed = time.perf_counter() - t0
            dev_style = "green" if active_dev.is_gpu else "yellow"
            console.print(f"[{dev_style}]✓[/{dev_style}] [dim]Engine ready ({active_dev.label} · {elapsed:.2f}s)[/dim]")
            if active_dev.note:
                console.print(f"  [dim yellow]ℹ {active_dev.note}[/dim yellow]")
            return agent, active_dev, elapsed

        except Exception as onnx_exc:
            active_dev = DeviceInfo(
                runtime="torch",
                backend="cpu",
                torch_device="cpu",
                ort_provider=None,
                hardware_name=platform.processor() or platform.machine() or "CPU",
                note=f"ONNX init fallback to PyTorch CPU ({onnx_exc})",
            )

    # Path B: PyTorch Router (CUDA / ROCm / MPS / XPU / CPU)
    action = "Loading" if is_cached else "Downloading"
    label = f"[dim]{action} Laya ({model_choice}) on [white]{active_dev.backend.upper()}[/white]...[/dim]"

    def _build_torch(target_dev: Any):
        from laya import Router

        sig = inspect.signature(Router.__init__).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
        kwargs: dict[str, Any] = {}
        if "preload" in sig or has_var_kw:
            kwargs["preload"] = (model_choice == "auto")
        if "device" in sig or has_var_kw:
            kwargs["device"] = target_dev
        if "cache_dir" in sig or has_var_kw:
            kwargs["cache_dir"] = str(MODEL_DIR)

        r = Router(**kwargs)
        if model_choice != "auto" and hasattr(r, "load"):
            r.load(model_choice)
        return r

    with console.status(label, spinner="dots", spinner_style=ACCENT):
        with quiet_backend():
            try:
                router = _build_torch(active_dev.torch_device)
                if active_dev.is_gpu:
                    actual = _sync_router_to_device(router, active_dev.torch_device)
                    if actual == "cpu":
                        active_dev = DeviceInfo(
                            runtime="torch",
                            backend="cpu",
                            torch_device="cpu",
                            ort_provider=None,
                            hardware_name=platform.processor() or platform.machine() or "CPU",
                            note=f"Weights remained on CPU despite {active_dev.backend.upper()} target",
                        )
            except (NameError, ImportError, ModuleNotFoundError):
                raise
            except Exception as exc:
                if active_dev.is_gpu:
                    router = _build_torch("cpu")
                    active_dev = DeviceInfo(
                        runtime="torch",
                        backend="cpu",
                        torch_device="cpu",
                        ort_provider=None,
                        hardware_name=platform.processor() or platform.machine() or "CPU",
                        note=f"{active_dev.backend.upper()} unsupported ({exc}); fell back to CPU",
                    )
                else:
                    raise

    elapsed = time.perf_counter() - t0
    dev_style = "green" if active_dev.is_gpu else "yellow"
    console.print(f"[{dev_style}]✓[/{dev_style}] [dim]Engine ready ({active_dev.label} · {elapsed:.2f}s)[/dim]")
    if active_dev.note:
        console.print(f"  [dim yellow]ℹ {active_dev.note}[/dim yellow]")
    return router, active_dev, elapsed


def _get_choice(prediction: dict[str, Any], key: str) -> str:
    ans = prediction.get("answers", {}).get(key, {})
    return str(ans.get("choice", "")).strip().lower()


def _format_model_name(raw: Any, fallback: str = FALLBACK_DEFAULT_MODEL) -> str:
    if isinstance(raw, dict):
        raw = raw.get("model") or raw.get("checkpoint") or raw.get("name") or fallback
    return _model_display_label(str(raw or fallback))


def _engine_predict(
    engine: Any,
    state: dict[str, Any],
    questions: dict[str, Any],
    model_choice: str,
    active_dev: DeviceInfo,
) -> tuple[dict[str, Any], DeviceInfo]:
    kwargs: dict[str, Any] = {}
    if model_choice != "auto" and active_dev.runtime == "torch":
        sig = inspect.signature(engine.predict).parameters
        if "model" in sig:
            kwargs["model"] = model_choice

    try:
        if active_dev.runtime == "onnx":
            with quiet_backend():
                return engine.predict(state, questions), active_dev
        with quiet_backend(), torch.inference_mode():
            pred = engine.predict(state, questions, **kwargs)
        if active_dev.is_gpu:
            _sync_router_to_device(engine, active_dev.torch_device)
        return pred, active_dev
    except Exception as exc:
        if active_dev.is_gpu:
            cpu_name = platform.processor() or platform.machine() or "CPU"
            if active_dev.runtime == "onnx" and ort is not None:
                resolved_key = FALLBACK_DEFAULT_MODEL if model_choice == "auto" else normalize_model_key(model_choice)
                onnx_path = get_onnx_model_path(resolved_key)
                sess, _ = _build_ort_session(onnx_path, "CPUExecutionProvider")
                if hasattr(engine, "session"):
                    engine.session = sess
                elif hasattr(engine, "_session"):
                    engine._session = sess
                fb = DeviceInfo(
                    runtime="onnx",
                    backend="cpu",
                    torch_device="cpu",
                    ort_provider="CPUExecutionProvider",
                    hardware_name=cpu_name,
                    note=f"{active_dev.ort_provider} runtime error; switched to ONNX CPU ({exc})",
                )
                with quiet_backend():
                    return engine.predict(state, questions), fb

            if active_dev.runtime == "torch":
                _sync_router_to_device(engine, "cpu")
                fb = DeviceInfo(
                    runtime="torch",
                    backend="cpu",
                    torch_device="cpu",
                    ort_provider=None,
                    hardware_name=cpu_name,
                    note=f"{active_dev.backend.upper()} runtime error; switched to CPU ({exc})",
                )
                with quiet_backend(), torch.inference_mode():
                    return engine.predict(state, questions, **kwargs), fb
        raise


def _extract_used_model(pred: dict[str, Any], model_choice: str) -> str:
    if model_choice != "auto":
        return _format_model_name(model_choice)
    raw = pred.get("routing") or pred.get("model") or pred.get("checkpoint")
    if raw:
        return _format_model_name(raw)
    return "convaiinnovations/laya (Router Auto)"


def decide_chart_with_laya(
    engine: Any,
    question: str,
    profile: dict[str, Any],
    model_choice: str,
    active_dev: DeviceInfo,
) -> tuple[str, str, str, DeviceInfo]:
    explicit = detect_explicit_chart(question)
    if explicit in CHART_REGISTRY:
        return explicit, f"Explicitly requested in question → {explicit.upper()}", "Skipped (explicit question match)", active_dev

    val_nature = "whole integer counts" if profile["all_nonneg_int"] else "decimal continuous measurements"
    state = {
        "question": question,
        "columns": ", ".join(profile["columns"]),
        "value_format": val_nature,
        "sample_rows": profile["preview"],
    }

    step1, active_dev = _engine_predict(
        engine,
        state,
        {
            "data_type": {
                "type": "choice",
                "instructions": "Is this quality control problem about discrete defect counts or continuous physical measurements?",
                "criteria": {
                    "discrete": "Discrete counts of defective pieces, rejects, errors, scratches, or defect occurrences",
                    "continuous": "Continuous physical measurements like length, diameter, weight, temperature, pressure, or time",
                },
            }
        },
        model_choice,
        active_dev,
    )
    model_used = _extract_used_model(step1, model_choice)
    data_type = _get_choice(step1, "data_type")

    if (not profile["all_nonneg_int"]) or profile["layout"] in ("wide", "long"):
        data_type = "continuous"

    if data_type == "discrete":
        step2, active_dev = _engine_predict(
            engine,
            state,
            {
                "count_kind": {
                    "type": "choice",
                    "instructions": "Does the data count defective pieces/items (pass/fail) or defect occurrences/flaws per unit?",
                    "criteria": {
                        "defectives": "Number of defective pieces, bad items, rejected units, or non-conforming parts",
                        "defects": "Number of defect occurrences, scratches, blemishes, soldering flaws, or errors per unit/sheet",
                    },
                },
                "sample_size_type": {
                    "type": "choice",
                    "instructions": "Is the sample size constant or variable across batches/subgroups?",
                    "criteria": {
                        "constant": "Constant, equal, or identical sample size across every subgroup",
                        "variable": "Variable, changing, or unequal sample size or inspected units across subgroups",
                    },
                },
            },
            model_choice,
            active_dev,
        )
        model_used = _extract_used_model(step2, model_choice)
        kind = _get_choice(step2, "count_kind")

        count_col_lower = str(profile.get("count_col") or "").lower()
        question_lower = question.lower()
        if any(w in count_col_lower for w in ("defective", "reject", "nonconform", "fail", "bad")) or any(
            w in question_lower for w in ("defective", "defectives", "proportion defective", "fraction defective", "nonconforming")
        ):
            kind = "defectives"
        elif any(w in count_col_lower for w in ("blemish", "scratch", "flaw", "defect")) and "defective" not in count_col_lower:
            kind = "defects"

        if profile["size_col"] is not None:
            is_const = profile["sample_size_constant"]
        else:
            # Bina size column ke sample size variable ho hi nahi sakta -> always constant (c / np)
            is_const = True

        size_lbl = "Constant size" if is_const else "Variable size"
        if kind == "defectives":
            chart = "np" if is_const else "p"
            return chart, f"Discrete → # Defectives → {size_lbl} → {chart.upper()}", model_used, active_dev
        chart = "c" if is_const else "u"
        return chart, f"Discrete → # Defects → {size_lbl} → {chart.upper()}", model_used, active_dev

    n = profile["inferred_n"]
    if n > 9:
        return "xbar_s", f"Continuous → Subgroup size n={n} (> 9) → Xbar-S", model_used, active_dev
    if 2 <= n <= 9:
        return "xbar_r", f"Continuous → Subgroup size n={n} (2 to 9) → Xbar-R", model_used, active_dev
    return "i_mr", "Continuous → Individual observations (n=1) → I-MR", model_used, active_dev


# ---------------------------------------------------------------------------
# 11. QC Solvers (All 7 Flowchart Charts)
# ---------------------------------------------------------------------------
def _extract_subgroups(df: Any, profile: dict[str, Any]) -> tuple[Any, list[Any]]:
    if profile["layout"] == "wide":
        cols = profile["data_cols"]
        groups = [row.dropna().to_numpy(dtype=float) for _, row in df[cols].iterrows()]
        return np.arange(1, len(groups) + 1), groups
    if profile["layout"] == "long" and profile["subgroup_col"]:
        col = profile["data_cols"][0]
        grouped = df.groupby(profile["subgroup_col"], sort=False)[col]
        groups = [g.dropna().to_numpy(dtype=float) for _, g in grouped]
        return np.arange(1, len(groups) + 1), groups
    col = profile["data_cols"][0]
    vals = df[col].dropna().to_numpy(dtype=float)
    return np.arange(1, len(vals) + 1), [np.array([v]) for v in vals]


def solve_xbar_r(df: Any, profile: dict[str, Any]) -> ChartSpec:
    x, groups = _extract_subgroups(df, profile)
    k = len(groups)
    if k < 1:
        raise ValueError("No valid subgroups found in the dataset.")
    n = int(round(np.mean([len(g) for g in groups])))
    if n not in SPC_CONSTANTS:
        raise ValueError(f"Xbar-R requires subgroup size 2..{max(SPC_CONSTANTS)} (got n={n}).")
    A2, D3, D4, d2 = SPC_CONSTANTS[n]

    means = np.array([np.mean(g) for g in groups])
    ranges = np.array([np.ptp(g) for g in groups])
    xbar_bar, r_bar = float(np.mean(means)), float(np.mean(ranges))

    ucl_x, lcl_x = xbar_bar + A2 * r_bar, xbar_bar - A2 * r_bar
    ucl_r, lcl_r = D4 * r_bar, D3 * r_bar

    all_obs = np.concatenate(groups)
    stdev_within = r_bar / d2
    stdev_overall = float(np.std(all_obs, ddof=1))

    show_raw = profile["layout"] == "wide" and n <= 6
    headers = ["Subgroup"] + (profile["data_cols"] if show_raw else ["n"]) + ["Mean (X̄)", "Range (R)", "Status"]
    rows: list[list[str]] = []
    for i, g in enumerate(groups):
        flags = []
        if means[i] > ucl_x or means[i] < lcl_x:
            flags.append("Xbar OOC")
        if ranges[i] > ucl_r or ranges[i] < lcl_r:
            flags.append("R OOC")
        if show_raw:
            raw_vals = df[profile["data_cols"]].iloc[i]
            raw_cells = ["-" if pd.isna(v) else f"{float(v):.2f}" for v in raw_vals]
        else:
            raw_cells = [str(len(g))]
        rows.append([str(x[i]), *raw_cells, f"{means[i]:.3f}", f"{ranges[i]:.3f}", ", ".join(flags) or "OK"])

    p_mean = PanelSpec("Mean", x, means, xbar_bar, ucl_x, lcl_x, "X̄")
    p_range = PanelSpec("Range", x, ranges, r_bar, ucl_r, lcl_r, "R̄")
    ooc_x, ooc_r = _find_ooc_indices(p_mean), _find_ooc_indices(p_range)

    report = ReportSpec(
        chart_name="Xbar and R Chart",
        problem_category="Continuous · Measurements",
        table_headers=headers,
        table_rows=rows,
        constants=[
            ("Subgroups (k)", str(k)),
            ("Subgroup Size (n)", str(n)),
            ("A2 (Xbar limits)", f"{A2:.3f}"),
            ("D3 (R LCL)", f"{D3:.3f}"),
            ("D4 (R UCL)", f"{D4:.3f}"),
            ("d2 (StDev within)", f"{d2:.3f}"),
        ],
        formulas_and_math=[
            ("Grand Mean (X̄̄)", "ΣX̄ / k", f"{np.sum(means):.3f} / {k}", f"{xbar_bar:.4f}"),
            ("Avg Range (R̄)", "ΣR / k", f"{np.sum(ranges):.3f} / {k}", f"{r_bar:.4f}"),
            ("Xbar UCL", "X̄̄ + A2 × R̄", f"{xbar_bar:.4f} + ({A2:.3f} × {r_bar:.4f})", f"{ucl_x:.4f}"),
            ("Xbar LCL", "X̄̄ − A2 × R̄", f"{xbar_bar:.4f} − ({A2:.3f} × {r_bar:.4f})", f"{lcl_x:.4f}"),
            ("R Chart UCL", "D4 × R̄", f"{D4:.3f} × {r_bar:.4f}", f"{ucl_r:.4f}"),
            ("R Chart LCL", "D3 × R̄", f"{D3:.3f} × {r_bar:.4f}", f"{lcl_r:.4f}"),
            ("StDev (within)", "R̄ / d2", f"{r_bar:.4f} / {d2:.3f}", f"{stdev_within:.4f}"),
            ("StDev (overall)", "s(all observations)", f"ddof=1 across N={len(all_obs)}", f"{stdev_overall:.4f}"),
        ],
        ooc_summary=[
            ("Xbar Chart", not ooc_x, "IN CONTROL" if not ooc_x else f"OUT OF CONTROL at subgroups {ooc_x}"),
            ("R Chart", not ooc_r, "IN CONTROL" if not ooc_r else f"OUT OF CONTROL at subgroups {ooc_r}"),
        ],
    )

    return ChartSpec(
        title="Xbar and R Control Charts",
        panels=[p_mean, p_range],
        footer_stats=[f"Subgroups: {k}", f"Mean: {xbar_bar:.3f}", f"StDev(within): {stdev_within:.4f}", f"StDev(overall): {stdev_overall:.4f}"],
        footer_note="Control limits are estimated using the StDev(within) = R̄ / d2.",
        report=report,
    )


def solve_xbar_s(df: Any, profile: dict[str, Any]) -> ChartSpec:
    x, groups = _extract_subgroups(df, profile)
    k = len(groups)
    if k < 1:
        raise ValueError("No valid subgroups found in the dataset.")
    n = int(round(np.mean([len(g) for g in groups])))
    if n < 2:
        raise ValueError(f"Xbar-S requires subgroup size >= 2 (got n={n}).")
    c4 = c4_constant(n)
    A3 = 3.0 / (c4 * math.sqrt(n))
    factor = (3.0 / c4) * math.sqrt(max(0.0, 1.0 - c4**2))
    B3, B4 = max(0.0, 1.0 - factor), 1.0 + factor

    means = np.array([np.mean(g) for g in groups])
    stds = np.array([np.std(g, ddof=1) for g in groups])
    xbar_bar, s_bar = float(np.mean(means)), float(np.mean(stds))
    ucl_x, lcl_x = xbar_bar + A3 * s_bar, xbar_bar - A3 * s_bar
    ucl_s, lcl_s = B4 * s_bar, B3 * s_bar

    all_obs = np.concatenate(groups)
    stdev_within = s_bar / c4
    stdev_overall = float(np.std(all_obs, ddof=1))

    rows = []
    for i, g in enumerate(groups):
        flags = []
        if means[i] > ucl_x or means[i] < lcl_x:
            flags.append("Xbar OOC")
        if stds[i] > ucl_s or stds[i] < lcl_s:
            flags.append("S OOC")
        rows.append([str(x[i]), str(len(g)), f"{means[i]:.3f}", f"{stds[i]:.4f}", ", ".join(flags) or "OK"])

    p_mean = PanelSpec("Mean", x, means, xbar_bar, ucl_x, lcl_x, "X̄")
    p_std = PanelSpec("StDev", x, stds, s_bar, ucl_s, lcl_s, "S̄")
    ooc_x, ooc_s = _find_ooc_indices(p_mean), _find_ooc_indices(p_std)

    report = ReportSpec(
        chart_name="Xbar and S Chart",
        problem_category="Continuous · Measurements",
        table_headers=["Subgroup", "n", "Mean (X̄)", "StDev (s)", "Status"],
        table_rows=rows,
        constants=[
            ("Subgroups (k)", str(k)),
            ("Subgroup Size (n)", str(n)),
            ("c4", f"{c4:.4f}"),
            ("A3", f"{A3:.4f}"),
            ("B3 / B4", f"{B3:.4f} / {B4:.4f}"),
        ],
        formulas_and_math=[
            ("Grand Mean (X̄̄)", "ΣX̄ / k", f"{np.sum(means):.3f} / {k}", f"{xbar_bar:.4f}"),
            ("Avg StDev (S̄)", "Σs / k", f"{np.sum(stds):.4f} / {k}", f"{s_bar:.4f}"),
            ("Xbar UCL", "X̄̄ + A3 × S̄", f"{xbar_bar:.4f} + ({A3:.4f} × {s_bar:.4f})", f"{ucl_x:.4f}"),
            ("Xbar LCL", "X̄̄ − A3 × S̄", f"{xbar_bar:.4f} − ({A3:.4f} × {s_bar:.4f})", f"{lcl_x:.4f}"),
            ("S Chart UCL", "B4 × S̄", f"{B4:.4f} × {s_bar:.4f}", f"{ucl_s:.4f}"),
            ("S Chart LCL", "B3 × S̄", f"{B3:.4f} × {s_bar:.4f}", f"{lcl_s:.4f}"),
            ("StDev (within)", "S̄ / c4", f"{s_bar:.4f} / {c4:.4f}", f"{stdev_within:.4f}"),
            ("StDev (overall)", "s(all observations)", f"ddof=1 across N={len(all_obs)}", f"{stdev_overall:.4f}"),
        ],
        ooc_summary=[
            ("Xbar Chart", not ooc_x, "IN CONTROL" if not ooc_x else f"OUT OF CONTROL at subgroups {ooc_x}"),
            ("S Chart", not ooc_s, "IN CONTROL" if not ooc_s else f"OUT OF CONTROL at subgroups {ooc_s}"),
        ],
    )

    return ChartSpec(
        title="Xbar and S Control Charts",
        panels=[p_mean, p_std],
        footer_stats=[f"Subgroups: {k}", f"Mean: {xbar_bar:.3f}", f"StDev(within): {stdev_within:.4f}", f"StDev(overall): {stdev_overall:.4f}"],
        footer_note="Control limits are estimated using the StDev(within) = S̄ / c4.",
        report=report,
    )


def solve_i_mr(df: Any, profile: dict[str, Any]) -> ChartSpec:
    col = profile["data_cols"][0]
    vals = df[col].dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        raise ValueError(f"I-MR chart requires at least 2 observations (got {len(vals)}).")
    x = np.arange(1, len(vals) + 1)
    mr = np.empty_like(vals)
    mr[0] = np.nan
    mr[1:] = np.abs(np.diff(vals))

    x_bar, mr_bar = float(np.mean(vals)), float(np.nanmean(mr))
    d2, D3, D4 = 1.128, 0.0, 3.267
    E2 = 3.0 / d2
    ucl_i, lcl_i = x_bar + E2 * mr_bar, x_bar - E2 * mr_bar
    ucl_mr, lcl_mr = D4 * mr_bar, D3 * mr_bar
    stdev_within = mr_bar / d2
    stdev_overall = float(np.std(vals, ddof=1))

    rows = []
    for i in range(len(vals)):
        flags = []
        if vals[i] > ucl_i or vals[i] < lcl_i:
            flags.append("I OOC")
        if not np.isnan(mr[i]) and mr[i] > ucl_mr:
            flags.append("MR OOC")
        rows.append([str(x[i]), f"{vals[i]:.3f}", "-" if np.isnan(mr[i]) else f"{mr[i]:.3f}", ", ".join(flags) or "OK"])

    p_i = PanelSpec("Individual Value", x, vals, x_bar, ucl_i, lcl_i, "X̄")
    p_mr = PanelSpec("Moving Range", x, mr, mr_bar, ucl_mr, lcl_mr, "MR̄")
    ooc_i, ooc_mr = _find_ooc_indices(p_i), _find_ooc_indices(p_mr)

    report = ReportSpec(
        chart_name="I-MR Chart",
        problem_category="Continuous · Individual Observations",
        table_headers=["Obs #", f"Value ({col})", "MR", "Status"],
        table_rows=rows,
        constants=[
            ("Observations (N)", str(len(vals))),
            ("MR Span", "2"),
            ("d2 (n=2)", f"{d2:.3f}"),
            ("E2 = 3 / d2", f"{E2:.3f}"),
            ("D3 / D4", f"{D3:.3f} / {D4:.3f}"),
        ],
        formulas_and_math=[
            ("Mean (X̄)", "ΣX / N", f"{np.sum(vals):.3f} / {len(vals)}", f"{x_bar:.4f}"),
            ("Avg MR (MR̄)", "Σ|Xᵢ − Xᵢ₋₁| / (N−1)", f"{np.nansum(mr):.3f} / {len(vals) - 1}", f"{mr_bar:.4f}"),
            ("I Chart UCL", "X̄ + E2 × MR̄", f"{x_bar:.4f} + ({E2:.3f} × {mr_bar:.4f})", f"{ucl_i:.4f}"),
            ("I Chart LCL", "X̄ − E2 × MR̄", f"{x_bar:.4f} − ({E2:.3f} × {mr_bar:.4f})", f"{lcl_i:.4f}"),
            ("MR Chart UCL", "D4 × MR̄", f"{D4:.3f} × {mr_bar:.4f}", f"{ucl_mr:.4f}"),
            ("MR Chart LCL", "D3 × MR̄", f"{D3:.3f} × {mr_bar:.4f}", f"{lcl_mr:.4f}"),
            ("StDev (within)", "MR̄ / d2", f"{mr_bar:.4f} / {d2:.3f}", f"{stdev_within:.4f}"),
            ("StDev (overall)", "s(X)", f"ddof=1 across N={len(vals)}", f"{stdev_overall:.4f}"),
        ],
        ooc_summary=[
            ("I Chart", not ooc_i, "IN CONTROL" if not ooc_i else f"OUT OF CONTROL at observations {ooc_i}"),
            ("MR Chart", not ooc_mr, "IN CONTROL" if not ooc_mr else f"OUT OF CONTROL at observations {ooc_mr}"),
        ],
    )

    return ChartSpec(
        title="Individuals and Moving Range (I-MR) Charts",
        panels=[p_i, p_mr],
        footer_stats=[f"Observations: {len(vals)}", f"Mean: {x_bar:.3f}", f"StDev(within): {stdev_within:.4f}", f"StDev(overall): {stdev_overall:.4f}"],
        footer_note="Control limits are estimated using the average moving range (span = 2).",
        report=report,
    )


def _get_count_and_size(df: Any, profile: dict[str, Any]) -> tuple[Any, Any, Any]:
    count_col = profile["count_col"]
    if not count_col:
        raise ValueError("Could not detect a count column for discrete attribute chart.")
    size_col = profile.get("size_col")
    if size_col and size_col in df.columns:
        sub_df = df[[count_col, size_col]].dropna()
        counts = sub_df[count_col].to_numpy(dtype=float)
        sizes = sub_df[size_col].to_numpy(dtype=float)
        x = np.arange(1, len(counts) + 1)
    else:
        counts = df[count_col].dropna().to_numpy(dtype=float)
        x = np.arange(1, len(counts) + 1)
        sizes = np.full_like(counts, float(profile["prompt_n"])) if profile.get("prompt_n") else None
    return x, counts, sizes


def solve_np(df: Any, profile: dict[str, Any]) -> ChartSpec:
    x, counts, sizes = _get_count_and_size(df, profile)
    if sizes is None:
        raise ValueError("np chart requires a constant sample size column or n mentioned in question.")
    if len(counts) == 0:
        raise ValueError("No valid data rows found for np chart.")
    k, n = len(counts), float(sizes[0])
    if n <= 0:
        raise ValueError("np chart requires sample size n > 0.")
    np_bar = float(np.mean(counts))
    p_bar = np_bar / n
    sigma = math.sqrt(max(0.0, np_bar * (1.0 - p_bar)))
    ucl, lcl = np_bar + 3.0 * sigma, max(0.0, np_bar - 3.0 * sigma)

    rows = [[str(x[i]), str(int(n)), str(int(counts[i])), "OOC" if (counts[i] > ucl or counts[i] < lcl) else "OK"] for i in range(k)]
    panel = PanelSpec("Defectives (np)", x, counts, np_bar, ucl, lcl, "np̄")
    ooc = _find_ooc_indices(panel)

    report = ReportSpec(
        chart_name="np Chart",
        problem_category="Discrete · Defectives · Constant Size",
        table_headers=["Subgroup", "Size (n)", "Defectives (np)", "Status"],
        table_rows=rows,
        constants=[("Subgroups (k)", str(k)), ("Constant Size (n)", str(int(n))), ("Total Defectives", str(int(np.sum(counts)))), ("Proportion (p̄)", f"{p_bar:.4f}")],
        formulas_and_math=[
            ("Center Line (np̄)", "Σnp / k", f"{int(np.sum(counts))} / {k}", f"{np_bar:.4f}"),
            ("Proportion (p̄)", "np̄ / n", f"{np_bar:.4f} / {int(n)}", f"{p_bar:.4f}"),
            ("Std Error (σ)", "√(np̄ × (1 − p̄))", f"√({np_bar:.4f} × {1 - p_bar:.4f})", f"{sigma:.4f}"),
            ("UCL", "np̄ + 3σ", f"{np_bar:.4f} + (3 × {sigma:.4f})", f"{ucl:.4f}"),
            ("LCL", "max(0, np̄ − 3σ)", f"max(0, {np_bar:.4f} − {3 * sigma:.4f})", f"{lcl:.4f}"),
        ],
        ooc_summary=[("np Chart", not ooc, "IN CONTROL" if not ooc else f"OUT OF CONTROL at subgroups {ooc}")],
    )

    return ChartSpec(
        title="np Control Chart (Number of Defectives)",
        panels=[panel],
        footer_stats=[f"Subgroups: {k}", f"Sample Size (n): {int(n)}", f"Avg Defectives: {np_bar:.3f}", f"Proportion (p̄): {p_bar:.4f}"],
        footer_note="Control limits are estimated using the binomial distribution.",
        report=report,
    )


def solve_p(df: Any, profile: dict[str, Any]) -> ChartSpec:
    x, counts, sizes = _get_count_and_size(df, profile)
    if sizes is None:
        raise ValueError("p chart requires a sample size column.")
    if len(counts) == 0 or np.any(sizes <= 0):
        raise ValueError("p chart requires valid rows with sample sizes > 0.")
    k = len(counts)
    props = counts / sizes
    total_d, total_n = float(np.sum(counts)), float(np.sum(sizes))
    p_bar = total_d / total_n
    sigmas = np.sqrt(np.maximum(0.0, p_bar * (1.0 - p_bar) / sizes))
    ucl = np.minimum(1.0, p_bar + 3.0 * sigmas)
    lcl = np.maximum(0.0, p_bar - 3.0 * sigmas)

    rows = [
        [str(x[i]), str(int(sizes[i])), str(int(counts[i])), f"{props[i]:.4f}", f"{lcl[i]:.4f}", f"{ucl[i]:.4f}", "OOC" if (props[i] > ucl[i] or props[i] < lcl[i]) else "OK"]
        for i in range(k)
    ]
    panel = PanelSpec("Proportion (p)", x, props, p_bar, ucl, lcl, "p̄")
    ooc = _find_ooc_indices(panel)

    report = ReportSpec(
        chart_name="p Chart",
        problem_category="Discrete · Defectives · Variable Size",
        table_headers=["Subgroup", "Size (nᵢ)", "Defectives", "Prop (pᵢ)", "LCLᵢ", "UCLᵢ", "Status"],
        table_rows=rows,
        constants=[("Subgroups (k)", str(k)), ("Total Inspected", str(int(total_n))), ("Total Defectives", str(int(total_d)))],
        formulas_and_math=[
            ("Center Line (p̄)", "Σdᵢ / Σnᵢ", f"{int(total_d)} / {int(total_n)}", f"{p_bar:.4f}"),
            ("Subgroup Prop (pᵢ)", "dᵢ / nᵢ", "Per subgroup in table above", f"[{np.min(props):.4f}, {np.max(props):.4f}]"),
            ("Variable UCLᵢ", "min(1, p̄ + 3√(p̄(1−p̄)/nᵢ))", f"p̄={p_bar:.4f}, nᵢ∈[{int(np.min(sizes))},{int(np.max(sizes))}]", f"[{np.min(ucl):.4f}, {np.max(ucl):.4f}]"),
            ("Variable LCLᵢ", "max(0, p̄ − 3√(p̄(1−p̄)/nᵢ))", f"p̄={p_bar:.4f}, nᵢ∈[{int(np.min(sizes))},{int(np.max(sizes))}]", f"[{np.min(lcl):.4f}, {np.max(lcl):.4f}]"),
        ],
        ooc_summary=[("p Chart", not ooc, "IN CONTROL" if not ooc else f"OUT OF CONTROL at subgroups {ooc}")],
    )

    return ChartSpec(
        title="p Control Chart (Proportion Defective)",
        panels=[panel],
        footer_stats=[f"Subgroups: {k}", f"Total Inspected: {int(total_n)}", f"Proportion (p̄): {p_bar:.4f}"],
        footer_note="Stepped control limits reflect unequal subgroup sizes.",
        report=report,
    )


def solve_c(df: Any, profile: dict[str, Any]) -> ChartSpec:
    x, counts, _ = _get_count_and_size(df, profile)
    if len(counts) == 0:
        raise ValueError("No valid data rows found for c chart.")
    k, total_c = len(counts), float(np.sum(counts))
    c_bar = float(np.mean(counts))
    sigma = math.sqrt(max(0.0, c_bar))
    ucl, lcl = c_bar + 3.0 * sigma, max(0.0, c_bar - 3.0 * sigma)

    rows = [[str(x[i]), str(int(counts[i])), "OOC" if (counts[i] > ucl or counts[i] < lcl) else "OK"] for i in range(k)]
    panel = PanelSpec("Defects (c)", x, counts, c_bar, ucl, lcl, "c̄")
    ooc = _find_ooc_indices(panel)

    report = ReportSpec(
        chart_name="c Chart",
        problem_category="Discrete · Defects · Constant Area/Unit",
        table_headers=["Subgroup", "Defects (c)", "Status"],
        table_rows=rows,
        constants=[("Subgroups (k)", str(k)), ("Total Defects", str(int(total_c)))],
        formulas_and_math=[
            ("Center Line (c̄)", "Σc / k", f"{int(total_c)} / {k}", f"{c_bar:.4f}"),
            ("Std Error (σ)", "√c̄", f"√({c_bar:.4f})", f"{sigma:.4f}"),
            ("UCL", "c̄ + 3√c̄", f"{c_bar:.4f} + (3 × {sigma:.4f})", f"{ucl:.4f}"),
            ("LCL", "max(0, c̄ − 3√c̄)", f"max(0, {c_bar:.4f} − {3 * sigma:.4f})", f"{lcl:.4f}"),
        ],
        ooc_summary=[("c Chart", not ooc, "IN CONTROL" if not ooc else f"OUT OF CONTROL at subgroups {ooc}")],
    )

    return ChartSpec(
        title="c Control Chart (Count of Defects)",
        panels=[panel],
        footer_stats=[f"Subgroups: {k}", f"Total Defects: {int(total_c)}", f"Avg Defects (c̄): {c_bar:.3f}"],
        footer_note="Control limits are estimated using the Poisson distribution.",
        report=report,
    )


def solve_u(df: Any, profile: dict[str, Any]) -> ChartSpec:
    x, counts, sizes = _get_count_and_size(df, profile)
    if sizes is None:
        raise ValueError("u chart requires an inspection unit / sample size column.")
    if len(counts) == 0 or np.any(sizes <= 0):
        raise ValueError("u chart requires valid rows with inspection unit sizes > 0.")
    k = len(counts)
    u_vals = counts / sizes
    total_c, total_n = float(np.sum(counts)), float(np.sum(sizes))
    u_bar = total_c / total_n
    sigmas = np.sqrt(np.maximum(0.0, u_bar / sizes))
    ucl = u_bar + 3.0 * sigmas
    lcl = np.maximum(0.0, u_bar - 3.0 * sigmas)

    rows = [
        [str(x[i]), f"{sizes[i]:.1f}", str(int(counts[i])), f"{u_vals[i]:.4f}", f"{lcl[i]:.4f}", f"{ucl[i]:.4f}", "OOC" if (u_vals[i] > ucl[i] or u_vals[i] < lcl[i]) else "OK"]
        for i in range(k)
    ]
    panel = PanelSpec("Defects per Unit (u)", x, u_vals, u_bar, ucl, lcl, "ū")
    ooc = _find_ooc_indices(panel)

    report = ReportSpec(
        chart_name="u Chart",
        problem_category="Discrete · Defects · Variable Units",
        table_headers=["Subgroup", "Units (nᵢ)", "Defects", "uᵢ", "LCLᵢ", "UCLᵢ", "Status"],
        table_rows=rows,
        constants=[("Subgroups (k)", str(k)), ("Total Units", f"{total_n:.1f}"), ("Total Defects", str(int(total_c)))],
        formulas_and_math=[
            ("Center Line (ū)", "Σcᵢ / Σnᵢ", f"{int(total_c)} / {total_n:.1f}", f"{u_bar:.4f}"),
            ("Defects/Unit (uᵢ)", "cᵢ / nᵢ", "Per subgroup in table above", f"[{np.min(u_vals):.4f}, {np.max(u_vals):.4f}]"),
            ("Variable UCLᵢ", "ū + 3√(ū / nᵢ)", f"ū={u_bar:.4f}, nᵢ∈[{np.min(sizes):.1f},{np.max(sizes):.1f}]", f"[{np.min(ucl):.4f}, {np.max(ucl):.4f}]"),
            ("Variable LCLᵢ", "max(0, ū − 3√(ū / nᵢ))", f"ū={u_bar:.4f}, nᵢ∈[{np.min(sizes):.1f},{np.max(sizes):.1f}]", f"[{np.min(lcl):.4f}, {np.max(lcl):.4f}]"),
        ],
        ooc_summary=[("u Chart", not ooc, "IN CONTROL" if not ooc else f"OUT OF CONTROL at subgroups {ooc}")],
    )

    return ChartSpec(
        title="u Control Chart (Defects per Unit)",
        panels=[panel],
        footer_stats=[f"Subgroups: {k}", f"Total Units: {total_n:.1f}", f"Defects/Unit (ū): {u_bar:.4f}"],
        footer_note="Stepped control limits reflect unequal inspection unit sizes.",
        report=report,
    )


CHART_REGISTRY: dict[str, Callable[[Any, dict[str, Any]], ChartSpec]] = {
    "np": solve_np,
    "p": solve_p,
    "c": solve_c,
    "u": solve_u,
    "xbar_s": solve_xbar_s,
    "xbar_r": solve_xbar_r,
    "i_mr": solve_i_mr,
}


# ---------------------------------------------------------------------------
# 12. Runner & CLI Entrypoint
# ---------------------------------------------------------------------------
def run_once(
    engine: Any,
    dev: DeviceInfo,
    model_load_s: float,
    model_choice: str,
    file_path: Path,
    question: str,
    output: Path | None,
    popup: bool,
    max_rows: int,
    show_full: bool,
) -> DeviceInfo:
    t_query_start = time.perf_counter()

    with console.status("[dim]Analyzing dataset & routing flowchart...[/dim]", spinner="dots", spinner_style=ACCENT):
        t0 = time.perf_counter()
        df = load_table(file_path)
        profile = profile_dataframe(df, question)
        data_load_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected_chart, reason, model_used, dev = decide_chart_with_laya(
            engine, question, profile, model_choice, dev
        )
        laya_infer_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        chart_spec = CHART_REGISTRY[selected_chart](df, profile)
        math_solve_s = time.perf_counter() - t2

    group_count = len(chart_spec.report.table_rows) if chart_spec.report else len(df)
    data_info = f"{file_path.name} ({group_count} subgroups / observations)"
    chart_name = chart_spec.report.chart_name if chart_spec.report else selected_chart.upper()
    chart_category = chart_spec.report.problem_category if chart_spec.report else ""

    print_qcplot_info_table(
        dev=dev,
        question=question,
        chart_name=chart_name,
        chart_category=chart_category,
        decision_reason=reason,
        data_info=data_info,
    )

    render_tui_report(chart_spec, reason, model_used, file_path, max_rows=max_rows, show_full=show_full)

    out_file = output or (OUTPUT_DIR / f"{file_path.stem}_{selected_chart}.png")
    t3 = time.perf_counter()
    with console.status("[dim]Saving high-res PNG chart...[/dim]", spinner="dots", spinner_style="cyan"):
        plot_control_chart(chart_spec, save_path=out_file, show=False)
    chart_render_s = time.perf_counter() - t3

    now = time.perf_counter()
    stats = RunStats(
        model_used=model_used,
        device_label=dev.label,
        is_gpu=dev.is_gpu,
        runtime_init_s=RUNTIME_INIT_SECS,
        model_load_s=model_load_s,
        data_load_s=data_load_s,
        laya_infer_s=laya_infer_s,
        math_solve_s=math_solve_s,
        chart_render_s=chart_render_s,
        query_total_s=now - t_query_start,
        script_total_s=now - SCRIPT_START,
        subgroups=len(chart_spec.report.table_rows) if chart_spec.report else len(df),
    )
    render_run_stats(stats, out_file)

    if popup:
        plot_control_chart(chart_spec, save_path=None, show=True)
    return dev


def build_parser(default_model: str, default_device: str, default_backend: str) -> RichHelpParser:
    parser = RichHelpParser(prog="qcplot", add_help=True)
    parser.add_argument("-f", "--file", type=Path, help="Input CSV/TSV/XLSX tabular file")
    parser.add_argument("-q", "--question", type=str, dest="question", help="Natural language QC question")
    parser.add_argument("--prompt", type=str, dest="question", help=argparse.SUPPRESS)
    parser.add_argument("--interactive", action="store_true", help="Fast multi-query interactive session")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")

    parser.add_argument(
        "--model",
        default=default_model,
        choices=["multilingual", "english", "typed-decisions", "auto"],
        help=f"Laya model checkpoint (default: {default_model})",
    )
    parser.add_argument("--models", action="store_true", help="List available, cached & ONNX models")
    parser.add_argument("--pull", type=str, metavar="MODEL", help="Download a Laya model into ./model")
    parser.add_argument("--export-onnx", type=str, metavar="MODEL", help="Export a model to ONNX for GPU")
    parser.add_argument("--set-default", type=str, metavar="MODEL", help="Set default Laya model")
    parser.add_argument(
        "-r",
        "--remove",
        "--delete",
        type=str,
        dest="remove",
        metavar="MODEL",
        help="Delete a downloaded model or 'all' to purge cache",
    )

    parser.add_argument("--init", action="store_true", help="First-time setup wizard")
    parser.add_argument(
        "--backend",
        default=default_backend,
        choices=["auto", "onnx", "torch"],
        help="Inference backend (auto picks GPU first)",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        choices=["auto", "cuda", "rocm", "directml", "mps", "coreml", "xpu", "openvino", "cpu"],
        help="Compute target (GPU-first by default)",
    )
    parser.add_argument("--device-info", action="store_true", help="Show CPU/GPU/ONNX providers & active default")
    parser.add_argument("--install", action="store_true", help="Auto-detect OS, hardware & install all dependencies")

    parser.add_argument("--max-rows", type=int, default=12, help="Max rows before truncating table")
    parser.add_argument("--full", action="store_true", help="Show all rows without truncation")
    parser.add_argument("--popup", action="store_true", help="Also open Matplotlib GUI window")
    return parser


def main() -> None:
    clear_screen()
    print_qcplot_logo()
    ensure_workspace_setup()

    cfg = load_config()
    parser = build_parser(
        default_model=cfg.get("default_model", FALLBACK_DEFAULT_MODEL),
        default_device=cfg.get("default_device", "auto"),
        default_backend=cfg.get("default_backend", "auto"),
    )
    args = parser.parse_args()

    # Instant utility commands
    if args.install:
        from qcplot.installer import run_installer
        run_installer(device=args.device)
        return
    if args.remove:
        cmd_remove_model(args.remove)
        return
    if args.models:
        cmd_list_models()
        return
    if args.set_default:
        cmd_set_default(args.set_default)
        return
    if args.device_info:
        cmd_device_info()
        return
    if args.init:
        cmd_init(device_req=args.device, backend_req=args.backend)
        return
    if args.pull:
        cmd_pull_model(args.pull, device_req=args.device, backend_req=args.backend)
        return
    if args.export_onnx:
        key = normalize_model_key(args.export_onnx)
        configure_environment(key, force_online=not is_model_cached(key))
        init_runtime()
        try:
            export_model_to_onnx(key, force=True)
        except Exception as exc:
            console.print(
                Panel(
                    f"[bold red]✖ Export Error:[/bold red] {exc}",
                    box=box.ROUNDED,
                    border_style="red",
                    padding=(0, 2),
                    expand=False,
                )
            )
            sys.exit(1)
        return

    if args.interactive:
        cached = configure_environment(args.model)
        init_runtime()
        dev = resolve_device(args.device, args.backend)
        engine, active_dev, model_load_s = load_laya_engine(dev, args.model, cached)

        console.print("[dim]Interactive session active · type [white]exit[/white] to quit[/dim]\n")
        while True:
            console.print(Rule(style="bright_black"))
            try:
                valid_path = prompt_for_valid_file(console)
                if valid_path is None:
                    console.print(f"[dim]Session ended · Total elapsed: {time.perf_counter() - SCRIPT_START:.2f}s[/dim]")
                    break

                try:
                    question = Prompt.ask(f"[bold {ACCENT}]❯[/bold {ACCENT}] [white]Question[/white]", console=console).strip()
                except (KeyboardInterrupt, EOFError):
                    console.print(f"\n[dim]Session ended · Total elapsed: {time.perf_counter() - SCRIPT_START:.2f}s[/dim]")
                    break

                if question.lower() in ("exit", "quit", "q"):
                    break
                if not question:
                    console.print("[dim yellow]ℹ Question cannot be empty. Please enter your QC query.[/dim yellow]\n")
                    continue

                active_dev = run_once(
                    engine,
                    active_dev,
                    model_load_s,
                    args.model,
                    valid_path,
                    question,
                    None,
                    popup=args.popup,
                    max_rows=args.max_rows,
                    show_full=args.full,
                )
            except KeyboardInterrupt:
                console.print("\n[dim]Query cancelled (Ctrl+C). Type 'exit' to quit or enter a new query.[/dim]")
            except Exception as exc:
                console.print(
                    Panel(
                        f"[bold red]✖ Error:[/bold red] {exc}",
                        box=box.ROUNDED,
                        border_style="red",
                        padding=(0, 2),
                        expand=False,
                    )
                )
        return

    # Non-interactive CLI mode: validate input file & question before loading engine
    if args.file:
        raw_target = Path(str(args.file).strip('"').strip("'"))
        if not raw_target.exists() or not raw_target.is_file():
            in_data = DATA_DIR / raw_target.name
            if in_data.exists() and in_data.is_file():
                valid_file = in_data
            else:
                render_file_not_found_error(raw_target)
                sys.exit(1)
        else:
            valid_file = raw_target
    else:
        valid_file = prompt_for_valid_file(console)
        if valid_file is None:
            return

    question = args.question
    if not question:
        try:
            question = Prompt.ask(f"[bold {ACCENT}]❯[/bold {ACCENT}] [white]Question[/white]", console=console).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted by user (Ctrl+C). Exiting...[/dim]")
            sys.exit(130)
    if question.lower() in ("exit", "quit", "q"):
        return
    if not question:
        console.print(
            Panel(
                "[bold red]✖ Empty Question:[/bold red] Please provide a question describing the QC check to perform.",
                box=box.ROUNDED,
                border_style="red",
                padding=(0, 2),
                expand=False,
            )
        )
        sys.exit(1)

    # Inputs are valid: initialize runtime and engine
    cached = configure_environment(args.model)
    init_runtime()
    dev = resolve_device(args.device, args.backend)
    engine, active_dev, model_load_s = load_laya_engine(dev, args.model, cached)

    try:
        run_once(
            engine,
            active_dev,
            model_load_s,
            args.model,
            valid_file,
            question,
            args.output,
            popup=args.popup,
            max_rows=args.max_rows,
            show_full=args.full,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted by user (Ctrl+C). Exiting...[/dim]")
        sys.exit(130)
    except Exception as exc:
        console.print(
            Panel(
                f"[bold red]✖ Error:[/bold red] {exc}",
                box=box.ROUNDED,
                border_style="red",
                padding=(0, 2),
                expand=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted by user (Ctrl+C). Exiting...[/dim]")
        sys.exit(130)
    except Exception as unhandled_exc:
        console.print(
            Panel(
                f"[bold red]✖ Fatal Error:[/bold red] {unhandled_exc}",
                box=box.ROUNDED,
                border_style="red",
                padding=(0, 2),
                expand=False,
            )
        )
        sys.exit(1)