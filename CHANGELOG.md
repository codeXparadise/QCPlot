# Changelog

All notable changes to **QCPlot** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 2026-11-22

### Added
- **Cross-Platform Auto-Installer (`src/qcplot/installer.py`)**:
  - Host OS detection (`Windows`, `Linux`, `macOS`) and CPU architecture introspection (`x86_64`, `ARM64`).
  - Hardware acceleration detection (`DirectML` on Windows, `CUDA`, `ROCm`, `Apple Silicon MPS / CoreML`, and CPU fallback).
  - Automated dependency installation using `uv` (fast sync) or standard `pip`.
  - Directory structure provisioning (`model/`, `model/onnx/`, `data/`, `outputs/`) and default configuration initialization.
  - Complete health verification checking 12 core packages (`rich`, `numpy`, `pandas`, `matplotlib`, `plotext`, `openpyxl`, `torch`, `transformers`, `onnx`, `onnxruntime`, `onnxscript`, `laya`).
- **Standalone Installer Entrypoint (`install.py`)**:
  - Run `python install.py` on fresh clones for zero-configuration, one-step setup.
- **CLI `--install` Flag**:
  - Added `--install` argument to `qcplot` / `main.py` to trigger environment provisioning and verification.
- **POSIX Shell Launcher (`scripts/qcplot.sh`)**:
  - Added an executable bash launcher for Linux and macOS environments that automatically detects and routes into `.venv`.
- **Auto-Recovery in `main.py`**:
  - Automatically relaunches inside `.venv` if invoked with system Python, and automatically triggers the cross-platform installer if required dependencies are missing.

### Changed
- **`pyproject.toml` & `requirements.txt`**:
  - Added PEP 508 platform markers for ONNX Runtime dependencies:
    - `onnxruntime-directml>=1.24.4; sys_platform == 'win32'`
    - `onnxruntime>=1.20.0; sys_platform != 'win32'`
  - Added `[tool.uv] environments` configuration for Windows, Linux, and macOS platforms.
- **`uv.lock`**:
  - Regenerated lockfile with multi-platform resolution to support Windows, Linux, and macOS seamlessly.
- **`README.md`**:
  - Updated Quickstart instructions to highlight the new automated cross-platform installer.

### Fixed
- Fixed critical install crash on Linux/macOS where running `uv run main.py --init` failed due to `onnxruntime-directml` lacking non-Windows wheels.

---

## [0.2.0] - 2026-11-22

### Added
- Autonomous Statistical Process Control (SPC) CLI engine.
- Fine-tuned Laya neural router for automated chart selection (Xbar-R, Xbar-s, I-MR, p, np, c, u charts).
- Deterministic ASTM E2587 and ISO 7870-2 factor lookup and 3-sigma calculations.
- Unicode Braille terminal TUI plotting alongside publication-grade 200-DPI PNG chart export.
- Hardware acceleration support for ONNX Runtime and PyTorch backends.
- Interactive mode (`--interactive`), model catalog management (`--models`, `--pull`, `--export-onnx`), and hardware diagnostics (`--device-info`).
