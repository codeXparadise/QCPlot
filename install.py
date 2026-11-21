#!/usr/bin/env python
"""QCPlot Standalone Cross-Platform Auto-Installer.

Detects the host OS (Windows, Linux, macOS), architecture, and GPU acceleration,
automatically installs all required packages, provisions the workspace, and verifies
everything so QCPlot works out of the box across any platform.

Usage:
    python install.py
    python install.py --skip-packages   # Setup workspace & verify only
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent

# If .venv exists and we are not running inside it, re-exec inside .venv
_venv_python = _project_root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
if _venv_python.exists() and sys.executable.lower() != str(_venv_python).lower():
    import subprocess
    res = subprocess.run([str(_venv_python), str(Path(__file__).resolve())] + sys.argv[1:])
    sys.exit(res.returncode)

_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qcplot.installer import run_installer

if __name__ == "__main__":
    skip = "--skip-packages" in sys.argv
    success = run_installer(skip_packages=skip)
    sys.exit(0 if success else 1)
