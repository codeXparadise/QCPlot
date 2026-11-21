"""QCPlot — Autonomous Statistical Process Control (SPC) Engine."""
from __future__ import annotations

__version__ = "0.2.0"

from qcplot.cli import main
from qcplot.installer import run_installer, detect_platform_info, install_packages

__all__ = ["main", "__version__", "run_installer", "detect_platform_info", "install_packages"]
