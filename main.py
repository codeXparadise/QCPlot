#!/usr/bin/env python
"""Main entrypoint for QCPlot forwarding to src/qcplot."""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent

# Auto-relaunch inside local .venv if invoked with system python missing dependencies
_venv_python = _project_root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
if _venv_python.exists() and sys.executable.lower() != str(_venv_python).lower():
    import subprocess
    _res = subprocess.run([str(_venv_python), str(Path(__file__).resolve())] + sys.argv[1:])
    sys.exit(_res.returncode)

_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if "--install" in sys.argv:
    from qcplot.installer import run_installer
    success = run_installer()
    sys.exit(0 if success else 1)

try:
    from qcplot import main
except ImportError as exc:
    print(f"\033[1;33m[!] QCPlot missing required dependencies: {exc}\033[0m")
    print("\033[1;36m[*] Automatically launching cross-platform installer...\033[0m")
    from qcplot.installer import run_installer
    if run_installer():
        from qcplot import main
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
