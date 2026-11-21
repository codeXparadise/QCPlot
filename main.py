#!/usr/bin/env python
"""Main entrypoint for QCPlot forwarding to src/qcplot."""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qcplot import main

if __name__ == "__main__":
    main()
