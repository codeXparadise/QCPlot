#!/usr/bin/env bash
# QCPlot POSIX Launcher (Linux / macOS)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/main.py" "$@"
elif command -v uv >/dev/null 2>&1; then
    exec uv run "$SCRIPT_DIR/main.py" "$@"
else
    exec python3 "$SCRIPT_DIR/main.py" "$@"
fi

