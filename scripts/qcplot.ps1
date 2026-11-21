$root = Join-Path $PSScriptRoot ".."
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$scriptPy = Join-Path $root "main.py"
if (Test-Path $venvPy) {
    & $venvPy $scriptPy @args
} else {
    & python $scriptPy @args
}
