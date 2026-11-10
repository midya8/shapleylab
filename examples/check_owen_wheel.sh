#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python -m pip wheel "$root" --no-deps -w "$work/wheels"
python -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --no-deps "$work"/wheels/*.whl
mkdir "$work/run"
cp "$root/examples/verify_owen_installed.py" "$work/run/verify.py"
cd "$work/run"
unset PYTHONPATH
"$work/venv/bin/python" verify.py
export PATH="$work/venv/bin:$PATH"
installed=$(python -c 'import pathlib,shapleylab;print(pathlib.Path(shapleylab.__file__).resolve().parent.parent)')
bash "$root/examples/verify_wheel.sh" "$installed"
