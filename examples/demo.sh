#!/usr/bin/env bash
set -euo pipefail
python -m shapleylab demo
python -m shapleylab explain --input examples/product.json
if python -m shapleylab explain --input examples/product.json --max-evaluations 14 >/tmp/shapleylab-budget-output 2>/tmp/shapleylab-budget-error; then
  echo 'budget gate should reject 15 required calls' >&2; exit 1
fi
test ! -s /tmp/shapleylab-budget-output
echo 'exact-work budget gate rejects before output as expected'

python -m shapleylab explain --input examples/grouped.json

python -m shapleylab explain --input examples/product.json --permutations 64 --seed 0
