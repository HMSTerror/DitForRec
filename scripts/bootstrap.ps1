Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv)) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python -m ditforrec.data.toy --output-root data
python -m ditforrec.data.preprocess --dataset toy --data-root data
python -m ditforrec.train --config configs/toy_debug.yaml
