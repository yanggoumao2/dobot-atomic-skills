#!/usr/bin/env bash
set -eo pipefail
source /home/yxy/data/projects/dobot-act/activate.sh
dataset=${1:?Usage: bash train_act60.sh /absolute/dataset/path}
python - "$dataset" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads((p/'splits.json').read_text())
if len(s['train']) != 62 or len(s['validation']) != 10:
    raise SystemExit('This run requires exactly 62 train / 10 validation episodes')
if set(s['train']) & set(s['validation']):
    raise SystemExit('Train/validation overlap')
print('Using reviewed 62/10 split:', p)
PY
trainer="$dataset/training_scripts/train_act_host.py"
name="act72_seed0_$(date +%Y%m%d_%H%M%S)"
python -u "$trainer" --dataset "$dataset" --run "$ACT_ROOT/runs/${name}_trial20" \
  --steps 20 --batch-size 8 --workers 4 --validate-every 10 --save-every 20
run="$ACT_ROOT/runs/${name}_10k"
python -u "$trainer" --dataset "$dataset" --run "$run" \
  --steps 10000 --batch-size 8 --workers 4 --validate-every 250 --save-every 1000 \
  2>&1 | tee "$ACT_ROOT/reports/${name}_10k.log"
python "$dataset/training_scripts/evaluate_act_host.py" --run "$run" --output "$run/prior_evaluation"
echo "TRAINING_COMPLETE $run"
