#!/usr/bin/env bash
set -euo pipefail
source /home/yxy/data/projects/dobot-act/activate.sh
exec python -u /home/yxy/data/projects/dobot-act/training/act_inference_server.py \
  --run /home/yxy/data/projects/dobot-act/runs/act65_seed0_b8_20k_resume_20261113_233602 \
  --checkpoint "${1:-last.pt}"
