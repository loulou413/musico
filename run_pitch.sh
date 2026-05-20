#!/usr/bin/env bash
# Pitch estimation experiments (Conditions A, B, C)
# Run from the project root: ./run_pitch.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=== GPU check ==="
python -c "
import torch
if torch.cuda.is_available():
    print('CUDA OK —', torch.cuda.get_device_name(0))
else:
    print('WARNING: no GPU found, running on CPU (much slower)')
"

# ── Condition A: CREPE pretrained on both datasets ─────────────────────────────
echo ""
echo "=== Condition A: CREPE pretrained (gives A-W and A-C) ==="
PYTHONPATH=. python scripts/run_pitch_experiments.py \
    --saraga-home    data/raw/saraga \
    --guitarset-home data/raw/guitarset \
    --output-dir     results/pitch/A_crepe \
    --max-tracks     60

# ── Condition B: Train CREPELike from scratch on Saraga ────────────────────────
echo ""
echo "=== Condition B: Train from scratch on Carnatic (gives B-W and B-C) ==="
PYTHONPATH=. python scripts/train_pitch_carnatic.py \
    --saraga-home  data/raw/saraga \
    --guitarset-home data/raw/guitarset \
    --output-dir   results/pitch/B_carnatic_from_scratch \
    --epochs 30 --max-tracks 60 \
    --compare-all

# ── Condition C: Convert CREPE weights, then fine-tune on Saraga ───────────────
echo ""
echo "=== Condition C step 1: Extract CREPE weights from torchcrepe ==="
PYTHONPATH=. python scripts/convert_crepe_weights.py \
    --output results/pitch/crepe_pretrained.pt

echo ""
echo "=== Condition C step 2: Fine-tune on Saraga (gives C-W and C-C) ==="
PYTHONPATH=. python scripts/train_pitch_carnatic.py \
    --saraga-home  data/raw/saraga \
    --guitarset-home data/raw/guitarset \
    --output-dir   results/pitch/C_finetuned_from_crepe \
    --init-checkpoint results/pitch/crepe_pretrained.pt \
    --epochs 30 --max-tracks 60 \
    --compare-all

echo ""
echo "=== All done! Open notebooks/03_pitch_analysis.ipynb to analyse results ==="
