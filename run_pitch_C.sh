#!/usr/bin/env bash
set -e

SARAGA_HOME=${SARAGA_HOME:-/scratch/izar/$USER/musico/saraga}
GUITARSET_HOME=${GUITARSET_HOME:-/scratch/izar/$USER/musico/guitarset}
PROJECT_DIR=${PROJECT_DIR:-/home/$USER/musico}

cd "$PROJECT_DIR"
mkdir -p logs results/pitch

source ~/miniforge3/etc/profile.d/conda.sh && conda activate musico

echo "=== Condition C step 1: extract CREPE weights ==="
PYTHONPATH=. python scripts/convert_crepe_weights.py \
    --output results/pitch/crepe_pretrained.pt

echo "=== Condition C step 2: fine-tune on Saraga ==="
PYTHONPATH=. python scripts/train_pitch_carnatic.py \
    --saraga-home    "$SARAGA_HOME" \
    --guitarset-home "$GUITARSET_HOME" \
    --output-dir     results/pitch/C_finetuned_from_crepe \
    --init-checkpoint results/pitch/crepe_pretrained.pt \
    --epochs 30 --max-tracks 60 \
    --compare-all

echo "=== Done ==="
