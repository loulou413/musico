#!/usr/bin/env bash
#SBATCH --job-name=musico-pitch
#SBATCH --output=logs/pitch_%j.out
#SBATCH --error=logs/pitch_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ── adjust these paths to where your data lives on SCITAS ────────────────────
SARAGA_HOME=${SARAGA_HOME:-/scratch/izar/$USER/musico/saraga}
GUITARSET_HOME=${GUITARSET_HOME:-/scratch/izar/$USER/musico/guitarset}
PROJECT_DIR=${PROJECT_DIR:-/home/$USER/musico}

cd "$PROJECT_DIR"
mkdir -p logs results/pitch

# ── load modules (check available versions with: module avail) ────────────────
module purge
module load gcc python cuda cudnn

# ── activate your venv (create it first — see README below) ──────────────────
source ~/miniforge3/etc/profile.d/conda.sh && conda activate musico

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ── Condition A: CREPE pretrained ─────────────────────────────────────────────
echo "=== Condition A ==="
PYTHONPATH=. python scripts/run_pitch_experiments.py \
    --saraga-home    "$SARAGA_HOME" \
    --guitarset-home "$GUITARSET_HOME" \
    --output-dir     results/pitch/A_crepe \
    --max-tracks     60

# ── Condition B: Train from scratch on Carnatic ───────────────────────────────
echo "=== Condition B ==="
PYTHONPATH=. python scripts/train_pitch_carnatic.py \
    --saraga-home    "$SARAGA_HOME" \
    --guitarset-home "$GUITARSET_HOME" \
    --output-dir     results/pitch/B_carnatic_from_scratch \
    --epochs 30 --max-tracks 60 \
    --compare-all

# ── Condition C: Convert CREPE weights then fine-tune ─────────────────────────
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

echo "=== Done! ==="
