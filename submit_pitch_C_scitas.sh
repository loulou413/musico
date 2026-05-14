#!/usr/bin/env bash
#SBATCH --job-name=musico-pitch-C
#SBATCH --output=logs/pitch_C_%j.out
#SBATCH --error=logs/pitch_C_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

SARAGA_HOME=${SARAGA_HOME:-/scratch/izar/$USER/musico/saraga}
GUITARSET_HOME=${GUITARSET_HOME:-/scratch/izar/$USER/musico/guitarset}
PROJECT_DIR=${PROJECT_DIR:-/home/$USER/musico}

cd "$PROJECT_DIR"
mkdir -p logs results/pitch

module purge
module load gcc python cuda cudnn

source ~/miniforge3/etc/profile.d/conda.sh && conda activate musico

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

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
