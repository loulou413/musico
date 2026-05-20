# Musico

Cross-domain music analysis: beat tracking and pitch estimation across Western and Carnatic music, using [Saraga](https://compmusic.upf.edu/saraga) and [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) / [GTZAN](https://marsyas.info/downloads/data_sets.html) datasets.

## Overview

The project runs three experiments (A, B, C) for each task:

| | Rhythm (beat tracking) | Pitch estimation |
|---|---|---|
| **A** | madmom DBNBeatTracker (Western pretrained) | CREPE (Western pretrained) |
| **B** | CNN+BiLSTM trained from scratch on Saraga | CREPELike trained from scratch on Saraga |
| **C** | Fine-tuned from Western pretrained | Fine-tuned from CREPE weights |

Each condition is evaluated on both Western and Carnatic test sets to measure cross-domain transfer.

## Setup

```bash
pip install -r requirements.txt
```

Download datasets:

```bash
python scripts/download_data.py
```

## Usage

**Rhythm experiments (A, B, C):**
```bash
python scripts/run_rhythm_experiments.py
python scripts/train_beat_carnatic.py          # trains condition B
python scripts/pretrain_beat_western.py        # trains condition C base
```

**Pitch experiments (A, B, C):**
```bash
python scripts/run_pitch_experiments.py
python scripts/train_pitch_carnatic.py         # trains condition B
python scripts/convert_crepe_weights.py        # imports CREPE weights for condition C
```

**Cross-domain aggregation** (reads results from disk, no retraining):
```bash
python scripts/run_cross_domain.py
```

## Structure

```
src/
  pitch/          — CREPE-compatible model, dataset, training, evaluation
  rhythm/         — CNN+BiLSTM beat model, dataset, training, evaluation
  preprocessing/  — audio loading, Saraga/MAESTRO/GTZAN loaders
  cross_domain/   — result aggregation and summary tables
scripts/          — entry points for each experiment
notebooks/        — exploratory analysis and result visualisation
config/           — config.yaml
results/          — output CSVs and JSON summaries (gitignored)
```

## Results layout

```
results/
  rhythm/
    A_madmom/beat_results.csv
    B_carnatic_from_scratch/test_summary.json
    C_finetuned_from_western/test_summary.json
  pitch/
    A_crepe/pitch_results.csv
    B_carnatic_from_scratch/comparison_all.csv
    C_finetuned_from_crepe/comparison_all.csv
```
