# Run Tutorial — Carnatic vs Western Music ML Project

This document tells Clara and Cassio exactly what commands to run, in what order, and what to expect.  
Louis handles the pitch side; see the sections marked **[Clara]** and **[Cassio]** for your parts.

---

## 0. Setup (everyone, once)

### 0.1 Create a virtual environment

```bash
cd /path/to/proj
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on TensorFlow (CREPE):** if you are on Apple Silicon or a machine without a GPU, replace `tensorflow>=2.12.0` in `requirements.txt` with `tensorflow-macos` or `tensorflow-cpu` before installing.

### 0.2 Verify the install

```bash
python - <<'EOF'
import librosa, mirdata, madmom, crepe, torch, mir_eval
print("All imports OK")
EOF
```

---

## 1. Download the datasets (everyone, once)

Run these three commands.  Each one downloads audio + annotations into `data/raw/`.  
They can take a while depending on your connection — the datasets are several GB each.

```bash
# Saraga Carnatic (~5 GB)
python scripts/download_data.py --dataset saraga --data-home data/raw/saraga

# GTZAN genre (~1.2 GB)  — used for beat tracking
python scripts/download_data.py --dataset gtzan  --data-home data/raw/gtzan

# MAESTRO (~90 GB full, or annotations only for a quick test)
python scripts/download_data.py --dataset maestro --data-home data/raw/maestro
# If you only want annotations (no audio) to save space:
python scripts/download_data.py --dataset maestro --data-home data/raw/maestro --partial
```

After downloading, each command prints `Done.` if the files are valid.

---

## 2. [Clara] Rhythm / Beat Tracking

Clara runs two experiments:
1. **Baseline** — the Western-trained madmom model on both Carnatic (Saraga) and Western (GTZAN) data.
2. **Carnatic model** — train a CNN+BiLSTM on Saraga beat annotations, then compare it to the baseline.

---

### 2.1 Baseline: madmom on both domains

```bash
python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/rhythm/madmom \
    --backend     madmom \
    --max-tracks  20        # remove this line to run on all tracks
```

**What it produces** in `results/rhythm/madmom/`:

| File | What it is |
|---|---|
| `beat_results.csv` | Per-track scores (F-measure, Cemgil, CMLt, downbeat F) |
| `beat_summary.json` | Mean ± std across all tracks, split by domain |
| `figures/beat_f_measure.png` | Boxplot: Carnatic vs Western |
| `figures/tempo_scatter.png` | Estimated vs reference BPM |
| `drift/<track_id>.png` | Beat drift over time for each track |

**Expected output snippet:**

```
[carnatic] saraga_carnatic/audio_...
    Sam alignment: {'precision': 0.12, 'recall': 0.09, 'f1': 0.10, 'n_hits': 2}
[western] gtzan_genre/audio_...
=== Summary ===
{
  "f_measure": {"mean": 0.71, "std": 0.18, ...},   ← Western
  ...
}
```

The F-measure will be noticeably lower for Carnatic tracks than for Western ones.  That gap is the core result.

---

### 2.2 Train the Carnatic beat model

```bash
python scripts/train_beat_carnatic.py \
    --saraga-home   data/raw/saraga \
    --output-dir    results/rhythm/carnatic_model \
    --epochs        50 \
    --max-tracks    80 \
    --compare-madmom
```

| Argument | What it does |
|---|---|
| `--epochs 50` | Maximum training epochs (early stopping may stop sooner) |
| `--max-tracks 80` | Use 80 Saraga tracks; remove to use all |
| `--compare-madmom` | After training, prints a side-by-side table vs madmom on Carnatic test tracks |

**What it produces** in `results/rhythm/carnatic_model/`:

| File | What it is |
|---|---|
| `best_model.pt` | Saved PyTorch checkpoint (best validation F-measure) |
| `history.json` | Loss + F-measure per epoch |
| `learning_curves.png` | Training / validation curves |
| `test_summary.json` | Carnatic model scores on held-out Saraga test tracks |
| `madmom_comparison.json` | Side-by-side: Carnatic model vs madmom |

**Expected terminal output at the end:**

```
=== Comparison on Carnatic test set ===
Metric                    Carnatic model   Madmom (Western)
------------------------------------------------------------
f_measure                         0.5412             0.3201
cemgil                            0.4890             0.2743
continuity                        0.3900             0.1850
downbeat_f_measure                0.3100             0.0870
```

The Carnatic model should beat madmom on all metrics for Carnatic music.

---

### 2.3 Evaluate the Carnatic model on Western data (transfer test)

This checks whether the Carnatic-trained model generalises back to Western music.

```bash
python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/rhythm/carnatic_on_western \
    --backend     carnatic \
    --carnatic-checkpoint results/rhythm/carnatic_model/best_model.pt \
    --max-tracks  20
```

Compare `beat_summary.json` here vs the one from step 2.1.  
You will likely see that the Carnatic model is worse on Western music — that asymmetry is a key finding.

---

### 2.4 Notebook for figures

Open the rhythm analysis notebook to produce report-ready figures:

```bash
jupyter notebook notebooks/02_rhythm_analysis.ipynb
```

It reads `results/rhythm/madmom/beat_results.csv` automatically.  Run all cells top to bottom.

---

## 3. [Cassio] Cross-Domain Summary

After Clara's rhythm experiments and Louis's pitch experiments are done, run the combined cross-domain script.

### 3.1 Run the cross-domain pipeline

```bash
python scripts/run_cross_domain.py \
    --saraga-home  data/raw/saraga \
    --gtzan-home   data/raw/gtzan \
    --maestro-home data/raw/maestro \
    --output-dir   results/cross_domain \
    --max-tracks   20
```

This runs both the beat tracker and the pitch estimator on both domains in one go and produces a combined summary table.

**What it produces** in `results/cross_domain/`:

| File | What it is |
|---|---|
| `beat_cross_domain.csv` | Per-track beat scores for all tracks, both domains |
| `pitch_cross_domain.csv` | Per-track pitch scores for all tracks, both domains |
| `cross_domain_summary.csv` | Aggregated mean ± std for every metric, grouped by domain and task |

### 3.2 Cross-domain notebook

```bash
jupyter notebook notebooks/04_cross_domain.ipynb
```

This notebook reads the CSVs and produces the grouped bar charts comparing performance across domains and tasks.

---

### 3.3 Full comparison table (after all training is done)

Once Louis has trained the pitch model, run the full 3-model × 2-domain comparison:

```bash
python scripts/train_pitch_carnatic.py \
    --saraga-home  data/raw/saraga \
    --maestro-home data/raw/maestro \
    --output-dir   results/pitch/carnatic_model \
    --compare-all \
    --max-tracks   60
```

This produces `results/pitch/carnatic_model/comparison_summary.csv`:

```
method    domain     RPA_mean  RPA_std  OA_mean  MAE_cents_mean
carnatic  carnatic   …         …        …        …
carnatic  western    …         …        …        …
crepe     carnatic   …         …        …        …
crepe     western    …         …        …        …
pyin      carnatic   …         …        …        …
pyin      western    …         …        …        …
```

That table goes directly into the report.

---

## 4. Quick-test mode (no data yet)

If you want to verify that the pipeline runs end-to-end before the full data download finishes, use `--max-tracks 3` everywhere:

```bash
python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/smoke_test \
    --max-tracks  3

python scripts/train_beat_carnatic.py \
    --saraga-home data/raw/saraga \
    --output-dir  results/smoke_test/beat_model \
    --epochs 2 --max-tracks 5
```

A full run with 3 tracks should finish in a few minutes.

---

## 5. Result files reference

```
results/
├── rhythm/
│   ├── madmom/
│   │   ├── beat_results.csv          ← per-track, madmom on both domains
│   │   ├── beat_summary.json
│   │   └── figures/
│   ├── carnatic_model/
│   │   ├── best_model.pt             ← trained checkpoint
│   │   ├── learning_curves.png
│   │   ├── test_summary.json         ← Carnatic model on Carnatic test
│   │   └── madmom_comparison.json    ← side-by-side
│   └── carnatic_on_western/
│       └── beat_results.csv          ← Carnatic model on Western test
├── pitch/
│   ├── pitch_results.csv             ← CREPE/pyin on both domains
│   ├── pitch_summary.json
│   ├── gamaka_errors.csv             ← Carnatic-specific gamaka analysis
│   ├── carnatic_model/
│   │   ├── best_model.pt
│   │   ├── learning_curves.png
│   │   └── comparison_summary.csv    ← 3 models × 2 domains
│   └── figures/
└── cross_domain/
    ├── beat_cross_domain.csv
    ├── pitch_cross_domain.csv
    └── cross_domain_summary.csv      ← goes into the report
```

---

## 6. Troubleshooting

**`ImportError: No module named 'madmom'`**  
madmom requires Cython at install time: `pip install cython` then `pip install madmom`.

**`CUDA out of memory`**  
Reduce `--batch-size` (e.g. `--batch-size 32` for beat, `--batch-size 64` for pitch).

**`mirdata: dataset not found`**  
Make sure `--saraga-home` points to the directory that was passed to `download_data.py`.  
The path must contain the `saraga_carnatic` subfolder that mirdata creates.

**Training is very slow (no GPU)**  
Use `--max-tracks 20 --epochs 10` for a faster run.  CPU training of the beat model with 20 tracks and 10 epochs takes roughly 15–20 minutes.
