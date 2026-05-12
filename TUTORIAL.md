# Carnatic vs Western Music ML — Run Tutorial

This document tells you **exactly which command to run, from which directory,
and what file each command produces**. Follow it top-to-bottom.

The full experimental design is a **3 × 2 grid**: three model conditions
evaluated on two musical domains.

|   | Western data (GTZAN / MAESTRO) | Carnatic data (Saraga) |
|---|---|---|
| **A. Western pretrained model** | A-W | A-C |
| **B. From-scratch model trained on Carnatic** | B-W | B-C |
| **C. Western pretrained + fine-tuned on Carnatic** | C-W | C-C |

For each task (beat, pitch), you produce all six numbers.

---

## Who runs what

| Person | Task | Sections to run |
|---|---|---|
| **Clara** | Rhythm / beat tracking | §0, §1, §2, §3 (beat parts), §5 |
| **Louis** | Pitch estimation | §0, §1, §2, §4 (pitch parts), §5 |
| **Cassio** | Cross-domain summary | §0, §1, §6 (after Clara + Louis are done) |

---

## §0 — One-time setup (everyone)

### 0.1 Activate the conda environment

The `musico` conda env was already created with all dependencies installed
(madmom from GitHub master, tensorflow-macos, torch, crepe, mirdata, compiam, etc.).

```bash
cd /path/to/musico                    # the project root
conda activate musico
```

If you somehow don't have the env, see [§7 Troubleshooting](#7-troubleshooting).

### 0.2 Always run commands from the project root with `PYTHONPATH=.`

Every `python scripts/…` command in this tutorial **must** be prefixed with
`PYTHONPATH=.` so the `src/` package is importable. Example:

```bash
PYTHONPATH=. python scripts/run_rhythm_experiments.py …
```

If you forget, you'll see `ModuleNotFoundError: No module named 'src'`.

### 0.3 Verify the install

```bash
PYTHONPATH=. python -c "import librosa, mirdata, madmom, crepe, torch, mir_eval, tensorflow; print('All imports OK')"
```

---

## §1 — Download the datasets (everyone, once)

Each command writes to `data/raw/<dataset>/`. They can take a while
(several GB each).

```bash
# Saraga Carnatic (~5 GB)  — used as the Indian/Carnatic data
PYTHONPATH=. python scripts/download_data.py --dataset saraga --data-home data/raw/saraga

# GTZAN genre (~1.2 GB)    — used for Western beat tracking
PYTHONPATH=. python scripts/download_data.py --dataset gtzan --data-home data/raw/gtzan

# MAESTRO annotations-only (~50 MB)  — used for Western pitch estimation
PYTHONPATH=. python scripts/download_data.py --dataset maestro --data-home data/raw/maestro --partial
```

Each command prints `Done.` on success.

> If a download fails partway, just re-run the command — `mirdata` will skip
> files that are already present.

---

## §2 — Quick smoke-test (optional, recommended before full runs)

Add `--max-tracks 3` to any experiment to run on only 3 tracks. A full
pipeline at 3 tracks finishes in a few minutes and tells you whether the
data paths and code are working before you commit to hours of real runs.

```bash
PYTHONPATH=. python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/smoke_test \
    --backend     madmom \
    --max-tracks  3
```

If this finishes without errors, proceed to the real runs.

---

## §3 — RHYTHM / BEAT TRACKING  [Clara]

You'll produce three beat models and evaluate each on both domains.

### 3.1 — Condition A: Western pretrained (madmom)

The **madmom** library ships a CNN+DBN beat tracker pretrained on Western
data. We run it as-is on both Western (GTZAN) and Carnatic (Saraga) data.

```bash
PYTHONPATH=. python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/rhythm/A_madmom \
    --backend     madmom
```

**Produces in `results/rhythm/A_madmom/`:**

| File | Contents |
|---|---|
| `beat_results.csv` | Per-track scores (F-measure, Cemgil, CMLt, downbeat F) |
| `beat_summary.json` | Mean ± std across all tracks, **split by domain (carnatic vs western)** |
| `figures/beat_f_measure.png` | Boxplot: Carnatic vs Western |
| `figures/tempo_scatter.png` | Estimated vs reference BPM |

> This single command produces both **A-W** (madmom on Western)
> and **A-C** (madmom on Carnatic). Look at the `domain` column / key
> in the output files.

### 3.2 — Condition B: Train a beat model from scratch on Carnatic

This trains the project's own `BeatActivationModel` (CNN + BiLSTM) on
Saraga from random initialisation.

```bash
PYTHONPATH=. python scripts/train_beat_carnatic.py \
    --saraga-home data/raw/saraga \
    --output-dir  results/rhythm/B_carnatic_from_scratch \
    --epochs 50 --max-tracks 80 \
    --compare-madmom
```

**Produces in `results/rhythm/B_carnatic_from_scratch/`:**

| File | Contents |
|---|---|
| `best_model.pt` | PyTorch checkpoint (best validation F-measure) |
| `history.json` | Loss + val F-measure per epoch |
| `learning_curves.png` | Train/val curves |
| `test_summary.json` | **B-C**: Carnatic model on Carnatic test |
| `madmom_comparison.json` | Side-by-side B-C vs A-C |

Then evaluate **B-W** (the from-scratch Carnatic model on Western data):

```bash
PYTHONPATH=. python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/rhythm/B_on_western \
    --backend     carnatic \
    --carnatic-checkpoint results/rhythm/B_carnatic_from_scratch/best_model.pt
```

The `beat_summary.json` in `results/rhythm/B_on_western/` gives you **B-W**.

### 3.3 — Condition C: Pretrain on Western, fine-tune on Carnatic

madmom isn't a PyTorch module so we can't fine-tune its weights directly.
Instead we pretrain the same `BeatActivationModel` architecture on GTZAN
beats, then fine-tune that checkpoint on Saraga.

**Step C.1 — Pretrain on GTZAN:**

```bash
PYTHONPATH=. python scripts/pretrain_beat_western.py \
    --gtzan-home data/raw/gtzan \
    --output-dir results/rhythm/C_western_pretrained \
    --epochs 30
```

Produces `results/rhythm/C_western_pretrained/best_model.pt` — the
Western-pretrained checkpoint.

**Step C.2 — Fine-tune on Saraga:**

```bash
PYTHONPATH=. python scripts/train_beat_carnatic.py \
    --saraga-home data/raw/saraga \
    --output-dir  results/rhythm/C_finetuned_from_western \
    --init-checkpoint results/rhythm/C_western_pretrained/best_model.pt \
    --epochs 50 --max-tracks 80 \
    --compare-madmom
```

> When `--init-checkpoint` is set, the learning rate is automatically
> dropped to 1e-4 (10× lower than from-scratch). This is the standard
> rule for fine-tuning so the pretrained features aren't wrecked.

**Produces in `results/rhythm/C_finetuned_from_western/`:**

| File | Contents |
|---|---|
| `best_model.pt` | Fine-tuned checkpoint |
| `test_summary.json` | **C-C**: fine-tuned model on Carnatic test |
| `madmom_comparison.json` | Side-by-side C-C vs A-C |
| `learning_curves.png` | Train/val curves |

**Step C.3 — Evaluate C on Western data (C-W):**

```bash
PYTHONPATH=. python scripts/run_rhythm_experiments.py \
    --saraga-home data/raw/saraga \
    --gtzan-home  data/raw/gtzan \
    --output-dir  results/rhythm/C_on_western \
    --backend     carnatic \
    --carnatic-checkpoint results/rhythm/C_finetuned_from_western/best_model.pt
```

`beat_summary.json` here = **C-W**.

### 3.4 — Beat analysis notebook

```bash
jupyter notebook notebooks/02_rhythm_analysis.ipynb
```

It reads `results/rhythm/A_madmom/beat_results.csv` by default.
Edit the path at the top of the notebook to swap in B or C results.

---

## §4 — PITCH ESTIMATION  [Louis]

Same three-condition design as rhythm, but the Western pretrained model
is **CREPE** (Kim et al. 2018) and we have a faithful PyTorch port that
can load CREPE's official weights for fine-tuning.

### 4.1 — Condition A: Western pretrained (CREPE)

```bash
PYTHONPATH=. python scripts/run_pitch_experiments.py \
    --saraga-home  data/raw/saraga \
    --maestro-home data/raw/maestro \
    --output-dir   results/pitch/A_crepe
```

**Produces in `results/pitch/A_crepe/`:**

| File | Contents |
|---|---|
| `pitch_results.csv` | Per-track RPA, OA, MAE cents, by domain |
| `pitch_summary.json` | Mean ± std, **split by domain** |
| `gamaka_errors.csv` | Carnatic-specific gamaka analysis |
| `figures/` | Boxplots, scatter, etc. |

This single run gives you **A-W** and **A-C**.

### 4.2 — Condition B: Train pitch model from scratch on Carnatic

```bash
PYTHONPATH=. python scripts/train_pitch_carnatic.py \
    --saraga-home  data/raw/saraga \
    --maestro-home data/raw/maestro \
    --output-dir   results/pitch/B_carnatic_from_scratch \
    --epochs 30 --max-tracks 60 \
    --compare-all
```

The `--compare-all` flag runs the from-scratch model **plus CREPE plus pyin**
on the test sets of both domains, producing the 3-models × 2-domains table.

**Produces in `results/pitch/B_carnatic_from_scratch/`:**

| File | Contents |
|---|---|
| `best_model.pt` | Trained checkpoint |
| `history.json` | Train/val loss + RPA per epoch |
| `learning_curves.png` | Loss / RPA / OA curves |
| `test_carnatic_summary.json` | **B-C** |
| `comparison_all.csv` | Every method × every test track |
| `comparison_summary.csv` | Aggregated by method × domain (**includes B-W and B-C**) |

### 4.3 — Condition C: Load real CREPE weights, fine-tune on Saraga

**Step C.1 — Convert Keras CREPE weights → PyTorch checkpoint** (one-time):

```bash
PYTHONPATH=. python scripts/convert_crepe_weights.py \
    --output results/pitch/crepe_pretrained.pt
```

This loads the official `crepe` Keras model, copies every conv/BN/dense
weight into a PyTorch `state_dict`, and **verifies** that running a random
input through both Keras and the PyTorch port produces identical outputs
(max diff < 1e-4). If verification fails the script aborts.

**Step C.2 — Fine-tune on Saraga:**

```bash
PYTHONPATH=. python scripts/train_pitch_carnatic.py \
    --saraga-home  data/raw/saraga \
    --maestro-home data/raw/maestro \
    --output-dir   results/pitch/C_finetuned_from_crepe \
    --init-checkpoint results/pitch/crepe_pretrained.pt \
    --epochs 30 --max-tracks 60 \
    --compare-all
```

**Produces in `results/pitch/C_finetuned_from_crepe/`:**

Same file set as 4.2. `comparison_summary.csv` here contains **C-W** and **C-C**.

### 4.4 — Pitch analysis notebook

```bash
jupyter notebook notebooks/03_pitch_analysis.ipynb
```

---

## §5 — Where every number in the report comes from

After running §3 and §4, your six headline numbers per task live here:

### Beat (F-measure):

| Cell | File | Look for |
|---|---|---|
| A-W | `results/rhythm/A_madmom/beat_summary.json` | `f_measure.mean` in the `western` block |
| A-C | `results/rhythm/A_madmom/beat_summary.json` | `f_measure.mean` in the `carnatic` block |
| B-W | `results/rhythm/B_on_western/beat_summary.json` | `f_measure.mean` (western) |
| B-C | `results/rhythm/B_carnatic_from_scratch/test_summary.json` | `f_measure.mean` |
| C-W | `results/rhythm/C_on_western/beat_summary.json` | `f_measure.mean` (western) |
| C-C | `results/rhythm/C_finetuned_from_western/test_summary.json` | `f_measure.mean` |

### Pitch (RPA):

| Cell | File | Look for |
|---|---|---|
| A-W | `results/pitch/A_crepe/pitch_summary.json` | `raw_pitch_accuracy.mean` (western) |
| A-C | `results/pitch/A_crepe/pitch_summary.json` | `raw_pitch_accuracy.mean` (carnatic) |
| B-W, B-C | `results/pitch/B_carnatic_from_scratch/comparison_summary.csv` | `method=carnatic` rows |
| C-W, C-C | `results/pitch/C_finetuned_from_crepe/comparison_summary.csv` | `method=carnatic` rows |

---

## §6 — Cross-domain aggregation  [Cassio]

### What this step does (and doesn't do)

The cross-domain step **does not retrain anything**. It reads the per-condition
result files that Clara (§3) and Louis (§4) produced, stitches them into one
unified long-form table, and prints the 3 × 2 summary per task and metric.

So Cassio's role is the **aggregator + analyst**, not a re-runner. The script
is cheap (milliseconds) and can be re-run any time new results land on disk.

### When can Cassio start?

- **Any time** — `run_cross_domain.py` simply reports which result files are
  missing and proceeds with what it finds. You can run it after only condition A
  is done to see the current 3 × 2 grid filling in.
- **No retraining** ever happens here, regardless of order.

### Run it

```bash
PYTHONPATH=. python scripts/run_cross_domain.py \
    --results-root results \
    --output-dir   results/cross_domain
```

The script first lists which expected files were found vs missing:

```
Discovering result files…
  [✓]       beat_A     results/rhythm/A_madmom/beat_results.csv
  [MISSING] beat_B_W
  [✓]       beat_B_C   results/rhythm/B_carnatic_from_scratch/test_summary.json
  …
```

Add `--strict` if you want it to crash on any missing file instead of warning.

### What it produces in `results/cross_domain/`

| File | Contents |
|---|---|
| `all_results_long.csv` | Long-form table: one row per (task, condition, domain, track, metric, value). The raw input for any notebook figure. |
| `cross_domain_summary.csv` | Aggregated mean ± std for every (task, metric, condition, domain) — **this is the table that goes into the report**. |

It also prints the 3 × 2 pivots to the terminal so you can sanity-check at a glance:

```
--- beat | f_measure ---
domain     western  carnatic
condition
A           0.7811    0.4203
B           0.5523    0.6087
C           0.6612    0.6791

--- pitch | raw_pitch_accuracy ---
…
```

### Where each input file comes from

| Aggregator label | Source script (§) | File on disk |
|---|---|---|
| `beat_A` | §3.1 | `results/rhythm/A_madmom/beat_results.csv` |
| `beat_B_W` | §3.2 step 2 | `results/rhythm/B_on_western/beat_results.csv` |
| `beat_B_C` | §3.2 step 1 | `results/rhythm/B_carnatic_from_scratch/test_summary.json` |
| `beat_C_W` | §3.3 step C.3 | `results/rhythm/C_on_western/beat_results.csv` |
| `beat_C_C` | §3.3 step C.2 | `results/rhythm/C_finetuned_from_western/test_summary.json` |
| `pitch_A` | §4.1 | `results/pitch/A_crepe/pitch_results.csv` |
| `pitch_B` | §4.2 | `results/pitch/B_carnatic_from_scratch/comparison_all.csv` |
| `pitch_C` | §4.3 step C.2 | `results/pitch/C_finetuned_from_crepe/comparison_all.csv` |

### Notebook for figures

```bash
jupyter notebook notebooks/04_cross_domain.ipynb
```

The notebook reads `results/cross_domain/all_results_long.csv` and produces
grouped bar charts comparing the three conditions on each domain × task ×
metric. Edit it freely — it's the place where the report figures get shaped.

---

## §7 — Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
You forgot `PYTHONPATH=.`. Always prefix every script with it.

**`No module named 'madmom'` / `tensorflow` / `crepe`**
You're not in the `musico` conda env. Run `conda activate musico`.

**`mirdata: dataset not found`**
The `--saraga-home` (or `--gtzan-home`, `--maestro-home`) path must point
at the same directory you passed to `download_data.py`. The path must
contain the dataset's subfolder that mirdata created.

**`CUDA out of memory`**
Reduce `--batch-size` (e.g. 32 for beat, 64 for pitch). On Apple Silicon
there's no CUDA; everything runs on CPU automatically.

**Training is very slow (CPU only)**
Use `--max-tracks 20 --epochs 10` for a faster run. CPU training of the
beat model at 20 tracks × 10 epochs ≈ 15–20 minutes.

**CREPE weight conversion fails verification**
The bit-exact port lives in `src/pitch/model.py`. If you change anything
in that file (architecture, padding, ReLU order, flatten order), the
verification in `convert_crepe_weights.py` will reject the conversion.
Don't edit that file.

**Fine-tuning seems to be making things worse**
Check that `--init-checkpoint` actually printed `Loading initial weights
from …` at the start. If the LR wasn't dropped to 1e-4 automatically,
pass `--lr 1e-4` explicitly.

---

## §8 — Full directory layout after everything runs

```
results/
├── rhythm/
│   ├── A_madmom/                              ← Western pretrained, both domains
│   │   ├── beat_results.csv, beat_summary.json, figures/
│   ├── B_carnatic_from_scratch/               ← B-C
│   │   ├── best_model.pt, history.json, test_summary.json,
│   │   │   madmom_comparison.json, learning_curves.png
│   ├── B_on_western/                          ← B-W
│   │   └── beat_results.csv, beat_summary.json
│   ├── C_western_pretrained/                  ← intermediate: GTZAN-pretrained ckpt
│   │   └── best_model.pt, learning_curves.png
│   ├── C_finetuned_from_western/              ← C-C
│   │   ├── best_model.pt, test_summary.json,
│   │   │   madmom_comparison.json, learning_curves.png
│   └── C_on_western/                          ← C-W
│       └── beat_results.csv, beat_summary.json
├── pitch/
│   ├── A_crepe/                               ← A-W and A-C
│   │   ├── pitch_results.csv, pitch_summary.json,
│   │   │   gamaka_errors.csv, figures/
│   ├── crepe_pretrained.pt                    ← converted CREPE weights (input to C)
│   ├── B_carnatic_from_scratch/               ← B-W and B-C
│   │   ├── best_model.pt, comparison_summary.csv, …
│   └── C_finetuned_from_crepe/                ← C-W and C-C
│       └── best_model.pt, comparison_summary.csv, …
└── cross_domain/
    ├── beat_cross_domain.csv
    ├── pitch_cross_domain.csv
    └── cross_domain_summary.csv               ← goes into the report
```
