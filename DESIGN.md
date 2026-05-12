# Project Design — What the Code Does and Why

A companion to `TUTORIAL.md`. The tutorial tells you *how to run* the
pipeline; this document tells you *what every part of the codebase is for,
why we made the design choices we did, and how the experimental conditions
map onto specific files*.

Read this if you want to:
- understand the research question we're answering,
- modify the architecture or training procedure,
- write the methodology section of the report,
- onboard onto the project for the first time.

---

## Table of contents

1. [The research question](#1-the-research-question)
2. [Why these two tasks (beat + pitch)](#2-why-these-two-tasks-beat--pitch)
3. [Why these datasets](#3-why-these-datasets)
4. [The 3 × 2 experimental design](#4-the-3--2-experimental-design)
5. [Why we use CREPELike (and what it actually is)](#5-why-we-use-crepelike-and-what-it-actually-is)
6. [Why we picked fine-tuning over from-scratch as the third condition](#6-why-we-picked-fine-tuning-over-from-scratch-as-the-third-condition)
7. [How we handle the fact that madmom can't be fine-tuned](#7-how-we-handle-the-fact-that-madmom-cant-be-fine-tuned)
8. [Codebase map — file by file](#8-codebase-map--file-by-file)
9. [The CREPE weight conversion in detail](#9-the-crepe-weight-conversion-in-detail)
10. [Training-loop design choices](#10-training-loop-design-choices)
11. [Evaluation metrics and why](#11-evaluation-metrics-and-why)
12. [Caveats, limits, and things we didn't do](#12-caveats-limits-and-things-we-didnt-do)

---

## 1. The research question

> **Western-trained MIR (Music Information Retrieval) models are the de facto
> standard. How well do they actually work on non-Western music, specifically
> South Indian (Carnatic) classical music? And if we retrain or fine-tune
> them on Carnatic data, what changes?**

The implicit assumption in most off-the-shelf MIR tools — madmom, CREPE,
pyin, librosa — is that Western tonal/metrical structure is universal.
Carnatic music violates several of those assumptions:

- **Rhythm**: Carnatic *talas* include cycles of 7, 8, 9, 11 beats, with
  internal subdivisions that don't map onto 4/4 or 3/4. Madmom's DBN
  transitions are tuned for Western metrical hierarchies.
- **Pitch**: Carnatic uses 22 *shruti* microtones, continuous *gamaka*
  ornaments (oscillating, sliding pitches), and improvisation around a
  drone. CREPE was trained on isolated notes and well-tempered Western
  performances — its bias is toward stable, semitone-quantised pitch.

Our project measures **how much** these mismatches hurt, and **whether
training on Carnatic data fixes it** — either from scratch or by starting
from Western weights and adapting.

---

## 2. Why these two tasks (beat + pitch)

Beat tracking and pitch estimation are the two most fundamental,
well-defined low-level MIR tasks. Both have:

- A clearly defined output format (beat times in seconds; F0 in Hz per frame).
- Reliable evaluation metrics (`mir_eval.beat`, `mir_eval.melody`).
- A reference Western-pretrained model that everyone in the field uses
  (madmom for beat, CREPE for pitch).
- A Carnatic dataset with ground-truth annotations (Saraga has both beat
  times and per-frame pitch from compIAM's research group).

If a Western model fails on Carnatic even at *these* basic tasks, that's
a strong claim about the limits of cross-cultural transfer. If it
succeeds, that's also interesting — it would mean low-level features
transfer across musical cultures even when high-level structure doesn't.

We deliberately did not tackle higher-level tasks (chord recognition,
genre classification, melody extraction with source separation) because
they introduce too many confounding factors to attribute the result
cleanly to the Western/Carnatic distinction.

---

## 3. Why these datasets

| Dataset | Role | Why this one |
|---|---|---|
| **Saraga (Carnatic)** | Carnatic data for all three conditions | The standard open Carnatic dataset, curated by the compIAM / MTG group. Has per-track beat annotations (`beats`) and per-frame pitch annotations (`pitch`) extracted by hand-verified algorithms. ~200 tracks, several hours each. |
| **GTZAN genre** | Western data for beat tracking | Public, widely used, has beat annotations for 10 genres × 100 tracks. The Western beat benchmark of choice. |
| **MAESTRO** | Western data for pitch | Classical piano performances with aligned MIDI ground truth — gives perfect pitch labels for evaluation. We use the `--partial` (annotations-only) variant to avoid downloading 90 GB of audio. |

We picked these specifically because the same Python library — `mirdata` —
provides unified loaders for all three, so the preprocessing layer
(`src/preprocessing/`) is symmetric across domains.

---

## 4. The 3 × 2 experimental design

The core deliverable is a **3-condition × 2-domain** table per task:

|   | Western data | Carnatic data |
|---|---|---|
| **A. Western pretrained model** | A-W (baseline) | A-C (transfer failure mode) |
| **B. From-scratch Carnatic model** | B-W (reverse transfer) | B-C (Carnatic-only ceiling) |
| **C. Western pretrained + fine-tuned on Carnatic** | C-W (catastrophic forgetting check) | C-C (transfer success?) |

Each cell answers a specific question:

- **A-W vs A-C** — the *domain gap*. How much worse is the Western model
  on out-of-distribution Carnatic data?
- **B-C vs A-C** — does *any* Carnatic training help? (Almost certainly
  yes, but how much?)
- **C-C vs B-C** — does **Western pretraining help** when fine-tuning, or
  does it hurt? This is the most interesting comparison scientifically.
  - If C-C > B-C: low-level features transfer across cultures → cultures
    share more structure than people assume.
  - If B-C > C-C: Western pretraining injects an inductive bias that
    actively hurts Carnatic modelling → strong argument for
    culturally-specific MIR.
- **C-W vs A-W** — *catastrophic forgetting*. Does fine-tuning on
  Carnatic destroy Western performance?
- **B-W vs A-W** — confirms that a Carnatic-only model can't substitute
  for the Western one.

The 3 × 2 grid lets us tell a complete transfer-learning story rather
than just "Western models are bad at Carnatic."

---

## 5. Why we use CREPELike (and what it actually is)

### What is CREPE?

CREPE (Kim, Salamon, Li, Bello, 2018) is the standard deep-learning
pitch estimator. It's a 6-layer 2D CNN that takes a single 64 ms audio
frame at 16 kHz (1024 samples) and outputs a 360-dim probability vector
over a 20-cents-per-bin pitch grid spanning ~33 Hz to ~2 kHz. The official
implementation is in Keras and ships with pretrained weights on the
MDB-stem-synth dataset (Western pop, jazz, classical).

### Why CREPE specifically?

Three reasons:

1. **It's the field-standard pretrained Western pitch model.** Comparing
   our Carnatic model to CREPE = comparing to what every MIR researcher
   would use by default.
2. **The output space (360 bins, 20 cents wide) is fine enough to capture
   gamakas.** Carnatic gamakas can move ±100 cents around a target pitch;
   a 20-cents grid resolves that without quantising it to semitones.
3. **The architecture is small enough (~22 M params) to train and
   fine-tune on a single CPU/laptop GPU.**

### What is CREPELike?

`CREPELike` is **a PyTorch reimplementation of CREPE 'full' that's
byte-for-byte compatible with the official Keras weights.** It lives in
`src/pitch/model.py`. Specifically:

- 6 Conv2D blocks with filter counts `[1024, 128, 128, 128, 256, 512]`,
  kernel `(K, 1)` (effectively 1D over time), with TF-style "same"
  padding (asymmetric when needed).
- Each block: `Conv(ReLU) → BatchNorm → MaxPool(2,1) → Dropout` —
  **note that ReLU is *before* BN**, matching Keras's
  `Conv2D(activation='relu')` convention. This is unusual in PyTorch
  (`Conv → BN → ReLU` is more common) but it's what CREPE does.
- After the last block, a `Permute((2,1,3))` followed by `Flatten`, then
  a `Dense(360, sigmoid)` classifier.

We verify in `scripts/convert_crepe_weights.py` that running a random
input through Keras CREPE and through `CREPELike` (with converted
weights) produces outputs that match to within **~1e-7** — essentially
fp32 precision. The model is a true port, not "inspired by."

### Why was this necessary?

The previous `CREPELike` (before this work) was a *loose* CREPE-style
CNN with different filter counts, different layer ordering (Conv → BN →
ReLU), 1D convs instead of (H, 1) 2D convs, and no facility for loading
CREPE's weights. To run condition C (load Western weights, fine-tune
on Carnatic), the PyTorch model **must** be bit-exact with CREPE,
otherwise loading the weights would produce nonsense outputs and the
"fine-tuning" experiment would be meaningless.

---

## 6. Why we picked fine-tuning over from-scratch as the third condition

The user's original two-step plan was:

1. Try Western models on Carnatic, retrain on Carnatic, compare.
2. Train without Western pretraining to see if results survive.

That gives you A/B. The natural third question is: **what if we start
from Western weights and only adapt to Carnatic?** This is the standard
transfer-learning setup in deep learning.

The decision rests on two empirical points from the literature:

1. **Saraga is small** (~200 tracks). Small datasets benefit
   disproportionately from pretraining, because most of what a CNN needs
   to learn (low-level spectral features, onset detection) can be
   learned from any music — Western or not. Pretraining on a large
   Western dataset, then specialising on Carnatic, almost always beats
   training from scratch on small data.
2. **Low-level features are largely culture-agnostic.** A beat is a
   transient onset. A pitched note is a harmonic series. The
   filters that detect these don't care if the music is Carnatic or
   Western. The places culture matters — meter type, scale, ornaments —
   are higher-level and learned in the later layers or post-processing.

So fine-tuning should win on B-C vs C-C. *Should*. The whole point of
the experiment is to measure whether it actually does. If it doesn't,
that's a publishable negative result.

### Why lower the learning rate during fine-tuning?

When `--init-checkpoint` is set in either training script, we
automatically drop the LR from `1e-3` to `1e-4` (10× lower). This is the
standard rule for fine-tuning. With a high LR, the first few gradient
steps would erase the pretrained features before they have a chance to
adapt — you'd end up close to "from scratch on Saraga" rather than
"transfer learning."

---

## 7. How we handle the fact that madmom can't be fine-tuned

This is the awkward part of the project, and it's worth being clear
about it.

**Madmom** is the field-standard Western beat tracker. It is **not** a
single PyTorch model — it's a pipeline of (a) a Convolutional Neural
Network beat activation predictor, whose weights are stored as pickled
NumPy arrays inside the `madmom` Python package, and (b) a Dynamic
Bayesian Network (HMM) post-processor that turns the activation into
beat times under metrical constraints.

You can't load madmom's CNN weights into a PyTorch model because:

1. The weights are in madmom's internal format, not Keras/TF/PyTorch.
2. The architecture is defined imperatively in madmom's source code with
   custom layer classes.
3. Even if you ported it, you'd still need to reimplement the DBN
   post-processor.

So our condition C for beat is: **pretrain the project's own
`BeatActivationModel` on a Western beat dataset (GTZAN), then fine-tune
on Saraga.** This means C-beat is not "fine-tune madmom on Carnatic"
but "fine-tune a Western-pretrained beat model on Carnatic, of an
architecture comparable to madmom's." For an academic comparison this
is fair and standard — what matters is that the *training data* for the
pretraining is Western, not that the architecture is madmom's exact one.

For **pitch**, we don't have this problem: CREPE is a clean Keras model
with public weights, and we built `CREPELike` to be a faithful port. So
C-pitch *is* "fine-tune the actual Western CREPE model on Carnatic."

---

## 8. Codebase map — file by file

### `src/preprocessing/`  — dataset loaders

Every dataset goes through a single `AudioTrack` dataclass
(`src/preprocessing/common.py`) with fields `track_id, audio, sr,
domain ('western'|'carnatic'), beat_times, pitch_times, pitch_hz,
metadata`. Downstream code only ever touches `AudioTrack`, so adding a
new dataset is a matter of writing one new loader.

| File | What it does |
|---|---|
| `common.py` | `AudioTrack` dataclass + audio loading/normalisation helpers. |
| `saraga.py` | Loads Saraga Carnatic tracks via mirdata. Extracts beats from the `sama_section` annotation (sama = first beat of the cycle) and from `beats` if present. Pitch from the per-frame pitch CSV. |
| `gtzan.py` | Loads GTZAN tracks via mirdata. Beats come from the GTZAN-Rhythm extension. |
| `maestro.py` | Loads MAESTRO. Pitch ground truth is derived from MIDI note-on/off events. |

### `src/rhythm/`  — beat tracking

| File | What it does |
|---|---|
| `model.py` | `BeatActivationModel`: a 2-block CNN + 2-layer BiLSTM + linear head. Input: log-mel spectrogram `(B, n_mels=128, T)`. Output: per-frame beat probability `(B, T)`. `activation_to_beat_times` does simple peak-picking; for higher accuracy you can swap in madmom's DBN. |
| `dataset.py` | `SaragaBeatDataset` — splits each track into fixed-length (~12 s) mel-spectrogram segments paired with Gaussian-smeared per-frame beat labels. The Gaussian smearing widens each beat label by ~3 frames so the model gets a smoother gradient signal than hard 0/1 targets. |
| `train.py` | Training loop with weighted BCE (positive weight 10× since beats are ~5% of frames), Adam, gradient clipping, `ReduceLROnPlateau`, early stopping. Saves the best checkpoint by validation F-measure. |
| `beat_tracking.py` | `run_beat_tracking(track, backend=...)` — unified inference interface. `backend='madmom'` uses the Western pretrained model; `backend='carnatic'` loads a `BeatActivationModel` from `--carnatic-checkpoint`. |
| `evaluation.py` | Wraps `mir_eval.beat.evaluate` to compute F-measure, Cemgil, CMLt, downbeat F1, etc. |
| `analysis.py` | DataFrame helpers + plotting (boxplots by domain, tempo scatter, drift over time). |

### `src/pitch/`  — pitch estimation

| File | What it does |
|---|---|
| `model.py` | `CREPELike` — bit-exact PyTorch port of CREPE 'full' (see §5 and §9). Also `activation_to_hz` (weighted-mean decoding, good for gamakas) and `activation_to_hz_viterbi` (penalises jumpy pitch tracks). |
| `dataset.py` | `SaragaPitchDataset` — frames audio at 16 kHz, 1024-sample windows, 10 ms hop. Each frame → Gaussian soft label over the 360-bin pitch grid. Defines `PITCH_BINS_HZ`, `PITCH_BINS_CENTS`, `TARGET_SR=16000`, `FRAME_LEN=1024`, `HOP_SAMPLES=160` — all matching CREPE so labels are directly comparable. |
| `train.py` | Training loop with masked BCE (only voiced frames contribute to loss), Adam, early stopping. Best checkpoint by validation Raw Pitch Accuracy. |
| `estimation.py` | `run_pitch_estimation(track, method=...)` — unified interface. `method='crepe'` uses the official TF CREPE; `method='pyin'` uses librosa's pyin; `method='carnatic'` loads a `CREPELike` from a checkpoint. |
| `evaluation.py` | `mir_eval.melody.evaluate` wrapper — RPA, OA, MAE cents, semitone-snap ratio (a Carnatic-specific metric: fraction of voiced frames within ±25 cents of a 12-TET note, useful for detecting gamaka mishandling). |
| `analysis.py` | Plotting helpers. |

### `src/cross_domain/`  — aggregation, not re-running

| File | What it does |
|---|---|
| `experiments.py` | **Reads** the per-condition result files produced by §3 and §4 from disk and stitches them into one long-form DataFrame and a 3 × 2 summary table. Crucially, it does *not* re-run models — Cassio's role is to aggregate Clara's and Louis's outputs, not to retrain anything. Missing input files emit a warning but don't crash, so the aggregator can be re-run any time new results land. Exposes `load_all_results`, `summarise`, `pivot_table`, `discover_results`. |

### `scripts/`  — entry points

| File | Purpose |
|---|---|
| `download_data.py` | Wraps mirdata's download + validate for each dataset. |
| `run_rhythm_experiments.py` | Runs a chosen beat backend (`madmom` or `carnatic`) on both domains. Used for conditions **A** (with `--backend madmom`), **B-W** and **C-W** (with `--backend carnatic --carnatic-checkpoint …`). |
| `run_pitch_experiments.py` | Same but for pitch (`crepe` and `pyin` baselines). |
| `train_beat_carnatic.py` | Trains `BeatActivationModel` on Saraga. With `--init-checkpoint` = fine-tuning (condition **C**). Without = from scratch (condition **B**). |
| `train_pitch_carnatic.py` | Same for pitch with `CREPELike`. `--init-checkpoint results/pitch/crepe_pretrained.pt` = fine-tune real CREPE; without = from scratch. |
| `pretrain_beat_western.py` | **Step 1 of beat condition C.** Trains `BeatActivationModel` on GTZAN beats to produce the Western checkpoint that `train_beat_carnatic.py --init-checkpoint` consumes. |
| `convert_crepe_weights.py` | **Step 1 of pitch condition C.** Loads the official Keras CREPE 'full' model, copies its weights into a `CREPELike` PyTorch `state_dict`, verifies bit-exact output equivalence, and saves the `.pt`. |
| `run_cross_domain.py` | Cassio's aggregation step. Reads Clara's and Louis's result files from disk, prints the 3 × 2 summary tables, and writes `results/cross_domain/all_results_long.csv` + `cross_domain_summary.csv` for the report. Does not retrain. Tolerates missing files (warns) so it can be run incrementally. |

### `notebooks/`

| File | Purpose |
|---|---|
| `00_data_exploration.ipynb` | Sanity-check loaded tracks, plot example mel spectrograms and pitch contours. |
| `02_rhythm_analysis.ipynb` | Per-track beat results, boxplots, tempo scatter. |
| `03_pitch_analysis.ipynb` | Per-track pitch results, gamaka analysis, semitone-snap distributions. |
| `04_cross_domain.ipynb` | The combined figure-producing notebook — bar charts of every metric × domain × condition. |

### `config/config.yaml`

Centralised hyperparameters: audio sample rate, hop length, mel params,
madmom DBN settings, CREPE inference options, etc. Most scripts read
defaults from here.

---

## 9. The CREPE weight conversion in detail

`scripts/convert_crepe_weights.py` is the most subtle piece of the
codebase, because **a wrong conversion is silent** — the model still
runs, just with garbage outputs that look plausible. We verify
correctness by comparing outputs to the original Keras model.

### Three transformations are needed

1. **Conv2D weight layout.** Keras stores convolutional kernels as
   `(H, W, in_channels, out_channels)`. PyTorch expects
   `(out_channels, in_channels, H, W)`. Conversion:
   `np.transpose(W, (3, 2, 0, 1))`.

2. **BatchNorm parameters.** Keras `get_weights()` returns
   `[gamma, beta, moving_mean, moving_var]` for each BN layer; these map
   one-to-one to PyTorch's `weight, bias, running_mean, running_var`.
   The `num_batches_tracked` buffer is set to 0 (it's not used at
   inference time anyway).

3. **Dense weight layout.** Keras stores `Dense` weights as `(in, out)`;
   PyTorch's `Linear` expects `(out, in)`. Conversion: `.T`.

### Three architectural gotchas

These are the things that broke during development and would silently
make the conversion wrong if not handled:

1. **TF "same" padding with stride > 1.** When `stride=4, kernel=512`
   on input length 1024, TF computes `total_pad = (out-1)*stride +
   kernel - in = 508`, splits it `pad_left = total // 2 = 254`,
   `pad_right = 254`. PyTorch's `padding='same'` doesn't support
   `stride > 1` cleanly across all versions, so we compute and apply
   the asymmetric pad manually with `F.pad`. The helper
   `_tf_same_pad_1d(in_len, kernel, stride)` does this.

2. **ReLU is *before* BatchNorm.** Keras `Conv2D(activation='relu')`
   applies ReLU inside the conv layer, so the order is
   `Conv → ReLU → BN → MaxPool → Dropout`. PyTorch convention is usually
   `Conv → BN → ReLU`. We deliberately use the Keras order, otherwise
   loaded weights produce wrong intermediate distributions.

3. **The `Permute((2,1,3))` before `Flatten`.** This is the one that
   bit hardest. Keras CREPE has, after the last conv:
   - Shape `(B, H=4, W=1, C=512)` channels-last.
   - `Permute(dims=(2,1,3))` reorders the non-batch axes as
     `(W, H, C) = (1, 4, 512)`.
   - `Flatten` in row-major: `[h0_c0, h0_c1, …, h0_c511, h1_c0, …]`.
     So the **H index is outer, C is inner**.

   In PyTorch with channels-first `(B, C, H, W)`, the equivalent is
   `x.permute(0, 2, 3, 1)` (→ `(B, H, W, C)`) then `.view(B, -1)`. A
   naïve `.view(B, -1)` directly on the channels-first tensor produces
   *correct shape but wrong order*, and the classifier weights expect
   the Keras order.

### The verification step

`convert_crepe_weights.py` runs a random batch through both Keras CREPE
and the converted PyTorch model and asserts `max |diff| < 1e-4`. We
actually observe `~1e-7`. If you change anything in `src/pitch/model.py`,
re-run the converter and it will refuse to save if outputs no longer
match. This guards against silent breakage.

---

## 10. Training-loop design choices

A few non-obvious decisions worth flagging:

### Weighted BCE for beats (`src/rhythm/train.py`)

Beats are ~5% of frames. Unweighted BCE makes the model predict "no
beat" everywhere, achieving 95% accuracy and 0% F-measure. We weight
positive frames 10× to push the model to actually fire on beats. The
value 10 is roughly `(1-prior)/prior` for a 5% prior; we didn't tune it.

### Gaussian-smeared beat labels (`src/rhythm/dataset.py`)

Hard 0/1 beat labels have a gradient signal of zero almost everywhere
and a discontinuity at the beat. We replace each beat label with a
narrow Gaussian (`σ ≈ 1 frame`), so the model gets a smooth target.
The post-processing peak-picker recovers discrete beat times at
inference.

### Masked BCE for pitch (`src/pitch/train.py`)

Unvoiced frames have no ground-truth pitch, so we mask them out of the
loss rather than asking the model to predict "all zeros." Forcing the
model to learn unvoiced detection from negative supervision (lack of
signal) tends to work better than mixing voiced/unvoiced into a single
target distribution.

### Soft pitch labels (`src/pitch/dataset.py`)

Gamakas slide continuously across multiple 20-cent bins. A hard one-hot
label punishes the model for any disagreement with the nearest bin,
which is wrong for gamakas. We use a Gaussian with σ=1 bin (20 cents)
around the true pitch, then renormalise. This is exactly what the
original CREPE paper does.

### Early stopping by validation metric, not loss

For both tasks we early-stop on the *task metric* (F-measure for beat,
RPA for pitch), not validation loss. The losses are surrogates; the
metrics are what we report. Stopping on the metric occasionally
disagrees with stopping on loss and tends to give a slightly better
final model.

### Why fixed-seed track splits

`make_splits()` in both `rhythm/dataset.py` and `pitch/dataset.py` uses
`np.random.default_rng(42)` to permute tracks. This means train/val/test
splits are deterministic across runs — the from-scratch model and the
fine-tuned model see identical Carnatic test tracks, so their numbers
are directly comparable.

---

## 11. Evaluation metrics and why

### Beat

We report what `mir_eval.beat.evaluate` reports:

- **F-measure**: precision/recall of detected beats with a ±70 ms
  tolerance window. The headline number.
- **Cemgil**: smoother continuous version of F-measure (Gaussian
  window). Less sensitive to small misalignments.
- **CMLt / CMLc**: continuity scores — penalises octave errors and
  tempo doublings/halvings. Useful for diagnosing whether the model is
  picking up a different metrical level rather than missing beats
  entirely. Common failure mode on Carnatic.
- **Downbeat F-measure**: F-measure restricted to sama (first beat of
  the tala). Usually much lower than the beat F-measure because sama
  detection is harder.

### Pitch

We report what `mir_eval.melody.evaluate` reports, plus one custom
metric:

- **Raw Pitch Accuracy (RPA)**: fraction of voiced frames where the
  estimated F0 is within ±50 cents of the ground truth.
- **Overall Accuracy (OA)**: like RPA but also penalises wrong
  voiced/unvoiced decisions.
- **MAE cents**: mean absolute pitch error in cents over voiced frames.
- **Semitone-snap ratio (custom, in `src/pitch/evaluation.py`)**: fraction
  of voiced frames whose estimated pitch is within ±25 cents of a 12-TET
  note. We compute this on the *estimate*, not the ground truth.
  CREPE was trained on Western data and snaps gamakas to the nearest
  semitone; a high snap ratio on Carnatic indicates this failure mode.
  We expect this to drop substantially after fine-tuning.

---

## 12. Caveats, limits, and things we didn't do

### Things we don't claim

- **We don't claim CREPELike is a better pitch model than CREPE.** It's
  the *same* model — just runnable in PyTorch so we can fine-tune it.
- **We don't claim our beat model (BeatActivationModel) is competitive
  with madmom on Western data.** Madmom has a sophisticated DBN
  post-processor; we use simple peak-picking. The pretrained Western
  checkpoint (condition C step 1) is a fair starting point for
  fine-tuning but is not state-of-the-art Western beat tracking.
- **We don't claim the Carnatic test sets are statistically large.**
  Saraga's ~200 tracks split 80/10/10 leaves ~20 test tracks. Effect
  sizes need to be interpreted accordingly.

### Things we deliberately did not do

- **No data augmentation.** Pitch shift, time stretch, additive noise
  would all be reasonable. We left them off so the comparison is clean.
- **No layer-freezing experiments.** A finer-grained transfer study
  would freeze the early conv layers and only fine-tune the last
  block + classifier. We do full fine-tuning with a lower LR instead.
  The `--init-checkpoint` mechanism doesn't currently support layer
  freezing.
- **No multi-task training.** It would be reasonable to train one
  network jointly on Western and Carnatic data, perhaps with a domain
  embedding. Out of scope.
- **No statistical significance tests.** Confidence intervals on the
  metric means are computed by `aggregate_results` (std across tracks)
  but we don't run paired tests between conditions. Should be added
  before report submission.

### Things you might want to add

- A "tiny" CREPE variant for faster fine-tuning. The official `crepe`
  package supports `tiny`, `small`, `medium`, `large`, `full`. Only
  `full` is currently supported by `convert_crepe_weights.py` (because
  filter counts in `CREPELike` are hard-coded to match `full`).
- A linear-probe baseline: freeze CREPE entirely, train only a small
  head on Carnatic data. Tells you how much of CREPE's existing
  representation is already useful for Carnatic without any adaptation.
- Per-raga / per-tala breakdowns. Saraga annotates each track with its
  raga and tala; performance probably varies a lot across them.

---

## Glossary (for collaborators new to MIR)

- **MIR**: Music Information Retrieval. Computational analysis of audio
  to extract musical content (pitch, beat, key, chord, genre, etc.).
- **F0 / pitch**: fundamental frequency of a monophonic voice.
- **Gamaka**: Carnatic ornamentation — pitch oscillations, slides,
  microtonal inflections around a target note. Not noise; an essential
  expressive feature.
- **Tala**: Carnatic rhythmic cycle. Can be 3, 4, 5, 7, 8, 9, or 11
  beats long with internal structure.
- **Sama**: the first beat of a tala cycle — analogous to the downbeat
  in Western music.
- **Shruti**: a microtonal unit in Carnatic theory (~22 per octave,
  unequally spaced).
- **Raga**: a melodic framework defining a scale, characteristic
  phrases, ornamentation rules, and emotional content.
- **F-measure**: harmonic mean of precision and recall.
- **RPA / OA**: Raw Pitch Accuracy / Overall Accuracy — see §11.
- **Cents**: logarithmic pitch unit, 100 cents per semitone.
- **DBN**: Dynamic Bayesian Network — the temporal post-processor
  madmom uses to convert frame-level beat probabilities into a
  metrically coherent beat sequence.
- **Transfer learning**: taking a model trained on one task/domain and
  adapting it to another, usually by fine-tuning with a smaller learning
  rate on the target data.
