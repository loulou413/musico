"""Train the beat activation model on Saraga (Carnatic) data.

The trained model can then be evaluated against the Western-trained madmom
baseline to measure cross-domain transfer in both directions.

Usage:
    python scripts/train_beat_carnatic.py \
        --saraga-home data/raw/saraga \
        --output-dir  results/rhythm/carnatic_model \
        --epochs      50 \
        --max-tracks  80

After training, evaluate with:
    python scripts/run_rhythm_experiments.py \
        --backend carnatic \
        --carnatic-checkpoint results/rhythm/carnatic_model/best_model.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.preprocessing.saraga import get_saraga_split
from src.rhythm.dataset import make_splits
from src.rhythm.model import BeatActivationModel
from src.rhythm.train import train, validate, load_model
from src.rhythm.dataset import SaragaBeatDataset
from src.rhythm.beat_tracking import run_beat_tracking
from src.rhythm.evaluation import evaluate_beats, aggregate_results
from src.rhythm.analysis import results_to_dataframe, plot_metric_by_domain

import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saraga-home", default="data/raw/saraga")
    parser.add_argument("--output-dir", default="results/rhythm/carnatic_model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Frames per training segment (~12 s at hop=512, sr=22050)")
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Cap number of Saraga tracks (useful for quick tests)")
    parser.add_argument("--compare-madmom", action="store_true",
                        help="Also run madmom baseline on test split and compare")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Path to a .pt checkpoint to initialise weights from "
                             "(fine-tuning). When set, --lr defaults to 1e-4 if "
                             "not explicitly overridden.")
    args = parser.parse_args()

    if args.init_checkpoint and args.lr == 1e-3:
        args.lr = 1e-4
        print(f"[fine-tune] Lowered default LR to {args.lr}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading Saraga tracks…")
    # Use all tracks; make_splits will create train/val/test from them
    all_tracks = get_saraga_split(
        args.saraga_home,
        split="test",       # mirdata may not define splits; we split ourselves
        max_tracks=args.max_tracks,
    )
    # Fall back to full dataset if mirdata returns nothing for "test"
    if len(all_tracks) == 0:
        import mirdata
        dataset = mirdata.initialize("saraga_carnatic", data_home=args.saraga_home)
        from src.preprocessing.saraga import load_saraga_track
        all_tracks = []
        for tid in dataset.track_ids[: args.max_tracks]:
            try:
                all_tracks.append(load_saraga_track(dataset.track(tid), args.saraga_home))
            except Exception as e:
                print(f"  Skip {tid}: {e}")

    print(f"  Loaded {len(all_tracks)} tracks")

    train_ds, val_ds, test_ds = make_splits(
        all_tracks,
        train_ratio=0.8,
        val_ratio=0.1,
        seq_len=args.seq_len,
        n_mels=args.n_mels,
    )
    print(f"  Segments — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # Save split sizes for reproducibility
    with open(out / "split_info.json", "w") as f:
        json.dump({
            "total_tracks": len(all_tracks),
            "train_segments": len(train_ds),
            "val_segments": len(val_ds),
            "test_segments": len(test_ds),
        }, f, indent=2)

    # ── Train ──────────────────────────────────────────────────────────────────
    model = BeatActivationModel(
        n_mels=args.n_mels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    if args.init_checkpoint:
        print(f"Loading initial weights from {args.init_checkpoint}")
        state = torch.load(args.init_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=str(out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        sr=22050,
        hop_length=512,
    )

    # ── Learning curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["val_f_measure"])
    axes[1].set_title("Validation beat F-measure")
    fig.savefig(out / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved to {out / 'learning_curves.png'}")

    # ── Test-set evaluation ────────────────────────────────────────────────────
    print("\nEvaluating on test tracks…")
    checkpoint = str(out / "best_model.pt")
    # Collect the actual AudioTrack objects used in test split
    # (they are stored inside the dataset)
    import numpy as np
    import mirdata
    from src.preprocessing.saraga import load_saraga_track

    # Reload test track objects (dataset only stores segments, not tracks)
    dataset_mir = mirdata.initialize("saraga_carnatic", data_home=args.saraga_home)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_tracks)).tolist()
    n_train = int(len(all_tracks) * 0.8)
    n_val   = int(len(all_tracks) * 0.1)
    test_tracks = [all_tracks[i] for i in indices[n_train + n_val:]]

    carnatic_results = []
    for track in test_tracks:
        pred = run_beat_tracking(track, backend="carnatic", carnatic_checkpoint=checkpoint)
        carnatic_results.append(evaluate_beats(track, pred))

    test_summary = aggregate_results(carnatic_results)
    with open(out / "test_summary.json", "w") as f:
        json.dump(test_summary, f, indent=2)
    print(f"Carnatic model — test F-measure: {test_summary['f_measure']['mean']:.4f}")

    # ── Madmom comparison ─────────────────────────────────────────────────────
    if args.compare_madmom:
        print("\nRunning madmom baseline on same test tracks…")
        madmom_results = []
        for track in test_tracks:
            pred = run_beat_tracking(track, backend="madmom")
            madmom_results.append(evaluate_beats(track, pred))
        madmom_summary = aggregate_results(madmom_results)

        print("\n=== Comparison on Carnatic test set ===")
        print(f"{'Metric':<25} {'Carnatic model':>15} {'Madmom (Western)':>18}")
        print("-" * 60)
        for metric in ["f_measure", "cemgil", "continuity", "downbeat_f_measure"]:
            cm = test_summary[metric]["mean"]
            mm = madmom_summary[metric]["mean"]
            print(f"{metric:<25} {cm:>15.4f} {mm:>18.4f}")

        with open(out / "madmom_comparison.json", "w") as f:
            json.dump({"carnatic_model": test_summary, "madmom": madmom_summary}, f, indent=2)

    print(f"\nAll outputs saved to {out}")


if __name__ == "__main__":
    main()
