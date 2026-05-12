"""Train the CREPE-like pitch model on Saraga (Carnatic) pitch annotations.

After training, evaluates the Carnatic model against pre-trained CREPE and pyin
on the same test tracks — measuring all three on both Carnatic and Western data.

Usage:
    python scripts/train_pitch_carnatic.py \
        --saraga-home  data/raw/saraga \
        --maestro-home data/raw/maestro \
        --output-dir   results/pitch/carnatic_model \
        --epochs       30 \
        --max-tracks   60 \
        --compare-all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing.saraga import get_saraga_split
from src.preprocessing.maestro import get_maestro_split
from src.pitch.dataset import make_splits, SaragaPitchDataset
from src.pitch.model import CREPELike
from src.pitch.train import train, load_model
from src.pitch.estimation import run_pitch_estimation
from src.pitch.evaluation import evaluate_pitch, aggregate_results
from src.pitch.analysis import results_to_dataframe, plot_metric_by_domain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saraga-home", default="data/raw/saraga")
    parser.add_argument("--maestro-home", default="data/raw/maestro")
    parser.add_argument("--output-dir", default="results/pitch/carnatic_model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--n-filters-scale", type=float, default=1.0,
                        help="Scale factor for CREPE conv layer widths. 1.0 = full "
                             "CREPE (~22M params, required for loading official CREPE "
                             "weights via --init-checkpoint). 0.5 = smaller, faster.")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--compare-all", action="store_true",
                        help="After training, compare Carnatic model vs CREPE vs pyin")
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

    # ── Load Saraga ────────────────────────────────────────────────────────────
    print("Loading Saraga tracks…")
    all_tracks = get_saraga_split(
        args.saraga_home, split="test", max_tracks=args.max_tracks
    )
    if len(all_tracks) == 0:
        import mirdata
        from src.preprocessing.saraga import load_saraga_track
        ds = mirdata.initialize("saraga_carnatic", data_home=args.saraga_home)
        all_tracks = []
        for tid in ds.track_ids[: args.max_tracks]:
            try:
                all_tracks.append(load_saraga_track(ds.track(tid), args.saraga_home))
            except Exception as e:
                print(f"  Skip {tid}: {e}")

    print(f"  Loaded {len(all_tracks)} tracks")

    train_ds, val_ds, test_ds = make_splits(
        all_tracks, train_ratio=0.8, val_ratio=0.1
    )
    print(f"  Frames — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    with open(out / "split_info.json", "w") as f:
        json.dump({
            "total_tracks": len(all_tracks),
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "test_frames": len(test_ds),
        }, f, indent=2)

    # ── Train ──────────────────────────────────────────────────────────────────
    model = CREPELike(n_filters_scale=args.n_filters_scale)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    if args.init_checkpoint:
        import torch
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
    )

    # ── Learning curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_rpa"])
    axes[1].set_title("Val Raw Pitch Accuracy")
    axes[2].plot(history["val_oa"])
    axes[2].set_title("Val Overall Accuracy")
    fig.savefig(out / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Recover test AudioTrack objects ───────────────────────────────────────
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_tracks)).tolist()
    n_train = int(len(all_tracks) * 0.8)
    n_val   = int(len(all_tracks) * 0.1)
    test_tracks_carnatic = [all_tracks[i] for i in indices[n_train + n_val:]]

    checkpoint = str(out / "best_model.pt")

    # ── Evaluate Carnatic model on Carnatic test set ───────────────────────────
    print("\nEvaluating Carnatic model on Carnatic test tracks…")
    carnatic_results = []
    for track in test_tracks_carnatic:
        pred = run_pitch_estimation(track, method="carnatic", carnatic_checkpoint=checkpoint)
        carnatic_results.append(evaluate_pitch(track, pred))

    carnatic_summary = aggregate_results(carnatic_results)
    with open(out / "test_carnatic_summary.json", "w") as f:
        json.dump(carnatic_summary, f, indent=2)

    if not args.compare_all:
        print(json.dumps(carnatic_summary, indent=2))
        print(f"\nAll outputs saved to {out}")
        return

    # ── Full comparison: 3 models × 2 domains ─────────────────────────────────
    print("\nLoading MAESTRO test tracks for Western evaluation…")
    western_tracks = get_maestro_split(
        args.maestro_home, split="test", max_tracks=args.max_tracks
    )

    methods = [
        ("carnatic", {"carnatic_checkpoint": checkpoint}),
        ("crepe",    {}),
        ("pyin",     {}),
    ]

    all_rows = []
    for method, extra_kwargs in methods:
        print(f"\n--- {method} ---")
        for track in test_tracks_carnatic + western_tracks:
            try:
                pred = run_pitch_estimation(track, method=method, **extra_kwargs)
                result = evaluate_pitch(track, pred)
                all_rows.append({
                    "method": method,
                    **result.__dict__,
                })
                print(f"  [{track.domain}] {track.track_id}  RPA={result.raw_pitch_accuracy:.3f}")
            except Exception as e:
                print(f"  Skip {track.track_id} ({method}): {e}")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "comparison_all.csv", index=False)

    # Summary table
    print("\n=== Comparison: RPA by method × domain ===")
    summary = df.groupby(["method", "domain"])["raw_pitch_accuracy"].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    summary.to_csv(out / "comparison_summary.csv")

    # Figures
    for metric in ["raw_pitch_accuracy", "overall_accuracy", "mean_abs_error_cents", "semitone_snap_ratio"]:
        plot_metric_by_domain(df, metric, str(out / "figures"))

    print(f"\nAll outputs saved to {out}")


if __name__ == "__main__":
    main()
