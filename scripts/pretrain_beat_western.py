"""Pretrain the BeatActivationModel on GTZAN Western beat annotations.

This produces a checkpoint that can be passed to `train_beat_carnatic.py`
via `--init-checkpoint`, so the Carnatic model starts from Western-learned
weights instead of random init.

Usage:
    PYTHONPATH=. python scripts/pretrain_beat_western.py \
        --gtzan-home data/raw/gtzan \
        --output-dir results/rhythm/western_pretrained \
        --epochs 30 --max-tracks 200
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.preprocessing.gtzan import get_gtzan_split
from src.rhythm.dataset import SaragaBeatDataset, make_splits
from src.rhythm.model import BeatActivationModel
from src.rhythm.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtzan-home", default="data/raw/gtzan")
    parser.add_argument("--output-dir", default="results/rhythm/western_pretrained")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Cap number of GTZAN tracks (useful for quick tests)")
    parser.add_argument("--genres", nargs="+", default=None,
                        help="Restrict to specific GTZAN genres (default: all)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading GTZAN tracks…")
    all_tracks = get_gtzan_split(
        args.gtzan_home,
        genres=args.genres,
        max_tracks=args.max_tracks,
    )
    all_tracks = [t for t in all_tracks if t.beat_times is not None and len(t.beat_times) > 0]
    print(f"  Loaded {len(all_tracks)} tracks with beat annotations")
    if len(all_tracks) == 0:
        raise RuntimeError(
            "No GTZAN tracks with beat annotations were found. "
            "Re-run scripts/download_data.py --dataset gtzan to fetch them."
        )

    train_ds, val_ds, test_ds = make_splits(
        all_tracks,
        train_ratio=0.8,
        val_ratio=0.1,
        seq_len=args.seq_len,
        n_mels=args.n_mels,
    )
    print(f"  Segments — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    with open(out / "split_info.json", "w") as f:
        json.dump({
            "total_tracks": len(all_tracks),
            "train_segments": len(train_ds),
            "val_segments": len(val_ds),
            "test_segments": len(test_ds),
            "genres": args.genres,
        }, f, indent=2)

    model = BeatActivationModel(
        n_mels=args.n_mels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_f_measure"])
    axes[1].set_title("Val beat F-measure (Western)")
    fig.savefig(out / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWestern-pretrained checkpoint: {out / 'best_model.pt'}")
    print("Pass it to train_beat_carnatic.py via --init-checkpoint to fine-tune on Saraga.")


if __name__ == "__main__":
    main()
