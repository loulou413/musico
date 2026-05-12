"""Training and evaluation loop for the Saraga beat activation model."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mir_eval

from .model import BeatActivationModel, activation_to_beat_times
from .dataset import SaragaBeatDataset


# ── Loss ─────────────────────────────────────────────────────────────────────

def beat_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
    """Weighted BCE — beats are rare (~5 % of frames), so upweight positives."""
    weight = torch.where(target > 0.5, torch.tensor(pos_weight, device=pred.device), torch.ones(1, device=pred.device))
    return nn.functional.binary_cross_entropy(pred, target, weight=weight)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: BeatActivationModel,
    loader: DataLoader,
    device: torch.device,
    sr: int = 22050,
    hop_length: int = 512,
) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    f_measures = []

    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        pred = model(mel)
        total_loss += beat_loss(pred, labels).item()
        n_batches += 1

        # Compute beat F-measure per segment
        for i in range(mel.shape[0]):
            ref_times = _labels_to_times(labels[i].cpu().numpy(), hop_length, sr)
            est_times = activation_to_beat_times(pred[i], sr=sr, hop_length=hop_length)
            if len(ref_times) > 0 and len(est_times) > 0:
                scores = mir_eval.beat.evaluate(ref_times, est_times)
                f_measures.append(scores["F-measure"])

    return {
        "loss": total_loss / max(n_batches, 1),
        "f_measure": float(np.mean(f_measures)) if f_measures else 0.0,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: BeatActivationModel,
    train_dataset: SaragaBeatDataset,
    val_dataset: SaragaBeatDataset,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 8,
    pos_weight: float = 10.0,
    sr: int = 22050,
    hop_length: int = 512,
    num_workers: int = 0,
) -> dict:
    """Train `model` and save the best checkpoint to `output_dir`.

    Returns a dict with training history.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "val_f_measure": []}
    best_f = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(mel)
            loss = beat_loss(pred, labels, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_metrics = validate(model, val_loader, device, sr=sr, hop_length=hop_length)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f_measure"].append(val_metrics["f_measure"])

        scheduler.step(val_metrics["f_measure"])

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_F={val_metrics['f_measure']:.4f}"
        )

        if val_metrics["f_measure"] > best_f:
            best_f = val_metrics["f_measure"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val F={best_f:.4f})")
                break

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val F-measure: {best_f:.4f}")
    print(f"Checkpoint saved to {out / 'best_model.pt'}")
    return history


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, **model_kwargs) -> BeatActivationModel:
    model = BeatActivationModel(**model_kwargs)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ── helper ────────────────────────────────────────────────────────────────────

def _labels_to_times(labels: np.ndarray, hop_length: int, sr: int) -> np.ndarray:
    """Convert binary frame label array back to seconds (for mir_eval)."""
    frames = np.where(labels > 0.5)[0]
    return frames * hop_length / sr
