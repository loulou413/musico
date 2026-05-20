"""Training and validation loop for the Saraga pitch model."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mir_eval
from tqdm import tqdm

from .model import CREPELike, activation_to_hz
from .dataset import SaragaPitchDataset, PITCH_BINS_HZ, TARGET_SR, HOP_SAMPLES, FRAME_LEN


def pitch_loss(
    pred: torch.Tensor,          # (B, 360) sigmoid outputs
    label: torch.Tensor,         # (B, 360) soft targets
    voiced: torch.Tensor,        # (B,) float 0/1
) -> torch.Tensor:
    """BCE over voiced frames only.

    Unvoiced frames contribute nothing to the pitch loss; the model learns
    to output low activations for them naturally via the lack of signal.
    """
    bce = nn.functional.binary_cross_entropy(pred, label, reduction="none")
    mask = voiced.unsqueeze(1)
    return (bce * mask).sum() / (mask.sum() * 360 + 1e-8)


@torch.no_grad()
def validate(
    model: CREPELike,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_ref_hz, all_est_hz, all_ref_voiced, all_est_voiced = [], [], [], []

    for frames, labels, voiced in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        voiced = voiced.to(device)

        pred = model(frames)
        total_loss += pitch_loss(pred, labels, voiced).item()
        n_batches += 1

        pred_np = pred.cpu().numpy()
        for i in range(frames.shape[0]):
            est_hz = activation_to_hz(pred_np[i])
            ref_hz = float((labels[i].cpu().numpy() * PITCH_BINS_HZ).sum())
            is_voiced = voiced[i].item() > 0.5

            all_ref_voiced.append(is_voiced)
            all_est_voiced.append(est_hz > 0)
            all_ref_hz.append(ref_hz if is_voiced else 0.0)
            all_est_hz.append(est_hz)

    ref_arr = np.array(all_ref_hz)
    est_arr = np.array(all_est_hz)
    times = np.arange(len(ref_arr)) * (HOP_SAMPLES / TARGET_SR)
    try:
        scores = mir_eval.melody.evaluate(times, ref_arr, times, est_arr)
        rpa = scores["Raw Pitch Accuracy"]
        oa  = scores["Overall Accuracy"]
    except Exception:
        rpa, oa = 0.0, 0.0

    return {
        "loss": total_loss / max(n_batches, 1),
        "raw_pitch_accuracy": rpa,
        "overall_accuracy": oa,
    }


def train(
    model: CREPELike,
    train_dataset: SaragaPitchDataset,
    val_dataset: SaragaPitchDataset,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 6,
    num_workers: int = 0,
) -> dict:
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
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": [], "val_rpa": [], "val_oa": []}
    best_rpa = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", unit="batch", leave=False)
        for frames, labels, voiced in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            voiced = voiced.to(device)

            optimizer.zero_grad()
            pred = model(frames)
            loss = pitch_loss(pred, labels, voiced)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        val_metrics = validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_rpa"].append(val_metrics["raw_pitch_accuracy"])
        history["val_oa"].append(val_metrics["overall_accuracy"])

        scheduler.step(val_metrics["raw_pitch_accuracy"])

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_RPA={val_metrics['raw_pitch_accuracy']:.4f}  "
            f"val_OA={val_metrics['overall_accuracy']:.4f}"
        )

        if val_metrics["raw_pitch_accuracy"] > best_rpa:
            best_rpa = val_metrics["raw_pitch_accuracy"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RPA={best_rpa:.4f})")
                break

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val RPA: {best_rpa:.4f}")
    print(f"Checkpoint saved to {out / 'best_model.pt'}")
    return history


def load_model(checkpoint_path: str, **model_kwargs) -> CREPELike:
    model = CREPELike(**model_kwargs)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
