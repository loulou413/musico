"""Convert official Keras CREPE pretrained weights to a PyTorch checkpoint
compatible with src.pitch.model.CREPELike.

This produces a .pt state_dict that can be fed to train_pitch_carnatic.py
via --init-checkpoint so the Carnatic model starts from CREPE's
Western-pretrained weights.

Usage:
    PYTHONPATH=. python scripts/convert_crepe_weights.py \
        --model-capacity full \
        --output results/pitch/crepe_pretrained.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.pitch.model import CREPELike


def convert(model_capacity: str = "full") -> dict:
    """Build the Keras CREPE model, copy its weights into a PyTorch CREPELike.

    Returns the resulting PyTorch state_dict.
    """
    from crepe.core import build_and_load_model

    keras_model = build_and_load_model(model_capacity)
    keras_layers = {layer.name: layer for layer in keras_model.layers}

    if model_capacity != "full":
        raise ValueError(
            f"Only 'full' is currently supported (CREPELike defaults match CREPE 'full'). "
            f"Got: {model_capacity}"
        )

    pt_model = CREPELike(n_filters_scale=1.0)
    state = pt_model.state_dict()

    for i in range(1, 7):  # conv1 .. conv6
        block_idx = i - 1
        conv_name = f"conv{i}"
        bn_name = f"conv{i}-BN"

        # ── Conv weights ──────────────────────────────────────────────────
        kW, kB = keras_layers[conv_name].get_weights()
        # Keras kW shape: (H, W, in_ch, out_ch) → PyTorch (out_ch, in_ch, H, W)
        pt_kW = np.transpose(kW, (3, 2, 0, 1))
        state[f"blocks.{block_idx}.conv.weight"] = torch.from_numpy(pt_kW).float()
        state[f"blocks.{block_idx}.conv.bias"] = torch.from_numpy(kB).float()

        # ── BatchNorm weights ─────────────────────────────────────────────
        bn_w = keras_layers[bn_name].get_weights()
        # Keras returns [gamma, beta, moving_mean, moving_var]
        gamma, beta, moving_mean, moving_var = bn_w
        state[f"blocks.{block_idx}.bn.weight"] = torch.from_numpy(gamma).float()
        state[f"blocks.{block_idx}.bn.bias"] = torch.from_numpy(beta).float()
        state[f"blocks.{block_idx}.bn.running_mean"] = torch.from_numpy(moving_mean).float()
        state[f"blocks.{block_idx}.bn.running_var"] = torch.from_numpy(moving_var).float()
        state[f"blocks.{block_idx}.bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    # ── Final classifier (Dense → Linear) ─────────────────────────────────
    cls_W, cls_B = keras_layers["classifier"].get_weights()
    # Keras Dense W: (in, out) → PyTorch Linear: (out, in)
    state["classifier.weight"] = torch.from_numpy(cls_W.T).float()
    state["classifier.bias"] = torch.from_numpy(cls_B).float()

    return state, keras_model, pt_model


def verify(state: dict, keras_model, pt_model, atol: float = 1e-4) -> None:
    """Load the converted weights and compare outputs on a random input."""
    pt_model.load_state_dict(state)
    pt_model.eval()

    rng = np.random.default_rng(0)
    test_input = rng.standard_normal((4, 1024)).astype(np.float32)

    # Keras input shape: (B, 1024)
    keras_out = keras_model.predict(test_input, verbose=0)

    # PyTorch input shape: (B, 1, 1024)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(test_input).unsqueeze(1)).numpy()

    max_diff = float(np.max(np.abs(keras_out - pt_out)))
    mean_diff = float(np.mean(np.abs(keras_out - pt_out)))
    print(f"\n=== Verification ===")
    print(f"max  |keras - torch|: {max_diff:.6e}")
    print(f"mean |keras - torch|: {mean_diff:.6e}")
    if max_diff > atol:
        raise RuntimeError(
            f"Outputs differ by more than tolerance ({atol}). "
            f"Conversion is broken. Inspect padding / permutation."
        )
    print(f"OK — outputs match within {atol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-capacity", default="full",
                        choices=["full"],
                        help="CREPE capacity (only 'full' supported)")
    parser.add_argument("--output", default="results/pitch/crepe_pretrained.pt",
                        help="Output path for the PyTorch state_dict")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the Keras-vs-PyTorch output comparison")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Max allowable absolute difference between Keras and PyTorch")
    args = parser.parse_args()

    print(f"Loading Keras CREPE '{args.model_capacity}' model…")
    state, keras_model, pt_model = convert(args.model_capacity)
    print(f"Converted {len(state)} tensors.")

    if not args.no_verify:
        verify(state, keras_model, pt_model, atol=args.atol)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"\nSaved PyTorch checkpoint: {out}")
    print("Pass it to train_pitch_carnatic.py via --init-checkpoint to fine-tune on Saraga.")


if __name__ == "__main__":
    main()
