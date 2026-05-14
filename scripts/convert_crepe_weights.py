"""Convert torchcrepe pretrained weights to a CREPELike checkpoint.

torchcrepe ships CREPE's official weights already in PyTorch format, so we
just remap layer names from torchcrepe's convention to CREPELike's.

Usage:
    PYTHONPATH=. python scripts/convert_crepe_weights.py \
        --output results/pitch/crepe_pretrained.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.pitch.model import CREPELike


def convert() -> tuple[dict, CREPELike]:
    import torchcrepe
    assets = Path(torchcrepe.__file__).parent / "assets" / "full.pth"
    tc_state = torch.load(assets, map_location="cpu", weights_only=True)

    pt_model = CREPELike(n_filters_scale=1.0)
    state = pt_model.state_dict()

    for i in range(6):
        src = f"conv{i + 1}"
        dst = f"blocks.{i}"
        state[f"{dst}.conv.weight"] = tc_state[f"{src}.weight"]
        state[f"{dst}.conv.bias"]   = tc_state[f"{src}.bias"]
        state[f"{dst}.bn.weight"]             = tc_state[f"{src}_BN.weight"]
        state[f"{dst}.bn.bias"]               = tc_state[f"{src}_BN.bias"]
        state[f"{dst}.bn.running_mean"]       = tc_state[f"{src}_BN.running_mean"]
        state[f"{dst}.bn.running_var"]        = tc_state[f"{src}_BN.running_var"]
        state[f"{dst}.bn.num_batches_tracked"] = tc_state[f"{src}_BN.num_batches_tracked"]

    state["classifier.weight"] = tc_state["classifier.weight"]
    state["classifier.bias"]   = tc_state["classifier.bias"]

    return state, pt_model


def verify(state: dict, pt_model: CREPELike, atol: float = 1e-4) -> None:
    """Compare CREPELike output against torchcrepe's own forward pass."""
    import torchcrepe
    from torchcrepe.model import Crepe

    pt_model.load_state_dict(state)
    pt_model.eval()

    tc_model = Crepe("full")
    assets = Path(torchcrepe.__file__).parent / "assets" / "full.pth"
    tc_model.load_state_dict(torch.load(assets, map_location="cpu", weights_only=True))
    tc_model.eval()

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((4, 1024)).astype(np.float32)
    x = torch.from_numpy(raw)

    with torch.no_grad():
        # CREPELike: (B, 1, 1024)
        our_out = pt_model(x.unsqueeze(1)).numpy()
        # torchcrepe Crepe: (B, 1024, 1) internal, but forward takes (B, 1024)
        tc_out = tc_model(x).numpy()

    max_diff = float(np.max(np.abs(our_out - tc_out)))
    mean_diff = float(np.mean(np.abs(our_out - tc_out)))
    print(f"\n=== Verification ===")
    print(f"max  |ours - torchcrepe|: {max_diff:.6e}")
    print(f"mean |ours - torchcrepe|: {mean_diff:.6e}")
    if max_diff > atol:
        raise RuntimeError(
            f"Outputs differ by more than tolerance ({atol}). "
            f"Conversion is broken."
        )
    print(f"OK — outputs match within {atol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/pitch/crepe_pretrained.pt")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    print("Extracting CREPE 'full' weights from torchcrepe…")
    state, pt_model = convert()
    print(f"Mapped {len(state)} tensors.")

    if not args.no_verify:
        verify(state, pt_model, atol=args.atol)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"\nSaved: {out}")
    print("Pass to train_pitch_carnatic.py via --init-checkpoint to fine-tune on Saraga.")


if __name__ == "__main__":
    main()
