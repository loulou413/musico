"""Visualize pitch estimation results across conditions and domains.

Can read either a CSV (comparison_all.csv) or the raw terminal log produced
by eval_pitch_carnatic.py / train_pitch_carnatic.py --compare-all.

Usage — from CSV:
    PYTHONPATH=. python scripts/visualize_pitch_results.py \
        --csv results/pitch/B_carnatic_from_scratch/comparison_all.csv \
        --out results/pitch/figures

Usage — from saved log file:
    PYTHONPATH=. python scripts/visualize_pitch_results.py \
        --log eval_output.txt \
        --out results/pitch/figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Palette ────────────────────────────────────────────────────────────────
METHOD_META = {
    "carnatic": dict(label="B — Carnatic CNN", color="#DD8452"),
    "crepe":    dict(label="A — CREPE (Western)", color="#4C72B0"),
    "pyin":     dict(label="Baseline — pYIN", color="#8C8C8C"),
    "c_model":  dict(label="C — Fine-tuned CREPE", color="#55A868"),
}
DOMAIN_LABELS = {"carnatic": "Carnatic (Saraga)", "western": "Western (GuitarSet)"}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["raw_pitch_accuracy"])
    df["domain"] = df["domain"].str.lower().str.strip()
    return df


def parse_log(log_path: str) -> pd.DataFrame:
    """Parse the terminal output from eval_pitch_carnatic.py.

    Handles lines like:
        --- carnatic ---
          [carnatic] 14_Angakaram  RPA=0.813
          [western]  00_BN1-129-Eb_solo_E  RPA=0.657
    """
    method_re = re.compile(r"^---\s+(\w+)\s+---")
    track_re  = re.compile(r"\[(carnatic|western)\]\s+(\S+)\s+RPA=([\d.]+|nan)")

    rows = []
    current_method = None
    with open(log_path) as f:
        for line in f:
            m = method_re.search(line)
            if m:
                current_method = m.group(1)
                continue
            t = track_re.search(line)
            if t and current_method:
                domain, track_id, rpa_str = t.groups()
                rpa = float(rpa_str)
                if not np.isnan(rpa):
                    rows.append({
                        "method": current_method,
                        "domain": domain,
                        "track_id": track_id,
                        "raw_pitch_accuracy": rpa,
                    })
    df = pd.DataFrame(rows)
    return df


def grouped_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    domains: list[str],
    ylabel: str,
    title: str,
    ymax: float | None = None,
    lower_is_better: bool = False,
) -> None:
    n_methods = len(methods)
    n_domains = len(domains)
    group_w = 0.7
    bar_w = group_w / n_methods
    x = np.arange(n_domains)

    for i, method in enumerate(methods):
        meta = METHOD_META.get(method, {"label": method, "color": "gray"})
        means, stds = [], []
        for domain in domains:
            sub = df[(df["method"] == method) & (df["domain"] == domain)][metric]
            means.append(sub.mean() if len(sub) else np.nan)
            stds.append(sub.std() if len(sub) > 1 else 0.0)

        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, means, bar_w * 0.9,
            yerr=stds, capsize=4,
            color=meta["color"], label=meta["label"],
            error_kw={"elinewidth": 1.2, "ecolor": "0.3"},
        )
        # Value labels on bars
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                va = "bottom" if not lower_is_better else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (bar.get_yerr() if hasattr(bar, "get_yerr") else 0) + 0.01,
                    f"{m:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS.get(d, d) for d in domains], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def make_legend(methods: list[str]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(
            color=METHOD_META.get(m, {"color": "gray"})["color"],
            label=METHOD_META.get(m, {"label": m})["label"],
        )
        for m in methods
    ]


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ["carnatic", "crepe", "pyin", "c_model"] if m in df["method"].unique()]
    domains = [d for d in ["carnatic", "western"] if d in df["domain"].unique()]

    # ── Figure 1: RPA + Overall Accuracy (2 panels) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    fig.suptitle("Pitch Estimation: RPA and Overall Accuracy by Condition × Domain",
                 fontsize=12, fontweight="bold", y=1.01)

    grouped_bar(axes[0], df, "raw_pitch_accuracy", methods, domains,
                ylabel="Raw Pitch Accuracy (RPA)", title="Raw Pitch Accuracy", ymax=1.05)
    grouped_bar(axes[1], df, "overall_accuracy", methods, domains,
                ylabel="Overall Accuracy", title="Overall Accuracy", ymax=1.05)

    legend_patches = make_legend(methods)
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pitch_rpa_oa.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pitch_rpa_oa.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved pitch_rpa_oa.pdf/png")

    # ── Figure 2: MAE + Semitone-snap (2 panels) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Pitch Estimation: Error and Western Bias by Condition × Domain",
                 fontsize=12, fontweight="bold", y=1.01)

    grouped_bar(axes[0], df, "mean_abs_error_cents", methods, domains,
                ylabel="Mean Absolute Error (cents)", title="Pitch Error (cents)",
                lower_is_better=True)
    grouped_bar(axes[1], df, "semitone_snap_ratio", methods, domains,
                ylabel="Semitone-Snap Ratio", title="Western Equal-Temperament Bias",
                ymax=1.05)

    fig.legend(handles=legend_patches, loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pitch_mae_snap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pitch_mae_snap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved pitch_mae_snap.pdf/png")

    # ── Figure 3: 4-metric overview heatmap ────────────────────────────────
    metrics = ["raw_pitch_accuracy", "overall_accuracy", "mean_abs_error_cents", "semitone_snap_ratio"]
    metric_labels = ["RPA", "Overall Acc.", "MAE (cents)", "Semitone-snap"]

    rows, row_labels = [], []
    for method in methods:
        for domain in domains:
            sub = df[(df["method"] == method) & (df["domain"] == domain)]
            if sub.empty:
                continue
            rows.append([sub[m].mean() for m in metrics])
            rows[-1][2] = sub["mean_abs_error_cents"].mean()  # keep cents as-is
            row_labels.append(f"{METHOD_META.get(method, {'label': method})['label']}\n({DOMAIN_LABELS.get(domain, domain)})")

    if rows:
        data = np.array(rows, dtype=float)
        # Normalise columns to [0,1] for colour (flip MAE)
        norm = np.zeros_like(data)
        for j in range(data.shape[1]):
            col = data[:, j]
            rng = col.max() - col.min()
            if rng == 0:
                norm[:, j] = 0.5
            else:
                norm[:, j] = (col - col.min()) / rng
        norm[:, 2] = 1 - norm[:, 2]  # lower MAE = better = darker

        fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.5))
        im = ax.imshow(norm, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title("Performance Heatmap (green = better)", fontsize=11, fontweight="bold")

        for i in range(len(rows)):
            for j in range(len(metrics)):
                v = data[i, j]
                txt = f"{v:.2f}" if j != 2 else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color="black" if norm[i, j] < 0.75 else "0.15")

        fig.tight_layout()
        fig.savefig(out_dir / "pitch_heatmap.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "pitch_heatmap.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved pitch_heatmap.pdf/png")

    # ── Print summary table ────────────────────────────────────────────────
    print("\n=== RPA by method × domain ===")
    summary = (
        df.groupby(["method", "domain"])["raw_pitch_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(summary.to_string())
    print()
    print("=== MAE (cents) by method × domain ===")
    mae = (
        df.groupby(["method", "domain"])["mean_abs_error_cents"]
        .agg(["mean", "std"])
        .round(1)
    )
    print(mae.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results/pitch/B_carnatic_from_scratch/comparison_all.csv",
        help="Path to comparison_all.csv (can be partial)",
    )
    parser.add_argument(
        "--out",
        default="results/pitch/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--also-load",
        nargs="*",
        default=[],
        help="Additional CSVs to merge (e.g. condition C results)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run eval_pitch_carnatic.py first (or wait for it to finish at least one method).")
        return

    df = load_data(str(csv_path))

    for extra in args.also_load:
        p = Path(extra)
        if p.exists():
            extra_df = load_data(str(p))
            df = pd.concat([df, extra_df], ignore_index=True)
            print(f"  Merged {extra}: {len(extra_df)} rows")
        else:
            print(f"  Warning: {extra} not found, skipping")

    print(f"Loaded {len(df)} rows with annotations from {csv_path}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Domains: {sorted(df['domain'].unique())}")

    out_dir = Path(args.out)
    plot_summary(df, out_dir)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
