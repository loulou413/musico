"""Plot per-track RPA for all annotated Carnatic (Saraga) tracks.

Run:  PYTHONPATH=. python scripts/plot_carnatic_tracks.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Data from eval run ────────────────────────────────────────────────────
# B = Carnatic CNN trained from scratch (condition B)
# A = CREPE pretrained on Western (condition A)  — partial, only 9 tracks

B = {
    "13_Thillana_Purnachandrika":         0.522,
    "14_Angakaram":                        0.813,
    "15_Samajavarada":                     0.608,
    "16_Evarura":                          0.636,
    "17_Eranapai":                         0.675,
    "18_Parama_Purusham":                  0.807,
    "19_Marubari":                         0.775,
    "20_Aparadhamula":                     0.698,
    "21_Sadabalarupapi_Shlokam":           0.940,
    "112_Rama_Rama_Guna_Seema":            0.059,
    "113_Prathi_Vaaram_Vaaram":            0.532,
    "114_Varashiki_Vahana":                0.089,
    "115_Idhu_Thaano_Thillai_Sthalam":    0.677,
    "116_Bhuvini_Dasudane":                0.037,
    "117_Karuna_Nidhi_Illalo":             0.052,
    "118_Tulasi_Bilva":                    0.118,
    "119_Shlokam_Shivah_Shaktyayukto":    0.524,
    "120_Velum_Mayilume":                  0.069,
    "178_Muruga_Muruga_Muruga":            0.697,
    "179_Chenduril_Nindradum_Kanda":       0.756,
    "180_Ninne_Bhajana":                   0.679,
    "181_Nera_Nammiti":                    0.651,
    "182_Dinamani_Vamsa":                  0.653,
    "183_Chakkani_Raja":                   0.455,
    "184_Ardhanareeshwaram":               0.629,
    "185_Pavamana_Suthudu_Pattu":          0.442,
    "186_Shloka":                          0.699,
    "187_Aaniraimekkani":                  0.776,
    "188_Vanajaksha_Ninne_Kori":           0.585,
    "189_Karuninchutakidi":                0.320,
    "190_Va_Va_Brindavana":                0.460,
    "191_Mati_Matiki":                     0.397,
    "192_Sri_Guru_Na_Palitosmi":           0.557,
    "193_Thoomani_Madatthu":               0.621,
    "194_Ramabhi_Rama_Manasu":             0.397,
    "195_Mangalam_Kosalendraya":           0.754,
    "196_Sri_Vidhya_Rajagopala":           0.556,
    "222_Madhava_Mamava":                  0.575,
    "223_Ramachandraya_Mangalam":          0.519,
    "224_Sapasyat_Kausalya":               0.685,
    "225_Seetapati_Namanasuna":            0.591,
    "226_Karunimpa_Idi":                   0.578,
    "227_Endaro_Mahanubhavulu":            0.618,
    "228_Bhogeendra_Shayinam":             0.683,
    "229_Thillana_Senchurutti":            0.479,
    "230_Chintayama_Kanda":                0.489,
    "231_Emani_Migula":                    0.501,
    "232_Saraguna_Palimpa":                0.474,
    "233_Nadatanum_Anisham":               0.453,
    "234_Seethamma":                       0.389,
    "235_Janakipathe_Jaya_Karunya_Jaladhe":0.363,
    "236_Eppadi_Padinaro":                 0.289,
    "237_Manasaramathi":                   0.423,
    "238_Tillana":                         0.366,
    "239_Bhavamulona":                     0.410,
    "240_Ardhanarishwaram":                0.575,
}

# CREPE available only for tracks 13-21 (eval was stopped at track 45)
A_partial = {
    "13_Thillana_Purnachandrika":  0.034,
    "14_Angakaram":                0.125,
    "15_Samajavarada":             0.110,
    "16_Evarura":                  0.087,
    "17_Eranapai":                 0.123,
    "18_Parama_Purusham":          0.171,
    "19_Marubari":                 0.195,
    "20_Aparadhamula":             0.117,
    "21_Sadabalarupapi_Shlokam":   0.570,
}

# ── Sort B tracks by RPA (descending) ────────────────────────────────────
tracks_sorted = sorted(B.items(), key=lambda x: x[1], reverse=True)
labels = [t.split("_", 1)[1].replace("_", " ") for t, _ in tracks_sorted]
b_vals = [v for _, v in tracks_sorted]
track_ids = [t for t, _ in tracks_sorted]
a_vals = [A_partial.get(t, np.nan) for t in track_ids]

n = len(labels)
y = np.arange(n)

# ── Figure 1: all B tracks, horizontal bar ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, n * 0.28 + 1.5))

# Color bars by RPA zone
colors = ["#2ecc71" if v >= 0.6 else "#f39c12" if v >= 0.3 else "#e74c3c"
          for v in b_vals]
ax.barh(y, b_vals, color=colors, alpha=0.85, height=0.7,
        label="B — Carnatic CNN")

# Overlay A where available
for i, (tid, av) in enumerate(zip(track_ids, a_vals)):
    if not np.isnan(av):
        ax.barh(y[i], av, color="#4C72B0", alpha=0.6, height=0.7,
                label="A — CREPE (Western)" if i == 0 else "")

# Mean line
mean_b = np.mean(b_vals)
ax.axvline(mean_b, color="black", linewidth=1.2, linestyle="--",
           label=f"B mean = {mean_b:.2f}")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("Raw Pitch Accuracy (RPA)", fontsize=11)
ax.set_title(
    "Per-track RPA on Carnatic (Saraga) — Condition B vs. A\n"
    f"n={n} annotated tracks  |  green ≥0.6  orange 0.3–0.6  red <0.3",
    fontsize=11, fontweight="bold",
)
ax.set_xlim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3, linestyle="--")
# Add legend (deduplicated)
handles, lbls = ax.get_legend_handles_labels()
seen, unique_h, unique_l = set(), [], []
for h, l in zip(handles, lbls):
    if l not in seen:
        seen.add(l); unique_h.append(h); unique_l.append(l)
ax.legend(unique_h, unique_l, loc="lower right", fontsize=9)

fig.tight_layout()
out = Path("results/pitch/figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "carnatic_per_track.pdf", bbox_inches="tight")
fig.savefig(out / "carnatic_per_track.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved carnatic_per_track.pdf/png  (n={n}, mean B RPA={mean_b:.3f})")

# ── Figure 2: side-by-side for the 9 tracks where both A and B exist ─────
both = [(t, B[t], A_partial[t]) for t in A_partial]
both_labels = [t.split("_", 1)[1].replace("_", " ") for t, *_ in both]
b2 = [b for _, b, _ in both]
a2 = [a for _, _, a in both]

fig2, ax2 = plt.subplots(figsize=(8, 4))
x = np.arange(len(both_labels))
w = 0.38
ax2.bar(x - w/2, b2, w, color="#DD8452", label="B — Carnatic CNN")
ax2.bar(x + w/2, a2, w, color="#4C72B0", label="A — CREPE (Western)")
ax2.set_xticks(x)
ax2.set_xticklabels(both_labels, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Raw Pitch Accuracy (RPA)", fontsize=10)
ax2.set_title("Head-to-head: B vs. A on same 9 Carnatic tracks", fontsize=11,
              fontweight="bold")
ax2.set_ylim(0, 1.05)
ax2.axhline(np.mean(b2), color="#DD8452", linestyle="--", linewidth=1,
            label=f"B mean = {np.mean(b2):.2f}")
ax2.axhline(np.mean(a2), color="#4C72B0", linestyle="--", linewidth=1,
            label=f"A mean = {np.mean(a2):.2f}")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(out / "carnatic_head2head.pdf", bbox_inches="tight")
fig2.savefig(out / "carnatic_head2head.png", bbox_inches="tight", dpi=150)
plt.close(fig2)
print(f"Saved carnatic_head2head.pdf/png  (9-track subset, B mean={np.mean(b2):.2f}, A mean={np.mean(a2):.2f})")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\nB (all {n} tracks):  mean={mean_b:.3f}  std={np.std(b_vals):.3f}")
print(f"B (9-track subset): mean={np.mean(b2):.3f}  std={np.std(b2):.3f}")
print(f"A (9-track subset): mean={np.mean(a2):.3f}  std={np.std(a2):.3f}")
