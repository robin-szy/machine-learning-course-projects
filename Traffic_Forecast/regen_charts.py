"""Regenerate only the static PNG charts (no GIF)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.io import loadmat

OUT = "Intro_to_Machine_Learning___Slides_Final_Presentation/Images"
os.makedirs(OUT, exist_ok=True)

RED    = "#c0392b"; BLUE   = "#2980b9"; ORANGE = "#e67e22"
GREEN  = "#27ae60"; GRAY   = "#95a5a6"; DARK   = "#2c3e50"; BG = "#fafafa"

matplotlib.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

# ── 1. RMSE comparison ────────────────────────────────────────────────────────
print("rmse_comparison.png")
models = [
    ("Persistence\nbaseline",    0.0545,  0,       GRAY),
    ("CNN1D-\nAttentionGCN",     0.0409,  10530,   "#7f8c8d"),
    ("GCN-GRU",                  0.0384,  42209,   "#95a5a6"),
    ("GCN-MLP h=64",             0.0383,  26401,   "#95a5a6"),
    ("GRU-GCN h=96",             0.0376,  67937,   BLUE),
    ("GCN-MLP h=32",             0.0373,  11201,   "#95a5a6"),
    ("CNN1D-GCN\nh=32",          0.0371,  13025,   ORANGE),
    ("CNN1D-GCN\nh=160",         0.0370,  171233,  ORANGE),
    ("GRU-GCN h=64",             0.0369,  34401,   BLUE),
    ("Pure MLP\nh=64",           0.0364,  15233,   GREEN),
    ("Transformer",              0.0362,  67585,   "#8e44ad"),
    ("CNN1D-GCN\nh=64",          0.0362,  34145,   ORANGE),
    ("Pure MLP\nh=96",           0.0355,  28193,   GREEN),
]
names  = [m[0] for m in models]
rmse   = [m[1] for m in models]
params = [m[2] for m in models]
colors = [m[3] for m in models]

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.barh(names, rmse, color=colors, edgecolor="white", height=0.6)
ax.axvline(0.0545, color=GRAY, lw=1.2, ls="--", alpha=0.6)
for bar, r, p in zip(bars, rmse, params):
    lbl = f"{r:.4f}" if p == 0 else f"{r:.4f}  ({p:,})"
    ax.text(r + 0.0003, bar.get_y() + bar.get_height()/2, lbl, va="center", fontsize=9)
ax.set_xlabel("Test RMSE (normalised flow)")
ax.set_xlim(0, 0.075)
ax.invert_yaxis()
ax.set_title("Final model comparison - test set RMSE", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Overfitting analysis ───────────────────────────────────────────────────
print("overfitting_analysis.png")
results = [
    ("GRU-GCN h=96",       0.0322, 0.0338, 0.0359),
    ("CNN1D-GCN h=160",    0.0304, 0.0327, 0.0357),
    ("CNN1D-GCN h=64",     0.0317, 0.0332, 0.0361),
    ("GCN-MLP h=64",       0.0316, 0.0334, 0.0358),
    ("Pure MLP h=96",      0.0314, 0.0335, 0.0354),
    ("CNN1D-AttnGCN",      0.0323, 0.0330, 0.0370),
    ("Transformer",        0.0286, 0.0330, 0.0362),
]
names2   = [r[0] for r in results]
tr_rmse  = [r[1] for r in results]
val_rmse = [r[2] for r in results]
te_rmse  = [r[3] for r in results]
gaps     = [v - t for t, v in zip(tr_rmse, val_rmse)]

x = np.arange(len(names2)); w = 0.26
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)

ax1.set_facecolor(BG)
ax1.bar(x - w, tr_rmse,  w, label="Train", color=BLUE,   alpha=0.85)
ax1.bar(x,     val_rmse, w, label="Val",   color=ORANGE, alpha=0.85)
ax1.bar(x + w, te_rmse,  w, label="Test",  color=RED,    alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(names2, rotation=30, ha="right")
ax1.set_ylabel("RMSE"); ax1.legend()
ax1.set_title("Train / Val / Test RMSE", fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)

gap_colors = [RED if g > 0.01 else (ORANGE if g > 0.004 else BLUE) for g in gaps]
ax2.set_facecolor(BG)
brs = ax2.bar(names2, gaps, color=gap_colors, edgecolor="white")
ax2.axhline(0.010, color=RED,    lw=1.5, ls="--", label="Overfit threshold (0.010)")
ax2.axhline(0.004, color=ORANGE, lw=1.2, ls=":",  label="Watch zone (0.004)")
ax2.set_xticks(range(len(names2)))
ax2.set_xticklabels(names2, rotation=30, ha="right")
ax2.set_ylabel("Val - Train RMSE (gap)")
ax2.set_title("Overfitting gap per model", fontweight="bold")
ax2.legend(); ax2.spines[["top","right"]].set_visible(False)
for bar, g in zip(brs, gaps):
    ax2.text(bar.get_x()+bar.get_width()/2, g+0.0001,
             f"{g:.4f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/overfitting_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. MLP vs GCN box ────────────────────────────────────────────────────────
print("mlp_vs_gcn.png")
fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
boxes = [
    (0.08, 0.22, 0.32, 0.52, GREEN,  "Pure MLP h=96\n+ Embedding",
     "28,193 params\nTest RMSE: 0.0355\nGap: +0.0021"),
    (0.58, 0.22, 0.32, 0.52, ORANGE, "CNN1D-GCN\n(h = 160)",
     "171,233 params\nTest RMSE: 0.0370\nGap: +0.0023"),
]
for (x, y, w, h, col, title, body) in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=col, alpha=0.15, edgecolor=col, lw=2,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.08, title, ha="center", va="top",
            fontsize=13, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(x+w/2, y+h/2-0.04, body, ha="center", va="center",
            fontsize=11, color=DARK, transform=ax.transAxes, multialignment="center")

ax.text(0.5, 0.88, "Similar RMSE - 6x fewer parameters",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=DARK, transform=ax.transAxes)
ax.annotate("", xy=(0.57, 0.50), xytext=(0.43, 0.50), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=2))
ax.text(0.5, 0.42, "MLP wins", ha="center", fontsize=12, color=DARK,
        transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{OUT}/mlp_vs_gcn.png", dpi=150, bbox_inches="tight")
plt.close()

print("Done.")
