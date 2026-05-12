"""
Generate all images needed for the presentation slides.
Outputs to:  Intro_to_Machine_Learning___Slides_Final_Presentation/Images/
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from torch import nn
from scipy.io import loadmat

OUT = "Intro_to_Machine_Learning___Slides_Final_Presentation/Images"
os.makedirs(OUT, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
RED    = "#c0392b"
BLUE   = "#2980b9"
ORANGE = "#e67e22"
GREEN  = "#27ae60"
GRAY   = "#95a5a6"
DARK   = "#2c3e50"
LIGHT  = "#ecf0f1"
BG     = "#fafafa"

# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
data   = np.load("dataset.npz")
X_tr   = data["X_train"]
X_te   = data["X_test"]
Y_tr   = data["Y_train"]
Y_te   = data["Y_test"]
adj    = data["adj_mat"]
X_all  = np.concatenate([X_tr, X_te], axis=0)
Y_all  = np.concatenate([Y_tr, Y_te], axis=0)

weekday = np.argmax(X_all[:, 0, 10:15], axis=1)
hour    = np.argmax(X_all[:, 0, 15:39], axis=1)
n_sensors = 36

# Corrected sensor coordinates
SENSOR_LAT = {
     0:38.869, 14:38.868, 25:38.871,  4:38.873, 21:38.871,
     1:38.843,  2:38.816,  3:38.778,
    13:38.843, 10:38.818,  8:38.778,  9:38.758, 12:38.706, 11:38.641,
    22:38.880, 23:38.876, 24:38.869,
     5:38.883,  6:38.892,  7:38.874,
    20:38.881, 19:38.892, 18:38.876, 17:38.860, 16:38.838, 15:38.806,
    26:38.894, 29:38.897, 31:38.888,
    35:38.909, 28:38.921, 27:38.937,
    34:38.868, 33:38.848, 30:38.807, 32:38.808,
}
SENSOR_LON = {
     0:-77.051, 14:-77.056, 25:-77.063,  4:-77.069, 21:-77.074,
     1:-77.082,  2:-77.094,  3:-77.115,
    13:-77.079, 10:-77.097,  8:-77.117,  9:-77.167, 12:-77.224, 11:-77.268,
    22:-77.035, 23:-77.041, 24:-77.059,
     5:-77.107,  6:-77.226,  7:-77.296,
    20:-77.110, 19:-77.229, 18:-77.298, 17:-77.397, 16:-77.447, 15:-77.511,
    26:-77.232, 29:-77.226, 31:-77.235,
    35:-77.220, 28:-77.210, 27:-77.193,
    34:-77.219, 33:-77.210, 30:-77.199, 32:-77.201,
}
lat_arr = np.array([SENSOR_LAT[i] for i in range(n_sensors)])
lon_arr = np.array([SENSOR_LON[i] for i in range(n_sensors)])

mat     = loadmat("traffic_dataset.mat")
feats   = np.array([m.toarray() for m in mat['tra_X_tr'][0]], dtype=np.float32)[0]
road_id = np.argmax(feats[:, 44:48], axis=1)
ROAD_COLORS = [RED, BLUE, GREEN, ORANGE]


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Sensor network map
# ══════════════════════════════════════════════════════════════════════════════
print("1/7  sensor_network.png")
fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
ax.set_facecolor(BG)

# edges
for i in range(n_sensors):
    for j in range(i+1, n_sensors):
        if adj[i, j] > 0.05:
            ax.plot([lon_arr[i], lon_arr[j]], [lat_arr[i], lat_arr[j]],
                    color="#aaaaaa", lw=1.2, alpha=0.55, zorder=1)

mean_vol = Y_all.mean(axis=0)
sc = ax.scatter(lon_arr, lat_arr, c=mean_vol, cmap="YlOrRd",
                s=180, vmin=0, vmax=0.7, zorder=4,
                edgecolors=DARK, linewidths=1.2)
for i in range(n_sensors):
    ax.text(lon_arr[i], lat_arr[i], str(i),
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color="white", zorder=5)

cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Mean flow (norm.)", fontsize=8)
ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
ax.set_title("36 sensors · I-95 / I-66 corridor · Northern Virginia", fontsize=9)
legend_h = [mpatches.Patch(color=ROAD_COLORS[r],
            label=["I-95", "I-66", "I-495 NW", "I-495 SW"][r]) for r in range(4)]
ax.legend(handles=legend_h, fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/sensor_network.png", dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Feature breakdown visual
# ══════════════════════════════════════════════════════════════════════════════
print("2/7  feature_breakdown.png")
fig, ax = plt.subplots(figsize=(8, 2), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 48); ax.set_ylim(0, 1); ax.axis("off")

groups = [
    (0,  10, "#2980b9",  "Traffic history\n(×10 steps)"),
    (10,  5, "#27ae60",  "Weekday\n(one-hot ×5)"),
    (15, 24, "#e67e22",  "Hour of day\n(one-hot ×24)"),
    (39,  1, "#8e44ad",  "Lanes"),
    (40,  4, "#c0392b",  "Direction\n(×4)"),
    (44,  4, "#16a085",  "Road name\n(×4)"),
]
y0, h = 0.25, 0.5
for start, width, color, label in groups:
    rect = mpatches.FancyBboxPatch((start+0.15, y0), width-0.3, h,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="white", lw=1.2)
    ax.add_patch(rect)
    cx = start + width/2
    ax.text(cx, y0 + h/2, label,
            ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold", multialignment="center")

ax.text(24, -0.02, "Feature index  (0 → 47)", ha="center", va="top",
        fontsize=8, color=DARK)
ax.set_title("Input features per sensor per sample  [48 total]",
             fontsize=9, color=DARK, pad=6)
plt.tight_layout()
plt.savefig(f"{OUT}/feature_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3 — RMSE comparison (test set)
# ══════════════════════════════════════════════════════════════════════════════
print("3/7  rmse_comparison.png")
models = [
    ("Persistence\nbaseline",    0.0545, 0, GRAY),
    ("V1 GRU-GCN\n(fixed adj)",  0.0538, 26593,  "#95a5a6"),
    ("V2 GRU-GCN\n(learnable)",  0.0365, 13153,  BLUE),
    ("GCN→Conv1D\n(per step)",   0.0380, 15009,  "#7f8c8d"),
    ("CNN1D+Attn\nGCN",          0.0368, 10497,  "#7f8c8d"),
    ("Pure MLP\n+ embed",        0.0357, 9601,   GREEN),
    ("CNN1D-GCN\n(h=160)",       0.0359, 171233, ORANGE),
]
names  = [m[0] for m in models]
rmse   = [m[1] for m in models]
params = [m[2] for m in models]
colors = [m[3] for m in models]

fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.barh(names, rmse, color=colors, edgecolor="white", height=0.6)
ax.axvline(0.0545, color=GRAY, lw=1.2, ls="--", alpha=0.6)
for bar, r, p in zip(bars, rmse, params):
    lbl = f"{r:.4f}" if p == 0 else f"{r:.4f}  ({p:,} params)"
    ax.text(r + 0.0003, bar.get_y() + bar.get_height()/2,
            lbl, va="center", fontsize=7.5)
ax.set_xlabel("Test RMSE (normalised flow)", fontsize=9)
ax.set_xlim(0, 0.075)
ax.invert_yaxis()
ax.set_title("Model comparison — test set RMSE", fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Overfitting analysis (train vs val vs test)
# ══════════════════════════════════════════════════════════════════════════════
print("4/7  overfitting_analysis.png")
results = [
    ("V1 GRU-GCN",     0.0436, 0.0445, 0.0538),
    ("V2 GRU-GCN",     0.0314, 0.0327, 0.0365),
    ("Pure MLP+emb",   0.0308, 0.0329, 0.0357),
    ("CNN1D-GCN h160", 0.0268, 0.0318, 0.0359),
    ("GCN→Conv1D",     0.0325, 0.0338, 0.0380),
    ("CNN1D+AttnGCN",  0.0309, 0.0331, 0.0368),
]
names2   = [r[0] for r in results]
tr_rmse  = [r[1] for r in results]
val_rmse = [r[2] for r in results]
te_rmse  = [r[3] for r in results]
gaps     = [v - t for t, v in zip(tr_rmse, val_rmse)]

x = np.arange(len(names2)); w = 0.26
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)

# Left: grouped bars
ax1.set_facecolor(BG)
ax1.bar(x - w, tr_rmse,  w, label="Train", color=BLUE,   alpha=0.85)
ax1.bar(x,     val_rmse, w, label="Val",   color=ORANGE, alpha=0.85)
ax1.bar(x + w, te_rmse,  w, label="Test",  color=RED,    alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(names2, rotation=30, ha="right", fontsize=7.5)
ax1.set_ylabel("RMSE", fontsize=9); ax1.legend(fontsize=8)
ax1.set_title("Train / Val / Test RMSE", fontsize=10, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)

# Right: gap bar chart with threshold line
gap_colors = [RED if g > 0.01 else (ORANGE if g > 0.004 else BLUE) for g in gaps]
ax2.set_facecolor(BG)
brs = ax2.bar(names2, gaps, color=gap_colors, edgecolor="white")
ax2.axhline(0.010, color=RED,    lw=1.5, ls="--", label="Overfit threshold (0.010)")
ax2.axhline(0.004, color=ORANGE, lw=1.2, ls=":",  label="Watch zone (0.004)")
ax2.set_xticks(range(len(names2)))
ax2.set_xticklabels(names2, rotation=30, ha="right", fontsize=7.5)
ax2.set_ylabel("Val − Train RMSE  (gap)", fontsize=9)
ax2.set_title("Overfitting gap per model", fontsize=10, fontweight="bold")
ax2.legend(fontsize=7.5); ax2.spines[["top","right"]].set_visible(False)
for bar, g in zip(brs, gaps):
    ax2.text(bar.get_x()+bar.get_width()/2, g+0.0001,
             f"{g:.4f}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT}/overfitting_analysis.png", dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Learned adjacency heatmap (copy from Traffic_Forecast if exists)
# ══════════════════════════════════════════════════════════════════════════════
import shutil
if os.path.exists("learned_adj.png"):
    shutil.copy("learned_adj.png", f"{OUT}/learned_adj.png")
    print("5/7  learned_adj.png (copied)")
else:
    print("5/7  learned_adj.png (not found, skipping)")


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Pure MLP vs CNN1D-GCN highlight
# ══════════════════════════════════════════════════════════════════════════════
print("6/7  mlp_vs_gcn.png")
fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")

boxes = [
    (0.12, 0.25, 0.30, 0.50, GREEN,  "Pure MLP\n+ Embedding",
     "9,601 params\nTest RMSE: 0.0357\nGap: +0.0021"),
    (0.58, 0.25, 0.30, 0.50, ORANGE, "CNN1D-GCN\n(h = 160)",
     "171,233 params\nTest RMSE: 0.0359\nGap: +0.0050"),
]
for (x, y, w, h, col, title, body) in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=col, alpha=0.15, edgecolor=col, lw=2,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.08, title, ha="center", va="top",
            fontsize=12, fontweight="bold", color=col,
            transform=ax.transAxes)
    ax.text(x+w/2, y+h/2-0.04, body, ha="center", va="center",
            fontsize=9.5, color=DARK, transform=ax.transAxes,
            multialignment="center")

ax.text(0.5, 0.88, "Same accuracy · 18× fewer parameters",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=DARK, transform=ax.transAxes)
ax.annotate("", xy=(0.57, 0.50), xytext=(0.43, 0.50),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=2))
ax.text(0.5, 0.44, "≈ equal RMSE", ha="center", fontsize=9,
        color=DARK, transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f"{OUT}/mlp_vs_gcn.png", dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Monday traffic GIF (24 frames)
# ══════════════════════════════════════════════════════════════════════════════
print("7/7  monday_traffic.gif  (24 frames) ...")
monday_mask = weekday == 0
typical_mon = np.zeros((24, n_sensors))
counts      = np.zeros(24)
for t in range(len(Y_all)):
    if weekday[t] == 0:
        h = int(hour[t])
        typical_mon[h] += Y_all[t]
        counts[h] += 1
for h in range(24):
    if counts[h] > 0:
        typical_mon[h] /= counts[h]

ROAD_C_S = [ROAD_COLORS[int(road_id[i])] for i in range(n_sensors)]

fig_gif, ax_gif = plt.subplots(figsize=(7, 5), facecolor=BG)

def draw_frame(h):
    ax_gif.clear()
    ax_gif.set_facecolor(BG)
    vol = typical_mon[h]
    # edges
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            if adj[i, j] > 0.05:
                ax_gif.plot([lon_arr[i], lon_arr[j]], [lat_arr[i], lat_arr[j]],
                            color="#cccccc", lw=1.0, alpha=0.6, zorder=1)
    sc2 = ax_gif.scatter(lon_arr, lat_arr,
                         c=vol, cmap="YlOrRd", vmin=0, vmax=0.8,
                         s=(30 + vol * 200), zorder=4,
                         edgecolors=DARK, linewidths=0.8)
    for i in range(n_sensors):
        ax_gif.text(lon_arr[i], lat_arr[i], str(i),
                    ha="center", va="center", fontsize=4.5,
                    fontweight="bold", color="white", zorder=5)
    ax_gif.set_title(f"Monday  {h:02d}:00  — mean traffic flow",
                     fontsize=10, fontweight="bold")
    ax_gif.set_xlabel("Longitude", fontsize=8)
    ax_gif.set_ylabel("Latitude",  fontsize=8)
    ax_gif.set_xlim(lon_arr.min()-0.03, lon_arr.max()+0.03)
    ax_gif.set_ylim(lat_arr.min()-0.02, lat_arr.max()+0.02)
    return sc2,

anim = FuncAnimation(fig_gif, draw_frame, frames=24,
                     interval=500, blit=False)
anim.save(f"{OUT}/monday_traffic.gif",
          writer=PillowWriter(fps=2), dpi=110)
plt.close(fig_gif)

# Also save individual frames for animate package
os.makedirs(f"{OUT}/monday_frames", exist_ok=True)
for h in range(24):
    fig_f, ax_f = plt.subplots(figsize=(5.5, 4), facecolor=BG)
    ax_f.set_facecolor(BG)
    vol = typical_mon[h]
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            if adj[i, j] > 0.05:
                ax_f.plot([lon_arr[i], lon_arr[j]], [lat_arr[i], lat_arr[j]],
                          color="#cccccc", lw=1.0, alpha=0.6, zorder=1)
    ax_f.scatter(lon_arr, lat_arr,
                 c=vol, cmap="YlOrRd", vmin=0, vmax=0.8,
                 s=(30 + vol*200), zorder=4,
                 edgecolors=DARK, linewidths=0.8)
    for i in range(n_sensors):
        ax_f.text(lon_arr[i], lat_arr[i], str(i),
                  ha="center", va="center", fontsize=4.5,
                  fontweight="bold", color="white", zorder=5)
    ax_f.set_title(f"Monday  {h:02d}:00", fontsize=9, fontweight="bold")
    ax_f.set_xlabel("Longitude", fontsize=7); ax_f.set_ylabel("Latitude", fontsize=7)
    ax_f.set_xlim(lon_arr.min()-0.03, lon_arr.max()+0.03)
    ax_f.set_ylim(lat_arr.min()-0.02, lat_arr.max()+0.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/monday_frames/frame{h:02d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig_f)

print("Done. All images saved to", OUT)
