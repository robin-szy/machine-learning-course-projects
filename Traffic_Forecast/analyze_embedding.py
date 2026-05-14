"""
Analyze what the sensor embedding learned in the Pure MLP model.

Trains the Pure MLP h=96 model (or loads if checkpoint exists), extracts
the 36x8 embedding matrix, then:
  1. PCA of embedding -> shows dominant axes
  2. Correlates each embedding dimension with road features
     (road_id, direction, lane count, sensor position)
  3. Saves embedding_analysis.png for slides
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH  = os.path.join(SCRIPT_DIR, "model_pure_mlp_emb.pth")
OUT_IMAGE  = os.path.join(SCRIPT_DIR,
    "Intro_to_Machine_Learning___Slides_Final_Presentation/Images/embedding_analysis.png")

SEED = 523

# ── Road metadata ─────────────────────────────────────────────────────────────
ROAD_LABELS = ["I-95", "I-66", "I-495 NW", "I-495 SW"]
ROAD_COLORS = ["#c0392b", "#2980b9", "#27ae60", "#e67e22"]


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class TrafficDataset(Dataset):
    def __init__(self, X, Y, continuous_idx, x_mean, x_std, y_mean, y_std):
        self.X, self.Y = X.astype(np.float32), Y.astype(np.float32)
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std
        self.continuous_idx = continuous_idx

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy(); y = self.Y[idx].copy()
        x[:, self.continuous_idx] = (x[:, self.continuous_idx] - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ── Pure MLP model (mirror of pure_MLP_hl96.py) ──────────────────────────────
class PureMLP(nn.Module):
    def __init__(self, hidden_size=96, static_size=38, dropout=0.2,
                 n_sensors=36, adj_mask=None):
        super().__init__()
        embedding_size = 8
        self.sensor_emb = nn.Embedding(n_sensors, embedding_size)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32), nn.ReLU(), nn.Dropout(dropout))
        self.combine = nn.Sequential(
            nn.Linear(10 + 32 + embedding_size, hidden_size), nn.ReLU())
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, adj=None):
        seq = x[:, :, :10]; static = x[:, :, 10:]
        B, N, _ = seq.shape
        s = self.static_mlp(static)
        e = self.sensor_emb(torch.arange(N, device=x.device)).unsqueeze(0).expand(B, -1, -1)
        z = self.combine(torch.cat([seq, s, e], dim=-1))
        h = self.post_mlp(z)
        return self.head(h + z).squeeze(-1)


def evaluate(model, loader, device, loss_fn, y_mean, y_std):
    model.eval(); total_sq = total = 0.0
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred_real = (model(x) * y_std_t + y_mean_t).clamp(0.0, 1.0)
            y_real    = y * y_std_t + y_mean_t
            total_sq += float(torch.sum((pred_real - y_real) ** 2).item())
            total    += y_real.numel()
    return float(np.sqrt(total_sq / max(total, 1)))


def train_pure_mlp(device):
    data_path = os.path.join(SCRIPT_DIR, "dataset.npz")
    data = np.load(data_path)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test,  Y_test  = data["X_test"],  data["Y_test"]
    adj_mat = data["adj_mat"]

    y_level = Y_train.mean(axis=1)
    y_bin   = pd.qcut(y_level, q=4, labels=False, duplicates="drop")
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.2,
        random_state=SEED, stratify=y_bin)

    X_tr, Y_tr   = X_train[train_idx], Y_train[train_idx]
    X_val, Y_val = X_train[val_idx],   Y_train[val_idx]
    continuous_idx = list(range(10)) + [39]
    x_mean = X_tr[:, :, continuous_idx].mean(axis=(0, 1))
    x_std  = X_tr[:, :, continuous_idx].std(axis=(0, 1)) + 1e-6
    y_mean = float(Y_tr.mean()); y_std = float(Y_tr.std()) + 1e-6

    def make_loader(X, Y, shuffle):
        ds = TrafficDataset(X, Y, continuous_idx, x_mean, x_std, y_mean, y_std)
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    train_loader = make_loader(X_tr,   Y_tr,   True)
    val_loader   = make_loader(X_val,  Y_val,  False)
    test_loader  = make_loader(X_test, Y_test, False)

    model = PureMLP(hidden_size=96, static_size=X_train.shape[-1]-10,
                    dropout=0.2, n_sensors=adj_mat.shape[0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn   = nn.HuberLoss(delta=1.0)

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(150):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        val_rmse = evaluate(model, val_loader, device, loss_fn, y_mean, y_std)
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 30:
            print(f"  Early stop epoch {epoch+1}, best val={best_val:.4f}")
            break

    model.load_state_dict(best_state)
    test_rmse = evaluate(model, test_loader, device, loss_fn, y_mean, y_std)
    print(f"  Pure MLP  val RMSE={best_val:.4f}  test RMSE={test_rmse:.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "x_mean": torch.tensor(x_mean), "x_std": torch.tensor(x_std),
        "y_mean": torch.tensor(y_mean), "y_std": torch.tensor(y_std),
        "continuous_idx": continuous_idx,
        "adj_mask": torch.tensor((adj_mat > 0.5).astype(np.float32)),
    }
    torch.save(checkpoint, CKPT_PATH)
    return model, adj_mat, test_rmse


def load_features():
    """Extract road_id, direction, lane count from .mat file."""
    mat = loadmat(os.path.join(SCRIPT_DIR, "traffic_dataset.mat"))
    feats = np.array([m.toarray() for m in mat["tra_X_tr"][0]], dtype=np.float32)[0]
    road_id   = np.argmax(feats[:, 44:48], axis=1)   # one-hot cols 44-47
    direction = np.argmax(feats[:, 40:44], axis=1)   # one-hot cols 40-43
    lanes     = feats[:, 39].astype(int)
    # mean traffic per sensor (proxy for congestion level)
    data = np.load(os.path.join(SCRIPT_DIR, "dataset.npz"))
    mean_flow = data["X_train"][:, :, :10].mean(axis=(0, 2))  # [N]
    return road_id, direction, lanes, mean_flow


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.path.exists(CKPT_PATH):
        print("Loading existing Pure MLP checkpoint...")
        ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        data  = np.load(os.path.join(SCRIPT_DIR, "dataset.npz"))
        model = PureMLP(hidden_size=96, static_size=data["X_train"].shape[-1]-10,
                        dropout=0.2, n_sensors=data["adj_mat"].shape[0]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        adj_mat = data["adj_mat"]
        test_rmse = None
    else:
        print("Training Pure MLP...")
        model, adj_mat, test_rmse = train_pure_mlp(device)

    model.eval()

    # Extract embedding matrix [N, 8]
    emb = model.sensor_emb.weight.detach().cpu().numpy()  # [36, 8]
    N   = emb.shape[0]

    road_id, direction, lanes, mean_flow = load_features()

    # ── PCA ────────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)
    var_exp = pca.explained_variance_ratio_

    # ── Correlation table ──────────────────────────────────────────────────────
    # Compute Pearson r between each embedding dim and each feature
    features = {
        "Road (encoded)": road_id.astype(float),
        "Direction":      direction.astype(float),
        "Lanes":          lanes.astype(float),
        "Mean flow":      mean_flow,
    }
    corr = {}
    for feat_name, feat_vals in features.items():
        rs = []
        for dim in range(emb.shape[1]):
            r = float(np.corrcoef(emb[:, dim], feat_vals)[0, 1])
            rs.append(r)
        corr[feat_name] = rs

    corr_df = pd.DataFrame(corr, index=[f"Dim {i}" for i in range(emb.shape[1])])
    print("\nCorrelation (embedding dim vs road feature):")
    print(corr_df.round(3).to_string())

    max_abs_corr = corr_df.abs().max()
    print(f"\nStrongest feature correlations:")
    for feat, val in max_abs_corr.sort_values(ascending=False).items():
        best_dim = corr_df[feat].abs().idxmax()
        print(f"  {feat:<20} max |r| = {val:.3f}  (dim {best_dim})")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    for road in range(4):
        mask = road_id == road
        ax1.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    c=ROAD_COLORS[road], label=ROAD_LABELS[road],
                    s=90, edgecolors="k", linewidths=0.4, zorder=3)
    for i in range(N):
        ax1.annotate(str(i), emb_2d[i], fontsize=7,
                     ha="center", va="center", color="#333333")
    ax1.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax1.set_title("PCA - colored by road")
    ax1.legend(loc="best", fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1],
                     c=mean_flow, cmap="YlOrRd", s=90,
                     edgecolors="k", linewidths=0.4, zorder=3)
    for i in range(N):
        ax2.annotate(str(i), emb_2d[i], fontsize=7,
                     ha="center", va="center", color="#333333")
    plt.colorbar(sc, ax=ax2, shrink=0.8, label="Mean flow")
    ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax2.set_title("PCA - colored by mean flow")

    ax3 = fig.add_subplot(gs[2])
    corr_mat = np.array([corr[f] for f in features]).T  # [8, 4]
    im = ax3.imshow(corr_mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(features)))
    ax3.set_xticklabels(list(features.keys()), rotation=25, ha="right", fontsize=11)
    ax3.set_yticks(range(emb.shape[1]))
    ax3.set_yticklabels([f"Dim {i}" for i in range(emb.shape[1])], fontsize=11)
    ax3.set_title("Feature correlation")
    plt.colorbar(im, ax=ax3, shrink=0.8, label="Pearson r")
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            ax3.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="white" if abs(corr_mat[i,j]) > 0.5 else "black")

    title_suffix = f"  (test RMSE={test_rmse:.4f})" if test_rmse else ""
    fig.suptitle(f"Pure MLP - Sensor Embedding Analysis{title_suffix}", fontsize=14)
    plt.savefig(OUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"\nSaved {OUT_IMAGE}")
    plt.close()


if __name__ == "__main__":
    main()
