"""
Temporal Transformer + Cross-Sensor Attention model.

Per sensor: lightweight self-attention over the 10 time steps (d_model=32).
Between sensors: one cross-sensor attention layer lets each sensor
query its peers, discovering spatial dependencies data-driven (no fixed graph).
Fused with static features and a learnable sensor embedding.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mat-file", default="traffic_dataset.mat")
    parser.add_argument("--dataset-file", default="dataset.npz")
    parser.add_argument("--model-file", default="model_transformer.pth")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="huber",
                        choices=["huber", "mse", "smoothl1", "mae"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--final-train", action="store_true", default=False)
    parser.add_argument("--stratify", type=str, default="true",
                        choices=["true", "false"])
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrafficDataset(Dataset):
    def __init__(self, X, Y, continuous_idx=None, x_mean=None, x_std=None,
                 y_mean=None, y_std=None):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.continuous_idx = list(range(10)) if continuous_idx is None else continuous_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx].copy()
        if self.x_mean is not None:
            x[:, self.continuous_idx] = (x[:, self.continuous_idx] - self.x_mean) / self.x_std
        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TemporalTransformer(nn.Module):
    """
    Architecture:
      1. Per sensor: project scalar traffic history [T=10] into d_model,
         add learnable positional encoding, run TransformerEncoder.
         Mean-pool over time -> temporal summary [d_model] per sensor.
      2. Static branch: Linear(38->32) + ReLU.
      3. Sensor embedding: nn.Embedding(36, 8).
      4. Combine (cat): [d_model + 32 + 8] -> hidden_size.
      5. Cross-sensor attention: each sensor attends to all others
         (no fixed graph, fully data-driven spatial dependencies).
      6. Head: hidden_size -> 32 -> 1.
    """
    def __init__(self, hidden_size=64, static_size=38, dropout=0.15,
                 n_sensors=36, adj_mask=None,
                 d_model=32, nhead=2, num_encoder_layers=2):
        super().__init__()
        embedding_size = 8
        self.d_model = d_model

        self.sensor_emb = nn.Embedding(n_sensors, embedding_size)
        self.pos_enc = nn.Embedding(10, d_model)
        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.combine = nn.Sequential(
            nn.Linear(d_model + 32 + embedding_size, hidden_size),
            nn.ReLU(),
        )

        # Cross-sensor attention: sensors attend to each other
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4,
            dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, adj=None):
        seq = x[:, :, :10]
        static = x[:, :, 10:]
        B, N, T = seq.shape

        # Temporal encoder: process each sensor's time series independently
        seq_flat = seq.reshape(B * N, T, 1)
        h = self.input_proj(seq_flat)                           # [B*N, T, d_model]
        pos = torch.arange(T, device=x.device)
        h = h + self.pos_enc(pos).unsqueeze(0)                  # add pos encoding
        h = self.temporal_encoder(h)                             # [B*N, T, d_model]
        h = h.mean(dim=1).reshape(B, N, self.d_model)           # [B, N, d_model]

        s = self.static_mlp(static)                              # [B, N, 32]

        sensor_ids = torch.arange(N, device=x.device)
        e = self.sensor_emb(sensor_ids).unsqueeze(0).expand(B, -1, -1)  # [B, N, 8]

        z = self.combine(torch.cat([h, s, e], dim=-1))          # [B, N, hidden]

        # Cross-sensor attention + residual
        z_attn, _ = self.cross_attn(z, z, z)
        z = self.cross_norm(z + z_attn)
        z = self.ff_norm(z + self.cross_ff(z))

        return self.head(z).squeeze(-1)                          # [B, N]


def normalize_adj(adj):
    adj = adj.astype(np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    deg = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(deg, 1e-6)


def evaluate(model, loader, device, loss_fn, adj, y_mean, y_std):
    model.eval()
    total_loss = total_sq = total_abs = total = 0.0
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float32, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += float(loss_fn(pred, y).item()) * len(y)
            pred_real = pred * y_std_t + y_mean_t
            pred_real = pred_real.clamp(0.0, 1.0)
            y_real    = y * y_std_t + y_mean_t
            total_sq  += float(torch.sum((pred_real - y_real) ** 2).item())
            total_abs += float(torch.sum(torch.abs(pred_real - y_real)).item())
            total     += y_real.numel()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    rmse = np.sqrt(total_sq / max(total, 1))
    mae  = total_abs / max(total, 1)
    return avg_loss, rmse, mae


def dataset_exists(data_dir=".", mat_file="traffic_dataset.mat", npz_file="dataset.npz"):
    mat_path = os.path.join(data_dir, mat_file)
    npz_path = os.path.join(data_dir, npz_file)
    if os.path.exists(npz_path):
        print(f"Found existing dataset: {npz_path}")
        return npz_path
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing required dataset file: {mat_path}")
    print(f"{npz_file} not found, creating from {mat_file}")
    mat = loadmat(mat_path)
    X_train = np.array([m.toarray() for m in mat['tra_X_tr'][0]], dtype=np.float32)
    X_test  = np.array([m.toarray() for m in mat['tra_X_te'][0]], dtype=np.float32)
    Y_train = mat["tra_Y_tr"].T.astype(np.float32)
    Y_test  = mat["tra_Y_te"].T.astype(np.float32)
    adj_mat = mat['tra_adj_mat']
    np.savez(npz_path, X_train=X_train, X_test=X_test,
             Y_train=Y_train, Y_test=Y_test, adj_mat=adj_mat)
    print(f"Saved dataset to: {npz_path}")
    return npz_path


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = dataset_exists(data_dir=args.data_dir)
    data = np.load(data_path)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test,  Y_test  = data["X_test"],  data["Y_test"]
    adj_mat = data["adj_mat"]

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)

    if args.final_train:
        print("FINAL TRAINING MODE: full dataset, no early stopping.")
        train_idx, val_idx = indices, None
    else:
        if args.stratify == "true":
            y_level = Y_train.mean(axis=1)
            y_bin = pd.qcut(y_level, q=4, labels=False, duplicates="drop")
            train_idx, val_idx = train_test_split(
                np.arange(len(X_train)),
                test_size=args.val_frac,
                random_state=args.seed,
                shuffle=True,
                stratify=y_bin,
            )
        else:
            split = int((1.0 - args.val_frac) * n)
            train_idx, val_idx = indices[:split], indices[split:]

    X_tr, Y_tr = X_train[train_idx], Y_train[train_idx]
    X_val, Y_val = (X_train[val_idx], Y_train[val_idx]) if val_idx is not None else (None, None)

    continuous_idx = list(range(10)) + [39]
    x_mean = X_tr[:, :, continuous_idx].mean(axis=(0, 1))
    x_std  = X_tr[:, :, continuous_idx].std(axis=(0, 1)) + 1e-6
    y_mean = float(Y_tr.mean())
    y_std  = float(Y_tr.std()) + 1e-6

    train_ds = TrafficDataset(X_tr, Y_tr, continuous_idx, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if not args.final_train:
        val_ds = TrafficDataset(X_val, Y_val, continuous_idx, x_mean, x_std, y_mean, y_std)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    adj_mask = torch.tensor((adj_mat > 0.5).astype(np.float32))
    adj_norm = normalize_adj(adj_mat)
    adj_t    = torch.tensor(adj_norm, dtype=torch.float32).to(device)

    model = TemporalTransformer(
        hidden_size=args.hidden_size,
        static_size=X_train.shape[-1] - 10,
        dropout=args.dropout,
        n_sensors=adj_mat.shape[0],
        adj_mask=adj_mask,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    loss_fn = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(delta=args.huber_delta),
        "smoothl1": nn.SmoothL1Loss(),
    }[args.loss]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    best_rmse, best_mae = float("inf"), None
    best_train_rmse, best_train_mae = None, None
    best_state = None
    best_epoch = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_n    += len(y)

        train_loss = total_loss / max(total_n, 1)

        if not args.final_train:
            val_loss, val_rmse, val_mae = evaluate(
                model, val_loader, device, loss_fn, adj_t, y_mean, y_std)
            train_eval_loss, train_rmse, train_mae = evaluate(
                model, train_loader, device, loss_fn, adj_t, y_mean, y_std)
            scheduler.step(val_rmse)

            if val_rmse < best_rmse - args.min_delta:
                best_rmse      = val_rmse
                best_mae       = val_mae
                best_train_rmse = train_rmse
                best_train_mae = train_mae
                best_epoch     = epoch + 1
                best_state     = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"epoch {epoch+1:03d} : train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"train_rmse={train_rmse:.4f}  "
                  f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}  "
                  f"best={best_rmse:.4f}")

            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}. Best RMSE={best_rmse:.4f}")
                break
        else:
            print(f"epoch {epoch+1:03d} FINAL | train={train_loss:.4f}")

    if args.final_train:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_version":    "transformer_mlp",
        "hidden_size":      args.hidden_size,
        "d_model":          args.d_model,
        "nhead":            args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "dropout":          args.dropout,
        "n_sensors":        adj_mat.shape[0],
        "adj_mask":         adj_mask,
        "x_mean":           torch.tensor(x_mean, dtype=torch.float32),
        "x_std":            torch.tensor(x_std,  dtype=torch.float32),
        "y_mean":           torch.tensor(y_mean, dtype=torch.float32),
        "y_std":            torch.tensor(y_std,  dtype=torch.float32),
        "huber_delta":      args.huber_delta,
    }
    torch.save(checkpoint, args.model_file)
    print(f"Saved model to {args.model_file}  (best val RMSE={best_rmse:.4f})")

    results_file = "runs/results.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    row = {
        "model_file":       os.path.basename(args.model_file),
        "model_version":    "transformer_mlp",
        "final_train":      args.final_train,
        "hidden_size":      args.hidden_size,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "dropout":          args.dropout,
        "seed":             args.seed,
        "loss":             args.loss,
        "huber_delta":      args.huber_delta,
        "stratify":         args.stratify,
        "total_params":     total_params,
        "best_epoch":       best_epoch if not args.final_train else args.epochs,
        "best_train_mae":   best_train_mae if not args.final_train else None,
        "best_train_rmse":  best_train_rmse if not args.final_train else None,
        "best_mae":         best_mae if not args.final_train else None,
        "best_rmse":        best_rmse if not args.final_train else None,
    }
    df = pd.DataFrame([row])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, index=False)


if __name__ == "__main__":
    train(parse_args())
