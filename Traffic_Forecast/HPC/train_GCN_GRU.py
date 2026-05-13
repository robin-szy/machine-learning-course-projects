"""
In comparison to v2, this version uses Conv1D instead of GRU
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


# -------------------------
# Arguments
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mat-file", default="traffic_dataset.mat")
    parser.add_argument("--dataset-file", default="dataset.npz")
    parser.add_argument("--model-file", default="model_v3.pth")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="huber",
                        choices=["huber", "mse", "smoothl1", "mae"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--final-train", action="store_true", default=False)
    parser.add_argument("--stratify", type=str, default="true",
                        choices=["true", "false"])
    return parser.parse_args()


# -------------------------
# General utilities 
# -------------------------
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


class AdaptiveGraphConv(nn.Module):
    """
    Graph convolution with learnable edge weights.

    The topology (which edges exist) is fixed from the adj_mask provided at
    construction, matching the SADL-I insight: we know the connectivity from
    the road network, but let the model learn the *strength* of each edge.

    Two improvements over the V1 fixed graph layer:
      1. Each edge weight is a free parameter, optimised end-to-end.
      2. A residual connection lets signal bypass the aggregation.
    """
    def __init__(self, in_size, out_size, n_sensors, adj_mask):
        super().__init__()
        self.register_buffer("adj_mask", adj_mask.float())  # Register buffer that should not be considered as a model parameter (optimizer does not train it)
        # Initialise logits at 0 so sigmoid(0)=0.5 — neither suppressed nor amplified
        self.edge_logits = nn.Parameter(torch.zeros(n_sensors, n_sensors))
        self.linear = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(out_size)

    def get_adj(self):
        w = torch.sigmoid(self.edge_logits) * self.adj_mask
        w = w + torch.eye(w.shape[0], device=w.device)   # self-loop
        deg = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return w / deg

    def forward(self, z):
        adj = self.get_adj()
        z_agg = torch.einsum("ij,bjf->bif", adj, z)
        return self.norm(torch.relu(self.linear(z_agg)))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class GCN_Conv1D(nn.Module):
    """
    GCN first, then temporal Conv1D.
    For each time step:
      1. combine current traffic value + static features + sensor embedding
      2. apply GCN over sensors
    Then:
      3. apply Conv1D over time for each sensor
    """
    def __init__(self, hidden_size=64, static_size=38, dropout=0.1,
                 n_sensors=36, adj_mask=None):
        super().__init__()

        embedding_size = 8
        self.sensor_emb = nn.Embedding(n_sensors, embedding_size)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # One traffic value per time step + static encoding + sensor embedding
        self.input_proj = nn.Sequential(
            nn.Linear(1 + 32 + embedding_size, hidden_size),
            nn.ReLU(),
        )

        if adj_mask is None:
            adj_mask = torch.ones(n_sensors, n_sensors)

        self.gcn1 = AdaptiveGraphConv(hidden_size, hidden_size, n_sensors, adj_mask)
        self.gcn2 = AdaptiveGraphConv(hidden_size, hidden_size, n_sensors, adj_mask)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        seq    = x[:, :, :10]     # [B, N, T]
        static = x[:, :, 10:]     # [B, N, static_size]
        B, N, T = seq.shape

        s = self.static_mlp(static)  # [B, N, 32]

        sensor_ids = torch.arange(N, device=x.device)
        e = self.sensor_emb(sensor_ids)          # [N, emb]
        e = e.unsqueeze(0).expand(B, -1, -1)     # [B, N, emb]

        gcn_outputs = []

        for t in range(T):
            traffic_t = seq[:, :, t:t+1]         # [B, N, 1]

            z_t = self.input_proj(
                torch.cat([traffic_t, s, e], dim=-1)
            )                                   # [B, N, H]

            z1 = self.gcn1(z_t)
            z2 = self.gcn2(z1)
            z_out = z2 + z_t                    # residual

            gcn_outputs.append(z_out)

        h = torch.stack(gcn_outputs, dim=2)      # [B, N, T, H]

        # Conv1D expects [batch, channels, time]
        h = h.reshape(B * N, T, -1)              # [B*N, T, H]
        h = h.transpose(1, 2)                    # [B*N, H, T]

        h = self.temporal_conv(h)                # [B*N, H, T]
        h = h.mean(dim=-1)                       # [B*N, H]

        h = h.reshape(B, N, -1)                  # [B, N, H]

        return self.head(h).squeeze(-1)

    def get_learned_adj(self):
        return self.gcn1.get_adj().detach().cpu().numpy()

# -------------------------
# Training  (identical logic to v1, different model class)
# -------------------------
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
            pred  = model(x)
            total_loss += float(loss_fn(pred, y).item()) * len(y)
            pred_real = pred * y_std_t + y_mean_t
            pred_real = pred_real.clamp(0.0, 1.0)   # Todo: Maybe remove. Output needs to be between 0 and 1.
            y_real    = y   * y_std_t + y_mean_t
            total_sq  += float(torch.sum((pred_real - y_real) ** 2).item())
            total_abs += float(torch.sum(torch.abs(pred_real - y_real)).item())
            total     += y_real.numel()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    rmse = np.sqrt(total_sq / max(total, 1))
    mae = total_abs / max(total, 1)

    return avg_loss, rmse, mae



def dataset_exists(data_dir=".", mat_file="traffic_dataset.mat", npz_file="dataset.npz"):
    """
    It basically just does what the main-chekcpoint.ipynb does. If the npz-dataset does
    not exist, yet, then it looks for the mat-file and converts it to the npz-file.
    """

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
            print("Stratifying validation set")

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

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Train/validation split failed. Check data size and --val-frac.")


    X_tr, Y_tr = X_train[train_idx], Y_train[train_idx]
    X_val, Y_val = (X_train[val_idx], Y_train[val_idx]) if val_idx is not None else (None, None)

    # For normalization, only from training split
    continuous_idx = list(range(10)) + [39]
    x_mean = X_tr[:, :, continuous_idx].mean(axis=(0, 1))
    x_std  = X_tr[:, :, continuous_idx].std(axis=(0, 1))  + 1e-6
    y_mean = float(Y_tr.mean())
    y_std  = float(Y_tr.std()) + 1e-6

    train_ds = TrafficDataset(X_tr, Y_tr, continuous_idx, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if not args.final_train:
        val_ds = TrafficDataset(X_val, Y_val, continuous_idx, x_mean, x_std, y_mean, y_std)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Build adj_mask (binary topology) for the learnable GCN
    adj_mask = torch.tensor((adj_mat > 0.5).astype(np.float32))

    # Fixed normalised adj still passed to forward() for API parity (model ignores it)
    adj_norm = normalize_adj(adj_mat)
    adj_t    = torch.tensor(adj_norm, dtype=torch.float32).to(device)

    model = GCN_Conv1D(
        hidden_size=args.hidden_size,
        static_size=X_train.shape[-1] - 10,
        dropout=args.dropout,
        n_sensors=adj_mat.shape[0],
        adj_mask=adj_mask,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss == "mae":
        loss_fn = nn.L1Loss()
    elif args.loss == "huber":
        loss_fn = nn.HuberLoss(delta=args.huber_delta)
    else:
        loss_fn = nn.SmoothL1Loss()

    # Optimizer
    if args.weight_decay == 0.0:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    best_rmse, best_mae = float("inf"), None
    best_train_rmse, best_train_mae = None, None
    best_state = None
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
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_n    += len(y)

        train_loss = total_loss / max(total_n, 1)

        if not args.final_train:
            val_loss, val_rmse, val_mae = evaluate(
                model, val_loader, device, loss_fn, adj_t, y_mean, y_std)

            train_eval_loss, train_rmse, train_mae = evaluate(
                model, train_loader, device, loss_fn, adj_t, y_mean, y_std
            )

            if val_rmse < best_rmse - args.min_delta:
                best_rmse = val_rmse
                best_mae  = val_mae
                best_train_rmse = train_rmse
                best_train_mae = train_mae
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"epoch {epoch + 1:03d} : train_loss={train_loss:.4f}  "
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
        "model_version":    "conv1d_gcn",
        "hidden_size":      args.hidden_size,
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
    print(f"Saved model to {args.model_file}  (best RMSE={best_rmse:.4f})")

    # Append to shared results CSV
    results_file = "runs/results.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    row = {
        "model_file": os.path.basename(args.model_file),
        "model_version": "conv1d_gcn",
        "final_train": args.final_train,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "stratify": args.stratify,
        "total_params": total_params,
        "best_train_mae": best_train_mae if not args.final_train else None,
        "best_train_rmse": best_train_rmse if not args.final_train else None,
        "best_mae":  best_mae  if not args.final_train else None,
        "best_rmse": best_rmse if not args.final_train else None,
    }
    df = pd.DataFrame([row])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, index=False)


if __name__ == "__main__":
    train(parse_args())
