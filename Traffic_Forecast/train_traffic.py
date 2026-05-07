
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from scipy.io import loadmat


# -------------------------
# Arguments for parsing
# (Mostly used for testing on HPC)
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mat-file", default="traffic_dataset.mat")
    parser.add_argument("--dataset-file", default="dataset.npz")
    parser.add_argument("--model-file", default="model.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="huber",
                        choices=["huber", "mse", "smoothl1", "mae"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--final-train", action="store_true", default=False)
    return parser.parse_args()


# -----------------
# General functions
# -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrafficDataset(Dataset):
    # The same as in first homework
    def __init__(self, X, Y, continuous_idx=None, x_mean=None, x_std=None, y_mean=None, y_std=None):
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

        # Normalization
        if self.x_mean is not None:
            x[:,self.continuous_idx] = (x[:,self.continuous_idx] - self.x_mean) / self.x_std

        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -----------------
# Model
# -----------------
class GRU_GCN(nn.Module):
    # The input sizes so I don't have to adapt the size every time when I add/remove a feature
    def __init__(self, hidden_size=64, static_size=38, dropout=0.1):
        super().__init__()

        # GRU for the sequential traffic volume data
        self.gru = nn.GRU(
            input_size = 1, # Because currently we input one sensor independently at once
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # A small MLP for the static stuff
        self.static_MLP = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combine both
        self.combine = nn.Sequential(
            nn.Linear(hidden_size+32, hidden_size),
            nn.ReLU(),
        )

        self.graph_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, adj):

        seq = x[:, :, :10]      # [B samples, 36, 10]
        static = x[:, :, 10:]   # [B samples, 36, 38]

        B, N, T = seq.shape

        # Treat every sensor as one short sequence of length 10
        # Todo: Discuss with Alex: The 36 sensors are dependent, but here for the GRU, we treat them as independent measurements.
        # We can maybe turn it around and do graph convolution first, then feed into GRU?
        seq = seq.reshape(B * N, T, 1)  # [B*36, 10, 1]

        _, h = self.gru(seq)
        h = h[-1].reshape(B, N, -1)  # [B, 36, hidden_size]. Hidden size learns stuff like rising trend or sudden drops, etc

        s = self.static_MLP(static)  # [B, 36, 32]

        z = torch.cat([h, s], dim=-1)
        z = self.combine(z)

        # Simple graph
        # Einstein's sum convention:
        # For each sensor node, batch sample and feature,
        # you sum over all neighboring nodes.
        # So you take weighted combination of all adjacent nodes that are connected.
        z = torch.einsum("ij,bjf->bif", adj, z)
        z = self.graph_layer(z)
        # Todo: Replace graph e.g. by a convolutional graph with more layers? Currently, we only look at direct neighbors. With more layers, we can consider neighbors of neighbors.


        pred = self.head(z).squeeze(-1)  # [B, 36]
        return pred


def normalize_adj(adj):
    adj = adj.astype(np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    deg = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(deg, 1e-6)


# -----------------
# Training
# -----------------
def train(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_path = dataset_exists(
        data_dir=args.data_dir,
    )
    data = np.load(data_path)

    X_train = data["X_train"]  # [1261, 36, 48]
    Y_train = data["Y_train"]  # [1261, 36]
    X_test = data["X_test"]  # [840, 36, 48]
    Y_test = data["Y_test"]  # [840, 36]
    adj_mat = data["adj_mat"]  # [36, 36]

    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)

    # Split training set into train and val-set
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)  # Uses global seed

    if args.final_train:
        print(
            "FINAL TRAINING MODE: Model is trained on full data set. No validation metrics or early stopping are used during training.")
        train_idx = indices
        val_idx = None
    else:
        split = int((1.0 - args.val_frac) * n)
        train_idx = indices[:split]
        val_idx = indices[split:]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Train/validation split failed. Check data size and --val-frac.")

    X_tr, Y_tr = X_train[train_idx], Y_train[train_idx]
    if not args.final_train:
        X_val, Y_val = X_train[val_idx], Y_train[val_idx]
    else:
        X_val, Y_val = None, None

    # For normalization, only from training split
    # Todo: Find out which columns are continuous
    continuous_idx = list(range(10)) + [39] # Let's only normalize continuous columns
    x_mean = X_tr[:, :, continuous_idx].mean(axis=(0, 1))
    x_std = X_tr[:, :, continuous_idx].std(axis=(0, 1)) + 1e-6

    # Todo: Look at y distribution. Does it make sense to normalize? Maybe bimodal? There are some cool tricks if bimodal, for example.
    y_mean = Y_tr.mean()
    y_std = Y_tr.std() + 1e-6

    train_dataset = TrafficDataset(
        X=X_tr,
        Y=Y_tr,
        continuous_idx=continuous_idx,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    # Load datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    if not args.final_train:

        val_dataset = TrafficDataset(
            X=X_val,
            Y=Y_val,
            continuous_idx=continuous_idx,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
    else:
        val_loader = None

    # Normalize adjacency and move to device
    # We add unity matrix, so we create a self-loop of each node in the graph
    # We should do this because we want to also consider own traffic data
    # Then we normalize by number of degrees, so that for each node, all the traffic data of itself and neighbors
    # gets averaged (otherwise, highly connected nodes get more weight).
    # Todo: Try without self-loop. Play around. Try without normalization
    adj = normalize_adj(adj_mat)
    adj = torch.tensor(adj, dtype=torch.float32).to(device)

    # Model
    model = GRU_GCN(
        hidden_size=args.hidden_size,
        static_size=X_train.shape[-1] - 10,
        dropout=args.dropout,
    ).to(device)


    # How many model parameters do I have?
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Loss function
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

    best_rmse = float("inf")
    best_mae = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):

        model.train()
        total_train_loss, total_train = 0.0, 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x, adj)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(y)
            total_train += len(y)

        train_loss = total_train_loss / max(total_train, 1)

        if not args.final_train:
            val_loss, val_rmse, val_mae = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                adj=adj,
                y_mean=y_mean,
                y_std=y_std,
            )

            improved = val_rmse < best_rmse - args.min_delta
            if improved:
                best_rmse = val_rmse
                best_mae = val_mae
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"epoch {epoch + 1:03d} : "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_rmse={val_rmse:.4f} "
                f"val_mae={val_mae:.4f} "
                f"best_rmse={best_rmse:.4f}"
            )

        else:
            print(
                f"epoch {epoch + 1:03d} "
                f"FINAL TRAINING |"
                f"train_loss={train_loss:.4f}"
            )

        # No early stopping during final training
        if not args.final_train and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                  f"Best val RMSE={best_rmse:.4f}"
                  )
            break

    if args.final_train:    # The best model is the last model during final training
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(x_std, dtype=torch.float32),
        "y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "y_std": torch.tensor(y_std, dtype=torch.float32),
        "huber_delta": args.huber_delta,
    }

    torch.save(checkpoint, args.model_file)
    if args.final_train:
        print(f"Saved final model to {args.model_file}")
    else:
        print(
            f"Saved best model to {args.model_file} with RMSE={best_rmse:.4f}")

    # Summary for HPC testing (to not open every log file every time)
    results_file = "runs/results.csv"
    row = {
        "model_file": os.path.basename(args.model_file),
        "final_train": args.final_train,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "total_params": total_params,
        "best_mae": best_mae if not args.final_train else None,
        "best_rmse": best_rmse if not args.final_train else None,
    }

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, index=False)


# -----------------
# Evaluation
# -----------------
def evaluate(model, loader, device, loss_fn, adj, y_mean, y_std):
    model.eval()
    total_loss, total_sq_error, total_abs_error, total = 0.0, 0.0, 0.0, 0

    y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std = torch.tensor(y_std, dtype=torch.float32, device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x, adj)

            # Keep loss comparable to training loss
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * len(y)

            # Metrics in original traffic-volume units. Before, we normalized.
            # Otherwise, metrics are normalized as well
            pred_real = pred * y_std + y_mean
            y_real = y * y_std + y_mean

            total_sq_error += float(torch.sum((pred_real - y_real) ** 2).item())
            total_abs_error += float(torch.sum(torch.abs(pred_real - y_real)).item())
            total += y_real.numel()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    rmse = np.sqrt(total_sq_error / max(total, 1))
    mae = total_abs_error / max(total, 1)

    return avg_loss, rmse, mae



def dataset_exists(data_dir = ".", mat_file="traffic_dataset.mat", npz_file="dataset.npz"):
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

    print(f"{npz_file} not found, therefore creating it from {mat_file}")

    mat = loadmat(mat_path)

    X_train = np.array([m.toarray() for m in mat['tra_X_tr'][0]], dtype=np.float32)
    X_test = np.array([m.toarray() for m in mat['tra_X_te'][0]], dtype=np.float32)
    #Y_train = np.array([m for m in mat['tra_Y_tr'][0]])    # Todo: Discuss with Alex: In Jupyter, prof did like this
    #Y_test = np.array([m for m in mat['tra_Y_te'][0]])     # Todo: but it only takes values of sensor 0, and basically ignores 31 sensor entries
    Y_train = mat["tra_Y_tr"].T.astype(np.float32)  # [1261, 36]
    Y_test = mat["tra_Y_te"].T.astype(np.float32)  # [840, 36]
    adj_mat = mat['tra_adj_mat']

    np.savez(npz_path, X_train=X_train, X_test=X_test, Y_train=Y_train,
             Y_test=Y_test, adj_mat=adj_mat)

    print(f"Saved dataset to: {npz_path}")

    return npz_path



if __name__ == "__main__":

    train(parse_args())

    # For testing final models
    # import glob
    # for model_file in sorted(glob.glob("runs/*.pth")):
    #     print("\n", model_file)
    #     test_on_labeled_set(
    #         test_dir="scanpaths/test",
    #         labels_file="scanpaths/test_labels.csv",
    #         model_file=model_file,
    #     )



# -----------------
# Final testing
# -----------------

# def test_on_labeled_set(test_dir="scanpaths/test",
#                         labels_file="scanpaths/test_labels.csv",
#                         model_file="model.pth"):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     checkpoint = torch.load(model_file, map_location=device, weights_only=True)
#
#     model = GRU_GCN(
#         hidden_size=int(checkpoint["hidden_size"]),
#         dropout=float(checkpoint.get("dropout", 0.0)),
#     ).to(device)
#
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#
#     seq_mean = checkpoint["seq_mean"].cpu().numpy()
#     seq_std = checkpoint["seq_std"].cpu().numpy()
#
#     labels = pd.read_csv(
#         labels_file,
#         sep=r"\s+",
#         header=None,
#         names=["reaction_time", "filename"],
#         engine="python",
#     )
#
#     y_true = []
#     y_pred = []
#     rows = []
#
#     with torch.no_grad():
#         for _, row in labels.iterrows():
#             path = os.path.join(test_dir, str(row["filename"]))
#             if not os.path.exists(path):
#                 print(f"Missing file, skipped: {path}")
#                 continue
#
#             true_rt = float(row["reaction_time"])
#
#             seq = read_sequence(path)
#             seq = (seq - seq_mean) / seq_std
#
#             x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
#             lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
#
#             pred_rt = model(x, lengths).item()
#
#             y_true.append(true_rt)
#             y_pred.append(pred_rt)
#
#             rows.append({
#                 "filename": row["filename"],
#                 "true_rt": true_rt,
#                 "pred_rt": float(pred_rt),
#                 "abs_error": abs(pred_rt - true_rt),
#                 "sq_error": (pred_rt - true_rt) ** 2,
#             })
#
#     y_true = np.array(y_true, dtype=np.float32)
#     y_pred = np.array(y_pred, dtype=np.float32)
#
#     rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
#     mae = float(np.mean(np.abs(y_pred - y_true)))
#
#     print(f"Test samples: {len(y_true)}")
#     print(f"Test RMSE: {rmse:.4f}")
#     print(f"Test MAE:  {mae:.4f}")
#
#     return {
#         "rmse": rmse,
#         "mae": mae,
#         "n": len(y_true),
#         "predictions": pd.DataFrame(rows),
#     }