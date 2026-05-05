
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence



# -------------------------
# Arguments for parsing
# (Mostly used for testing on HPC)
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--metadata-file", default="scanpaths_metadata.csv")
    parser.add_argument("--scanpath-dir", default="scanpaths/train_val")
    parser.add_argument("--model-file", default="model.pth")
    parser.add_argument("--epochs", type=int, default=37)
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
    parser.add_argument("--bidirectional", action="store_true", default=False)
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


def read_sequence(path):
    #Main function creating all the features (with helper functions)

    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.upper().strip() for c in df.columns]

    # Sequential features

    # During data exploration, I found that duration is heavily right-skewed -> log-transform
    x = df["FPOGX"].to_numpy(dtype=np.float32)
    y = df["FPOGY"].to_numpy(dtype=np.float32)
    dur = df["FPOGD"].to_numpy(dtype=np.float32)
    dur_log = np.log1p(dur)

    # During data exploration, I found that dist is heavily right-skewed -> log-transform
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dist = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
    dist_log = np.log1p(dist)

    dt = np.diff(dur, prepend=dur[0])
    speed = dist / (np.abs(dt) + 1e-6)
    speed = np.clip(speed, 0.0, 20.0)
    speed_log = np.log1p(speed)

    angle = np.arctan2(dy, dx)
    angle_sin = np.sin(angle).astype(np.float32)
    angle_cos = np.cos(angle).astype(np.float32)
    angle_unwrapped = np.unwrap(angle)
    dangle = np.diff(angle_unwrapped, prepend=angle_unwrapped[0]).astype(np.float32)
    cum_time = np.cumsum(dur).astype(np.float32)

    seq = np.stack(
        [
            x,
            y,
            dt,
            speed_log,
            dist_log,
            angle_sin,
            angle_cos,
            dangle,
            dur_log,
            cum_time
        ],
        axis=1,
    ).astype(np.float32)

    return seq


def load_items(data_dir, metadata_file, scanpath_dir):

    meta_path = os.path.join(data_dir, metadata_file)
    scan_dir = os.path.join(data_dir, scanpath_dir)
    meta = pd.read_csv(meta_path, sep=r"\s+", header=None, names=["reaction_time", "filename"], engine="python")
    items = []
    for _, row in meta.iterrows():
        path = os.path.join(scan_dir, str(row["filename"]))
        if os.path.exists(path):
            items.append((path, float(row["reaction_time"])))
    if not items:
        raise ValueError(f"No usable scanpaths found from {meta_path} and {scan_dir}")
    return items


def compute_seq_norm(items):
    """
    Compute mean/std for sequence features using only the training set.
    """
    features = []
    for path, _ in items:
        seq = read_sequence(path)
        features.append(seq)
    features = np.concatenate(features, axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6

    return mean.astype(np.float32), std.astype(np.float32)


class ScanpathDataset(Dataset):
    # The same as in first homework
    def __init__(self, items, seq_mean=None, seq_std=None):
        self.items = items
        self.seq_mean = seq_mean
        self.seq_std = seq_std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq = read_sequence(path)

        if self.seq_mean is not None:
            seq = (seq - self.seq_mean) / self.seq_std

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_batch(batch):

    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    feat_dim = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, feat_dim)

    for i, seq in enumerate(seqs):
        x[i, :seq.shape[0]] = seq

    y = torch.stack(labels)
    return x, lengths, y


# -----------------
# Model
# -----------------
class GRURegressor(nn.Module):
    # The input sizes so I don't have to adapt the size every time when I add/remove a feature
    def __init__(self, hidden_size, seq_input_size, dropout=0.05, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        gru_output_size = hidden_size * self.num_directions

        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Linear(gru_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.out = nn.Softplus()


    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h = self.gru(packed)

        if self.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h_last = h[-1]

        raw = self.fc(h_last).squeeze(1)    # Sequential only

        return self.out(raw)

# -----------------
# Training
# -----------------
def train(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_items(args.data_dir, args.metadata_file, args.scanpath_dir)
    random.shuffle(items)

    if args.final_train:
        print(
            "FINAL TRAINING MODE: Model is trained on validation and test set. No validation metrics or early stopping are used during training.")
        train_items = items
        val_items = []
    else:
        split = int((1.0 - args.val_frac) * len(items))
        train_items = items[:split]
        val_items = items[split:]

        if not train_items or not val_items:
            raise ValueError("Train/validation split failed. Check data size and --val-frac.")

    # Sequence normalization
    seq_mean, seq_std = compute_seq_norm(train_items)

    train_dataset = ScanpathDataset(
        train_items,
        seq_mean=seq_mean,
        seq_std=seq_std
    )

    # Load datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    if not args.final_train:

        val_dataset = ScanpathDataset(
            val_items,
            seq_mean=seq_mean,
            seq_std=seq_std
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
    else:
        val_loader = None

    model = GRURegressor(
        hidden_size=args.hidden_size,
        seq_input_size=len(seq_mean),
        dropout=args.dropout,
        bidirectional=args.bidirectional,
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

        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x, lengths)
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
        "seq_input_size": int(len(seq_mean)),
        "seq_mean": torch.tensor(seq_mean, dtype=torch.float32),
        "seq_std": torch.tensor(seq_std, dtype=torch.float32),
        "huber_delta": args.huber_delta,
        "bidirectional": bool(args.bidirectional),
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
        "bidirectional": args.bidirectional,
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
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, total_sq_error, total_abs_error, total = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            # Prediction
            pred = model(x, lengths)

            # Loss
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * len(y)

            total_sq_error += float(torch.sum((pred - y) ** 2).item())
            total_abs_error += float(torch.sum(torch.abs(pred - y)).item())
            total += len(y)

    avg_loss = total_loss / max(total, 1)
    rmse = np.sqrt(total_sq_error / max(total, 1))
    mae = total_abs_error / max(total, 1)

    return avg_loss, rmse, mae



# -----------------
# Final testing
# -----------------
def test_on_labeled_set(test_dir="scanpaths/test",
                        labels_file="scanpaths/test_labels.csv",
                        model_file="model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    model = GRURegressor(
        hidden_size=int(checkpoint["hidden_size"]),
        seq_input_size=int(checkpoint["seq_input_size"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        bidirectional=bool(checkpoint.get("bidirectional", False)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    seq_mean = checkpoint["seq_mean"].cpu().numpy()
    seq_std = checkpoint["seq_std"].cpu().numpy()

    labels = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None,
        names=["reaction_time", "filename"],
        engine="python",
    )

    y_true = []
    y_pred = []
    rows = []

    with torch.no_grad():
        for _, row in labels.iterrows():
            path = os.path.join(test_dir, str(row["filename"]))
            if not os.path.exists(path):
                print(f"Missing file, skipped: {path}")
                continue

            true_rt = float(row["reaction_time"])

            seq = read_sequence(path)
            seq = (seq - seq_mean) / seq_std

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)

            pred_rt = model(x, lengths).item()

            y_true.append(true_rt)
            y_pred.append(pred_rt)

            rows.append({
                "filename": row["filename"],
                "true_rt": true_rt,
                "pred_rt": float(pred_rt),
                "abs_error": abs(pred_rt - true_rt),
                "sq_error": (pred_rt - true_rt) ** 2,
            })

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    print(f"Test samples: {len(y_true)}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "n": len(y_true),
        "predictions": pd.DataFrame(rows),
    }




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

