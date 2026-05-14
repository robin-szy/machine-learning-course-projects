"""
Final benchmark: trains all models from final_models/ with their own tuned
hyperparameters (from each file's parse_args defaults), using fixed seed=523
and stratified split for a fair comparison. Reports train/val/test RMSE and
the val-train gap (overfitting indicator).

Also computes persistence baseline.
"""

import os
import sys
import importlib.util
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model registry ─────────────────────────────────────────────────────────────
# Each entry: (file path relative to Traffic_Forecast/, class name, display label)
MODEL_REGISTRY = [
    ("final_models/GRU_GCN_hl64.py",        "GRU_GCN_v2",   "GRU-GCN h=64"),
    ("final_models/GRU_GCN_hl96.py",        "GRU_GCN_v2",   "GRU-GCN h=96"),
    ("final_models/CNN1D_GCN_hl32.py",      "Conv1D_GCN",   "CNN1D-GCN h=32"),
    ("final_models/CNN1D_GCN_hl64.py",      "Conv1D_GCN",   "CNN1D-GCN h=64"),
    ("final_models/CNN1D_GCN_hl160.py",     "Conv1D_GCN",   "CNN1D-GCN h=160"),
    ("final_models/CNN1D_AttentionGCN.py",  "Conv1D_GCN",   "CNN1D-AttentionGCN"),
    ("final_models/GCN_GRU.py",             "GCN_Conv1D",   "GCN-GRU"),
    ("final_models/GCN_MLP_h32.py",         "GCN_Conv1D",   "GCN-MLP h=32"),
    ("final_models/GCN_MLP_h64.py",         "GCN_Conv1D",   "GCN-MLP h=64"),
    ("final_models/pure_MLP_hl64.py",       "GCN_Conv1D",   "Pure MLP h=64"),
    ("final_models/pure_MLP_hl96.py",       "GCN_Conv1D",   "Pure MLP h=96"),
    ("final_models/Transformer_MLP.py",     "TemporalTransformer", "Transformer"),
]

# ── Shared data infrastructure ────────────────────────────────────────────────

class TrafficDataset(Dataset):
    def __init__(self, X, Y, continuous_idx, x_mean, x_std, y_mean, y_std):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.x_mean = x_mean
        self.x_std  = x_std
        self.y_mean = y_mean
        self.y_std  = y_std
        self.continuous_idx = continuous_idx

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx].copy()
        x[:, self.continuous_idx] = (x[:, self.continuous_idx] - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, loss_fn, y_mean, y_std):
    model.eval()
    total_loss = total_sq = total_abs = total = 0.0
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += float(loss_fn(pred, y).item()) * len(y)
            pred_real = (pred * y_std_t + y_mean_t).clamp(0.0, 1.0)
            y_real    = y * y_std_t + y_mean_t
            total_sq  += float(torch.sum((pred_real - y_real) ** 2).item())
            total_abs += float(torch.sum(torch.abs(pred_real - y_real)).item())
            total     += y_real.numel()
    rmse = np.sqrt(total_sq / max(total, 1))
    mae  = total_abs / max(total, 1)
    return total_loss / max(len(loader.dataset), 1), rmse, mae


def persistence_rmse(loader, device, y_mean, y_std):
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float32, device=device)
    total_sq = total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = x[:, :, 9]  # last observed step
            pred_real = pred * y_std_t + y_mean_t
            y_real    = y    * y_std_t + y_mean_t
            total_sq += torch.sum((pred_real - y_real) ** 2).item()
            total    += y_real.numel()
    return float(np.sqrt(total_sq / total))


def load_module(rel_path, module_name):
    abs_path = os.path.join(SCRIPT_DIR, rel_path)
    old_argv = sys.argv[:]
    sys.argv = ["benchmark"]
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def train_model(ModelClass, args, train_loader, val_loader, test_loader,
                adj_mask, static_size, n_sensors, y_mean, y_std, device):
    """Train one model, return (train_rmse, val_rmse, test_rmse, n_params, best_epoch)."""

    # Build model - handle optional transformer-specific args gracefully
    try:
        model = ModelClass(
            hidden_size=args.hidden_size,
            static_size=static_size,
            dropout=args.dropout,
            n_sensors=n_sensors,
            adj_mask=adj_mask,
            d_model=getattr(args, "d_model", 32),
            nhead=getattr(args, "nhead", 2),
            num_encoder_layers=getattr(args, "num_encoder_layers", 2),
        ).to(device)
    except TypeError:
        model = ModelClass(
            hidden_size=args.hidden_size,
            static_size=static_size,
            dropout=args.dropout,
            n_sensors=n_sensors,
            adj_mask=adj_mask,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    loss_fn = nn.HuberLoss(delta=getattr(args, "huber_delta", 1.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse  = float("inf")
    best_train_rmse = None
    best_state = None
    best_epoch = None
    no_improve = 0
    patience   = getattr(args, "patience", 30)
    min_delta  = getattr(args, "min_delta", 1e-4)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        _, val_rmse, _   = evaluate(model, val_loader,   device, loss_fn, y_mean, y_std)
        _, tr_rmse,  _   = evaluate(model, train_loader, device, loss_fn, y_mean, y_std)

        if val_rmse < best_val_rmse - min_delta:
            best_val_rmse   = val_rmse
            best_train_rmse = tr_rmse
            best_epoch      = epoch + 1
            best_state      = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    early stop epoch {epoch+1}  best_val={best_val_rmse:.4f}")
            break

    if epoch + 1 == args.epochs and no_improve < patience:
        print(f"    completed {args.epochs} epochs  best_val={best_val_rmse:.4f}")

    model.load_state_dict(best_state)
    _, test_rmse, test_mae = evaluate(model, test_loader, device, loss_fn, y_mean, y_std)

    return best_train_rmse, best_val_rmse, test_rmse, test_mae, n_params, best_epoch


def main():
    SEED      = 523
    VAL_FRAC  = 0.2
    BATCH     = 32

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    data_path = os.path.join(SCRIPT_DIR, "dataset.npz")
    data = np.load(data_path)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test,  Y_test  = data["X_test"],  data["Y_test"]
    adj_mat  = data["adj_mat"]
    n_sensors   = adj_mat.shape[0]
    static_size = X_train.shape[-1] - 10

    # Stratified split (fixed)
    y_level = Y_train.mean(axis=1)
    y_bin   = pd.qcut(y_level, q=4, labels=False, duplicates="drop")
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=VAL_FRAC,
        random_state=SEED, shuffle=True, stratify=y_bin)

    X_tr, Y_tr   = X_train[train_idx], Y_train[train_idx]
    X_val, Y_val = X_train[val_idx],   Y_train[val_idx]

    continuous_idx = list(range(10)) + [39]
    x_mean = X_tr[:, :, continuous_idx].mean(axis=(0, 1))
    x_std  = X_tr[:, :, continuous_idx].std(axis=(0, 1))  + 1e-6
    y_mean = float(Y_tr.mean())
    y_std  = float(Y_tr.std()) + 1e-6

    def make_loader(X, Y, shuffle):
        ds = TrafficDataset(X, Y, continuous_idx, x_mean, x_std, y_mean, y_std)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    train_loader = make_loader(X_tr,    Y_tr,    shuffle=True)
    val_loader   = make_loader(X_val,   Y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  Y_test,  shuffle=False)

    adj_mask = torch.tensor((adj_mat > 0.5).astype(np.float32))

    # Persistence baseline
    base_rmse = persistence_rmse(val_loader, device, y_mean, y_std)
    print(f"Persistence baseline val RMSE: {base_rmse:.4f}\n")

    results = []

    for rel_path, class_name, label in MODEL_REGISTRY:
        print(f"{'='*60}")
        print(f"Training: {label}  ({class_name} from {rel_path})")

        mod_name = label.replace(" ", "_").replace("=", "").replace("-", "_")
        try:
            mod = load_module(rel_path, mod_name)
            args = mod.parse_args()
            ModelClass = getattr(mod, class_name)
        except Exception as exc:
            print(f"  LOAD ERROR: {exc}")
            results.append({
                "model": label, "train_rmse": None, "val_rmse": None,
                "test_rmse": None, "gap": None, "params": None, "best_epoch": None,
                "error": str(exc),
            })
            continue

        set_seed(SEED)
        try:
            tr_rmse, val_rmse, test_rmse, test_mae, n_params, best_epoch = train_model(
                ModelClass, args, train_loader, val_loader, test_loader,
                adj_mask, static_size, n_sensors, y_mean, y_std, device)
        except Exception as exc:
            print(f"  TRAIN ERROR: {exc}")
            import traceback; traceback.print_exc()
            results.append({
                "model": label, "train_rmse": None, "val_rmse": None,
                "test_rmse": None, "gap": None, "params": None, "best_epoch": None,
                "error": str(exc),
            })
            continue

        gap = val_rmse - tr_rmse
        results.append({
            "model": label, "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(val_rmse, 4), "test_rmse": round(test_rmse, 4),
            "test_mae": round(test_mae, 4),
            "gap": round(gap, 4), "params": n_params, "best_epoch": best_epoch,
            "error": None,
        })
        print(f"  train={tr_rmse:.4f}  val={val_rmse:.4f}  test={test_rmse:.4f}"
              f"  gap={gap:+.4f}  params={n_params:,}  best_epoch={best_epoch}")

    # Summary table
    df = pd.DataFrame(results)
    df = df.sort_values("test_rmse", na_position="last").reset_index(drop=True)

    print(f"\n{'='*80}")
    print("FINAL BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<28} {'Train':>7} {'Val':>7} {'Test':>7} {'Gap':>7} {'Params':>10} {'BestEp':>7}")
    print("-" * 80)
    for _, row in df.iterrows():
        if row["test_rmse"] is None:
            print(f"  {row['model']:<26} ERROR: {row['error']}")
        else:
            print(f"  {row['model']:<26} {row['train_rmse']:>7.4f} {row['val_rmse']:>7.4f}"
                  f" {row['test_rmse']:>7.4f} {row['gap']:>+7.4f}"
                  f" {int(row['params']):>10,} {int(row['best_epoch']):>7}")

    print(f"\n  Persistence baseline (val): {base_rmse:.4f}")

    out_csv = os.path.join(SCRIPT_DIR, "runs/final_benchmark.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
