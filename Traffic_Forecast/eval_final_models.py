"""
Evaluate all pre-trained final models (trained on full train set) on the test set.
Each .pth checkpoint stores normalization stats and model config.
"""

import os, sys, importlib.util
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(SCRIPT_DIR, "final_models/final_model_files")
FINAL_MODELS = os.path.join(SCRIPT_DIR, "final_models")

# Map model_version string (in checkpoint) -> (py file, class name)
VERSION_MAP = {
    "gcn_mlp_hl32":       ("GCN_MLP_h32.py",        "GCN_Conv1D"),
    "gcn_mlp_hl64":       ("GCN_MLP_h64.py",        "GCN_Conv1D"),
    "pure_mlp_hl96":      ("pure_MLP_hl96.py",       "GCN_Conv1D"),
    "pure_mlp_hl64":      ("pure_MLP_hl64.py",       "GCN_Conv1D"),
    "conv1d_gcn_hl32":    ("CNN1D_GCN_hl32.py",      "Conv1D_GCN"),
    "conv1d_gcn_attent":  ("CNN1D_AttentionGCN.py",  "Conv1D_GCN"),
    "conv1d_gcn_hl64":    ("CNN1D_GCN_hl64.py",      "Conv1D_GCN"),
    "gru_gcn_hl96":       ("GRU_GCN_hl96.py",        "GRU_GCN_v2"),
    "gru_gcn_hl64":       ("GRU_GCN_hl64.py",        "GRU_GCN_v2"),
    "conv1d_gcn_hl160":   ("CNN1D_GCN_hl160.py",     "Conv1D_GCN"),
    "gcn_gru":            ("GCN_GRU.py",             "GCN_Conv1D"),
    "transformer_mlp":    ("Transformer_MLP.py",     "TemporalTransformer"),
}

DISPLAY_NAMES = {
    "gcn_mlp_hl32":      "GCN-MLP h=32",
    "gcn_mlp_hl64":      "GCN-MLP h=64",
    "pure_mlp_hl96":     "Pure MLP h=96",
    "pure_mlp_hl64":     "Pure MLP h=64",
    "conv1d_gcn_hl32":   "CNN1D-GCN h=32",
    "conv1d_gcn_attent": "CNN1D-AttentionGCN",
    "conv1d_gcn_hl64":   "CNN1D-GCN h=64",
    "gru_gcn_hl96":      "GRU-GCN h=96",
    "gru_gcn_hl64":      "GRU-GCN h=64",
    "conv1d_gcn_hl160":  "CNN1D-GCN h=160",
    "gcn_gru":           "GCN-GRU",
    "transformer_mlp":   "Transformer",
}


class TestDataset(Dataset):
    def __init__(self, X, Y, continuous_idx, x_mean, x_std, y_mean, y_std):
        self.X, self.Y = X.astype(np.float32), Y.astype(np.float32)
        self.continuous_idx = continuous_idx
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy(); y = self.Y[idx].copy()
        x[:, self.continuous_idx] = (x[:, self.continuous_idx] - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_module(py_file, mod_name):
    old_argv = sys.argv[:]
    sys.argv = ["eval"]
    path = os.path.join(FINAL_MODELS, py_file)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def eval_checkpoint(pth_path, X_test, Y_test, device):
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    version = ckpt["model_version"]

    if version not in VERSION_MAP:
        return None, f"Unknown version '{version}'"

    py_file, class_name = VERSION_MAP[version]
    mod_name = version.replace("-", "_")
    mod = load_module(py_file, mod_name)
    ModelClass = getattr(mod, class_name)

    adj_mask    = ckpt["adj_mask"].to(device)
    hidden_size = int(ckpt["hidden_size"])
    dropout     = float(ckpt["dropout"])
    n_sensors   = int(ckpt["n_sensors"])
    static_size = X_test.shape[-1] - 10

    try:
        model = ModelClass(
            hidden_size=hidden_size, static_size=static_size,
            dropout=dropout, n_sensors=n_sensors, adj_mask=adj_mask,
            d_model=int(ckpt.get("d_model", 32)),
            nhead=int(ckpt.get("nhead", 2)),
            num_encoder_layers=int(ckpt.get("num_encoder_layers", 2)),
        ).to(device)
    except TypeError:
        model = ModelClass(
            hidden_size=hidden_size, static_size=static_size,
            dropout=dropout, n_sensors=n_sensors, adj_mask=adj_mask,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    continuous_idx = list(range(10)) + [39]
    x_mean = ckpt["x_mean"].cpu().numpy()
    x_std  = ckpt["x_std"].cpu().numpy()
    y_mean = float(ckpt["y_mean"])
    y_std  = float(ckpt["y_std"])

    ds     = TestDataset(X_test, Y_test, continuous_idx, x_mean, x_std, y_mean, y_std)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    y_mean_t = torch.tensor(y_mean, device=device)
    y_std_t  = torch.tensor(y_std,  device=device)

    total_sq = total_abs = total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred_real = (model(x) * y_std_t + y_mean_t).clamp(0.0, 1.0)
            y_real    = y * y_std_t + y_mean_t
            total_sq  += float(torch.sum((pred_real - y_real) ** 2).item())
            total_abs += float(torch.sum(torch.abs(pred_real - y_real)).item())
            total     += y_real.numel()

    rmse = float(np.sqrt(total_sq / total))
    mae  = float(total_abs / total)
    n_params = sum(p.numel() for p in model.parameters())
    return {"rmse": rmse, "mae": mae, "params": n_params, "version": version}, None


def persistence_rmse(X_test, Y_test):
    last_obs = X_test[:, :, 9]
    return float(np.sqrt(np.mean((last_obs - Y_test) ** 2)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data    = np.load(os.path.join(SCRIPT_DIR, "dataset.npz"))
    X_test  = data["X_test"]
    Y_test  = data["Y_test"]

    p_rmse = persistence_rmse(X_test, Y_test)
    print(f"Persistence baseline test RMSE: {p_rmse:.4f}\n")

    pth_files = sorted(
        f for f in os.listdir(MODELS_DIR) if f.endswith(".pth"))

    # Also include transformer if trained
    transformer_pth = os.path.join(SCRIPT_DIR, "model_transformer.pth")
    extra = []
    if os.path.exists(transformer_pth):
        extra.append(("model_transformer.pth", transformer_pth))

    results = []
    for fname in pth_files:
        path = os.path.join(MODELS_DIR, fname)
        print(f"Evaluating {fname} ...")
        metrics, err = eval_checkpoint(path, X_test, Y_test, device)
        if err:
            print(f"  ERROR: {err}")
            continue
        label = DISPLAY_NAMES.get(metrics["version"], metrics["version"])
        print(f"  {label:<28}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}"
              f"  params={metrics['params']:,}")
        results.append({"model": label, "rmse": round(metrics["rmse"], 4),
                        "mae": round(metrics["mae"], 4), "params": metrics["params"],
                        "pth": fname})

    for fname, path in extra:
        print(f"Evaluating {fname} ...")
        metrics, err = eval_checkpoint(path, X_test, Y_test, device)
        if err:
            print(f"  ERROR: {err}")
            continue
        label = DISPLAY_NAMES.get(metrics["version"], metrics["version"])
        print(f"  {label:<28}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}"
              f"  params={metrics['params']:,}")
        results.append({"model": label, "rmse": round(metrics["rmse"], 4),
                        "mae": round(metrics["mae"], 4), "params": metrics["params"],
                        "pth": fname})

    df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    print(f"\n{'='*65}")
    print("FINAL MODEL TEST SET RESULTS  (trained on full train set)")
    print(f"{'='*65}")
    print(f"{'Model':<28} {'Test RMSE':>10} {'MAE':>8} {'Params':>10}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"  {row['model']:<26} {row['rmse']:>10.4f} {row['mae']:>8.4f} {int(row['params']):>10,}")
    print(f"\n  Persistence baseline:       {p_rmse:>10.4f}")

    out = os.path.join(SCRIPT_DIR, "runs/final_model_test_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    return df, p_rmse


if __name__ == "__main__":
    main()
