"""
Benchmark all models with a consistent train/val split + test evaluation.
Captures train RMSE, val RMSE, and test RMSE so we can assess overfitting.
"""

import os, sys, random, copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

# ── shared config ─────────────────────────────────────────────────────────────
SEED       = 523
VAL_FRAC   = 0.2
EPOCHS     = 150
PATIENCE   = 30
MIN_DELTA  = 1e-4
BATCH      = 32
DATA_DIR   = "."
NPZ_FILE   = "dataset.npz"
MAT_FILE   = "traffic_dataset.mat"

# ── data utilities ────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_dataset():
    npz = os.path.join(DATA_DIR, NPZ_FILE)
    if not os.path.exists(npz):
        mat = loadmat(os.path.join(DATA_DIR, MAT_FILE))
        X_train = np.array([m.toarray() for m in mat['tra_X_tr'][0]], dtype=np.float32)
        X_test  = np.array([m.toarray() for m in mat['tra_X_te'][0]], dtype=np.float32)
        Y_train = mat["tra_Y_tr"].T.astype(np.float32)
        Y_test  = mat["tra_Y_te"].T.astype(np.float32)
        adj_mat = mat['tra_adj_mat']
        np.savez(npz, X_train=X_train, X_test=X_test,
                 Y_train=Y_train, Y_test=Y_test, adj_mat=adj_mat)
    d = np.load(npz)
    return d["X_train"], d["Y_train"], d["X_test"], d["Y_test"], d["adj_mat"]

class TrafficDS(Dataset):
    def __init__(self, X, Y, cont_idx, x_mean, x_std, y_mean, y_std):
        self.X, self.Y = X.astype(np.float32), Y.astype(np.float32)
        self.cont_idx = cont_idx
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x, y = self.X[i].copy(), self.Y[i].copy()
        x[:, self.cont_idx] = (x[:, self.cont_idx] - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def make_loaders(X_tr, Y_tr, X_val, Y_val, cont_idx, x_mean, x_std, y_mean, y_std):
    tr = DataLoader(TrafficDS(X_tr, Y_tr, cont_idx, x_mean, x_std, y_mean, y_std),
                    batch_size=BATCH, shuffle=True)
    va = DataLoader(TrafficDS(X_val, Y_val, cont_idx, x_mean, x_std, y_mean, y_std),
                    batch_size=BATCH, shuffle=False)
    return tr, va

def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    loss_fn = nn.HuberLoss(delta=1.0)
    total_sq = total_abs = total = total_loss = 0.0
    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.tensor(y_std,  dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            total_loss += loss_fn(p, y).item() * len(y)
            p_real = (p * ys + ym).clamp(0, 1)
            y_real = y * ys + ym
            total_sq  += (p_real - y_real).pow(2).sum().item()
            total_abs += (p_real - y_real).abs().sum().item()
            total     += y_real.numel()
    return (np.sqrt(total_sq / max(total, 1)),
            total_abs / max(total, 1),
            total_loss / max(len(loader.dataset), 1))

def persistence_baseline(loader, device, y_mean, y_std):
    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.tensor(y_std,  dtype=torch.float32, device=device)
    total_sq = total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p_real = (x[:, :, 9] * ys + ym).clamp(0, 1)
            y_real = y * ys + ym
            total_sq += (p_real - y_real).pow(2).sum().item()
            total    += y_real.numel()
    return np.sqrt(total_sq / max(total, 1))

def train_model(model, train_loader, val_loader, device, y_mean, y_std,
                lr=0.001, weight_decay=0.001, label="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)
    best_val, best_state, no_imp = float("inf"), None, 0
    best_train_rmse = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        val_rmse, val_mae, _ = evaluate(model, val_loader, device, y_mean, y_std)
        train_rmse, _, _     = evaluate(model, train_loader, device, y_mean, y_std)

        if val_rmse < best_val - MIN_DELTA:
            best_val = val_rmse
            best_train_rmse = train_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 20 == 0 or no_imp == 0:
            print(f"  [{label}] ep{epoch:03d} train={train_rmse:.4f} val={val_rmse:.4f} "
                  f"best={best_val:.4f}")

        if no_imp >= PATIENCE:
            print(f"  [{label}] early stop ep{epoch}  best_val={best_val:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_val, best_train_rmse


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

def normalize_adj(adj):
    adj = adj.astype(np.float32) + np.eye(adj.shape[0], dtype=np.float32)
    return adj / adj.sum(1, keepdims=True).clip(1e-6)

class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_size, out_size, n, adj_mask):
        super().__init__()
        self.register_buffer("adj_mask", adj_mask.float())
        self.edge_logits = nn.Parameter(torch.zeros(n, n))
        self.linear = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(out_size)
    def get_adj(self):
        w = torch.sigmoid(self.edge_logits) * self.adj_mask
        w = w + torch.eye(w.shape[0], device=w.device)
        return w / w.sum(1, keepdim=True).clamp(1e-6)
    def forward(self, z):
        return self.norm(torch.relu(self.linear(
            torch.einsum("ij,bjf->bif", self.get_adj(), z))))

class GraphAttentionConv(nn.Module):
    def __init__(self, in_size, out_size, n, adj_mask, dropout=0.1):
        super().__init__()
        self.register_buffer("adj_mask", adj_mask.float())
        self.linear    = nn.Linear(in_size, out_size, bias=False)
        self.attn_src  = nn.Linear(out_size, 1, bias=False)
        self.attn_dst  = nn.Linear(out_size, 1, bias=False)
        self.leaky     = nn.LeakyReLU(0.2)
        self.norm      = nn.LayerNorm(out_size)
    def forward(self, z):
        h = self.linear(z)
        e = self.leaky(self.attn_src(h) + self.attn_dst(h).transpose(1, 2))
        mask = (self.adj_mask + torch.eye(self.adj_mask.shape[0], device=z.device)).clamp(max=1)
        e = e.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
        return self.norm(torch.relu(torch.bmm(torch.softmax(e, dim=-1), h)))

# ── 1. GRU-GCN V1 (original, Alex) ───────────────────────────────────────────
class GRU_GCN_V1(nn.Module):
    def __init__(self, hidden=64, static_size=38, dropout=0.1, n=36, adj=None):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.static_MLP = nn.Sequential(nn.Linear(static_size,32), nn.ReLU(), nn.Dropout(dropout))
        self.combine    = nn.Sequential(nn.Linear(hidden+32, hidden), nn.ReLU())
        adj_n = torch.tensor(normalize_adj(adj), dtype=torch.float32) if adj is not None else torch.eye(n)
        self.register_buffer("adj_norm", adj_n)
        self.gcn        = nn.Linear(hidden, hidden)
        self.head       = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        B, N, T = x.shape[0], x.shape[1], 10
        seq = x[:,:,:10].reshape(B*N,T,1)
        _, h = self.gru(seq); h = h[-1].reshape(B,N,-1)
        s = self.static_MLP(x[:,:,10:])
        z = self.combine(torch.cat([h,s],dim=-1))
        z_agg = torch.einsum("ij,bjf->bif", self.adj_norm, z)
        z = torch.relu(self.gcn(z_agg))
        return self.head(z).squeeze(-1)

# ── 2. GRU-GCN V2 (Alex, learnable adj + embedding) ─────────────────────────
class GRU_GCN_V2(nn.Module):
    def __init__(self, hidden=32, static_size=38, dropout=0.1, n=36, adj_mask=None):
        super().__init__()
        emb = 8
        self.sensor_emb = nn.Embedding(n, emb)
        self.gru        = nn.GRU(1, hidden, batch_first=True)
        self.static_mlp = nn.Sequential(nn.Linear(static_size,32), nn.ReLU(), nn.Dropout(dropout))
        self.combine    = nn.Sequential(nn.Linear(hidden+32+emb, hidden), nn.ReLU())
        if adj_mask is None: adj_mask = torch.ones(n,n)
        self.gcn1 = AdaptiveGraphConv(hidden,hidden,n,adj_mask)
        self.gcn2 = AdaptiveGraphConv(hidden,hidden,n,adj_mask)
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x, adj=None):
        B,N = x.shape[0], x.shape[1]
        seq_flat = x[:,:,:10].reshape(B*N,10,1)
        _, h = self.gru(seq_flat); h = h[-1].reshape(B,N,-1)
        s = self.static_mlp(x[:,:,10:])
        e = self.sensor_emb(torch.arange(N,device=x.device)).unsqueeze(0).expand(B,-1,-1)
        z = self.combine(torch.cat([h,s,e],dim=-1))
        z_out = self.gcn2(self.gcn1(z)) + z
        return self.head(z_out).squeeze(-1)

# ── 3. Pure MLP + embedding (Robin, GCN commented out) ───────────────────────
class PureMLP(nn.Module):
    def __init__(self, hidden=32, static_size=38, dropout=0.1, n=36, adj_mask=None):
        super().__init__()
        emb = 8
        self.sensor_emb = nn.Embedding(n, emb)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU())
        self.static_mlp = nn.Sequential(nn.Linear(static_size,32), nn.ReLU(), nn.Dropout(dropout))
        self.combine    = nn.Sequential(nn.Linear(10+32+emb, hidden), nn.ReLU())
        self.post_mlp   = nn.Sequential(nn.Linear(hidden,hidden), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(hidden,hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        B,N = x.shape[0], x.shape[1]
        s = self.static_mlp(x[:,:,10:])
        e = self.sensor_emb(torch.arange(N,device=x.device)).unsqueeze(0).expand(B,-1,-1)
        z = self.combine(torch.cat([x[:,:,:10], s, e], dim=-1))
        h = self.post_mlp(z)
        return self.head(h + z).squeeze(-1)

# ── 4. CNN1D → GCN (Robin's best tuned: hidden=160) ──────────────────────────
class Conv1D_GCN(nn.Module):
    def __init__(self, hidden=160, static_size=38, dropout=0.15, n=36, adj_mask=None):
        super().__init__()
        emb = 8
        self.sensor_emb = nn.Embedding(n, emb)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU())
        self.static_mlp = nn.Sequential(nn.Linear(static_size,32), nn.ReLU(), nn.Dropout(dropout))
        self.combine    = nn.Sequential(nn.Linear(hidden+32+emb, hidden), nn.ReLU())
        if adj_mask is None: adj_mask = torch.ones(n,n)
        self.gcn1 = AdaptiveGraphConv(hidden,hidden,n,adj_mask)
        self.gcn2 = AdaptiveGraphConv(hidden,hidden,n,adj_mask)
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        B,N,T = x.shape[0], x.shape[1], 10
        h = self.temporal_conv(x[:,:,:T].reshape(B*N,1,T)).mean(-1).reshape(B,N,-1)
        s = self.static_mlp(x[:,:,10:])
        e = self.sensor_emb(torch.arange(N,device=x.device)).unsqueeze(0).expand(B,-1,-1)
        z = self.combine(torch.cat([h,s,e],dim=-1))
        z_out = self.gcn2(self.gcn1(z)) + z
        return self.head(z_out).squeeze(-1)

# ── 5. GCN per-timestep → Conv1D (Robin, "slow") ─────────────────────────────
class GCN_GRU(nn.Module):
    def __init__(self, hidden=32, static_size=38, dropout=0.1, n=36, adj_mask=None):
        super().__init__()
        emb = 8
        self.sensor_emb = nn.Embedding(n, emb)
        self.static_mlp = nn.Sequential(nn.Linear(static_size,32), nn.ReLU(), nn.Dropout(dropout))
        self.input_proj = nn.Sequential(nn.Linear(1+32+emb, hidden), nn.ReLU())
        if adj_mask is None: adj_mask = torch.ones(n,n)
        self.gcn1 = AdaptiveGraphConv(hidden,hidden,n,adj_mask)
        self.gcn2 = AdaptiveGraphConv(hidden,hidden,n,adj_mask)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden,hidden,3,padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden,hidden,3,padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        B,N,T = x.shape[0], x.shape[1], 10
        s = self.static_mlp(x[:,:,10:])
        e = self.sensor_emb(torch.arange(N,device=x.device)).unsqueeze(0).expand(B,-1,-1)
        outs = []
        for t in range(T):
            z_t = self.input_proj(torch.cat([x[:,:,t:t+1], s, e], dim=-1))
            outs.append(self.gcn2(self.gcn1(z_t)) + z_t)
        h = torch.stack(outs,2).reshape(B*N,T,-1).transpose(1,2)
        h = self.temporal_conv(h).mean(-1).reshape(B,N,-1)
        return self.head(h).squeeze(-1)

# ── 6. CNN1D + Graph Attention (Robin) ───────────────────────────────────────
class Conv1D_AttentionGCN(nn.Module):
    def __init__(self, hidden=32, static_size=38, dropout=0.1, n=36, adj_mask=None):
        super().__init__()
        emb = 8
        self.sensor_emb = nn.Embedding(n, emb)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU())
        self.static_mlp = nn.Sequential(nn.Linear(static_size,32), nn.ReLU(), nn.Dropout(dropout))
        self.combine    = nn.Sequential(nn.Linear(hidden+32+emb, hidden), nn.ReLU())
        if adj_mask is None: adj_mask = torch.ones(n,n)
        self.gcn1 = GraphAttentionConv(hidden,hidden,n,adj_mask,dropout)
        self.gcn2 = GraphAttentionConv(hidden,hidden,n,adj_mask,dropout)
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        B,N,T = x.shape[0], x.shape[1], 10
        h = self.temporal_conv(x[:,:,:T].reshape(B*N,1,T)).mean(-1).reshape(B,N,-1)
        s = self.static_mlp(x[:,:,10:])
        e = self.sensor_emb(torch.arange(N,device=x.device)).unsqueeze(0).expand(B,-1,-1)
        z = self.combine(torch.cat([h,s,e],dim=-1))
        z_out = self.gcn2(self.gcn1(z)) + z
        return self.head(z_out).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    device = torch.device("cpu")

    X_train, Y_train, X_test, Y_test, adj_mat = load_dataset()
    print(f"Dataset: X_train={X_train.shape}  X_test={X_test.shape}")

    # fixed split (same for all models)
    n = len(X_train)
    idx = np.random.permutation(n)
    split = int((1 - VAL_FRAC) * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
    X_va, Y_va = X_train[va_idx], Y_train[va_idx]

    cont_idx = list(range(10)) + [39]
    x_mean = X_tr[:,:,cont_idx].mean(axis=(0,1))
    x_std  = X_tr[:,:,cont_idx].std(axis=(0,1)) + 1e-6
    y_mean = float(Y_tr.mean())
    y_std  = float(Y_tr.std()) + 1e-6

    train_loader, val_loader = make_loaders(X_tr, Y_tr, X_va, Y_va,
                                            cont_idx, x_mean, x_std, y_mean, y_std)
    test_loader = DataLoader(
        TrafficDS(X_test, Y_test, cont_idx, x_mean, x_std, y_mean, y_std),
        batch_size=BATCH, shuffle=False)

    n_sensors   = adj_mat.shape[0]
    static_size = X_train.shape[-1] - 10
    adj_mask    = torch.tensor((adj_mat > 0.5).astype(np.float32))
    adj_np      = adj_mat

    baseline_rmse = persistence_baseline(test_loader, device, y_mean, y_std)
    print(f"\nPersistence baseline (test) RMSE = {baseline_rmse:.4f}\n")

    configs = [
        ("V1 GRU-GCN (fixed adj)",
         GRU_GCN_V1(hidden=64, static_size=static_size, dropout=0.1,
                    n=n_sensors, adj=adj_np),
         0.001, 0.001),
        ("V2 GRU-GCN (learnable adj+emb)",
         GRU_GCN_V2(hidden=32, static_size=static_size, dropout=0.1,
                    n=n_sensors, adj_mask=adj_mask),
         0.001, 0.001),
        ("Pure MLP + emb (no GCN)",
         PureMLP(hidden=32, static_size=static_size, dropout=0.1,
                 n=n_sensors),
         0.001, 0.001),
        ("CNN1D-GCN (learnable adj, h=160)",
         Conv1D_GCN(hidden=160, static_size=static_size, dropout=0.15,
                    n=n_sensors, adj_mask=adj_mask),
         0.001, 0.001),
        ("GCN-per-step then Conv1D (slow)",
         GCN_GRU(hidden=32, static_size=static_size, dropout=0.1,
                 n=n_sensors, adj_mask=adj_mask),
         0.001, 0.001),
        ("CNN1D + Graph Attention",
         Conv1D_AttentionGCN(hidden=32, static_size=static_size, dropout=0.1,
                             n=n_sensors, adj_mask=adj_mask),
         0.001, 0.001),
    ]

    results = []
    for name, model, lr, wd in configs:
        set_seed(SEED)
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Training: {name}  ({params:,} params)")
        print(f"{'='*60}")
        best_val, best_train = train_model(model, train_loader, val_loader,
                                           device, y_mean, y_std, lr, wd, name)
        test_rmse, test_mae, _ = evaluate(model, test_loader, device, y_mean, y_std)
        gap = best_val - best_train
        print(f"  RESULT  train={best_train:.4f}  val={best_val:.4f}  "
              f"test={test_rmse:.4f}  gap={gap:+.4f}")
        results.append((name, params, best_train, best_val, test_rmse, test_mae, gap))

    print("\n\n" + "="*80)
    print("FULL BENCHMARK RESULTS")
    print("="*80)
    hdr = f"{'Model':<40} {'Params':>8}  {'Train':>7}  {'Val':>7}  {'Test':>7}  {'Gap(V-T)':>9}  {'Overfit?'}"
    print(hdr)
    print("-"*80)
    for name, params, tr, va, te, _, gap in results:
        flag = " *** OVERFIT" if gap > 0.010 else ""
        print(f"{name:<40} {params:>8,}  {tr:.4f}   {va:.4f}   {te:.4f}   {gap:+.4f}  {flag}")
    print(f"\n{'Persistence baseline':<40} {'—':>8}  {'—':>7}  {'—':>7}  {baseline_rmse:.4f}")
    print("="*80)
    print("\nNote: all RMSE values are in normalised [0,1] traffic-flow units.")
    print("Gap = val_rmse - train_rmse at best-val checkpoint.")
    print("Flag '*** OVERFIT' when gap > 0.010 (val more than 1 pp above train).")


if __name__ == "__main__":
    main()
