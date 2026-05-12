"""
Generate learned_dependency_map.html  — replicates the style of Fig. 9 in the
SADL paper: sensors as numbered circles, edges coloured and weighted by the
learned spatial-dependency strength from our trained GRU-GCN-v2 model.
"""

import numpy as np
import torch
from torch import nn
from scipy.io import loadmat
import folium
from folium.plugins import FloatImage
import branca.colormap as bcm

# ── Load dataset ──────────────────────────────────────────────────────────────
data = np.load("dataset.npz")
adj_mat = data["adj_mat"]
n_sensors = adj_mat.shape[0]
adj_mask  = torch.tensor((adj_mat > 0.5).astype(np.float32))

# ── Sensor coordinates (from build_visualization_nb.py) ───────────────────────
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

ROAD_LABELS = ['I-95 (main)', 'I-66', 'I-495 NW connector', 'I-495 SW connector']
# road_id per sensor (from static features in dataset)
mat = loadmat("traffic_dataset.mat")
feats   = np.array([m.toarray() for m in mat['tra_X_tr'][0]], dtype=np.float32)[0]
road_id = np.argmax(feats[:, 44:48], axis=1)

# ── Load trained model and extract learned adjacency ─────────────────────────
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_size, out_size, n, adj_mask):
        super().__init__()
        self.register_buffer("adj_mask", adj_mask.float())
        self.edge_logits = nn.Parameter(torch.zeros(n, n))
        self.linear = nn.Linear(in_size, out_size)
        self.norm   = nn.LayerNorm(out_size)
    def get_adj(self):
        w = torch.sigmoid(self.edge_logits) * self.adj_mask
        w = w + torch.eye(w.shape[0], device=w.device)
        return w / w.sum(1, keepdim=True).clamp(1e-6)
    def forward(self, z):
        return self.norm(torch.relu(self.linear(
            torch.einsum("ij,bjf->bif", self.get_adj(), z))))

class GRU_GCN_V2(nn.Module):
    def __init__(self, hidden=32, static_size=38, dropout=0.1, n=36, adj_mask=None):
        super().__init__()
        self.gru        = nn.GRU(1, hidden, batch_first=True)
        self.static_mlp = nn.Sequential(nn.Linear(static_size, 32), nn.ReLU(), nn.Dropout(dropout))
        self.combine    = nn.Sequential(nn.Linear(hidden+32, hidden), nn.ReLU())
        if adj_mask is None: adj_mask = torch.ones(n, n)
        self.gcn1 = AdaptiveGraphConv(hidden, hidden, n, adj_mask)
        self.gcn2 = AdaptiveGraphConv(hidden, hidden, n, adj_mask)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x, adj=None):
        B, N = x.shape[0], x.shape[1]
        _, h = self.gru(x[:,:,:10].reshape(B*N,10,1))
        h = h[-1].reshape(B, N, -1)
        s = self.static_mlp(x[:,:,10:])
        z = self.combine(torch.cat([h,s], dim=-1))
        return self.head(self.gcn2(self.gcn1(z)) + z).squeeze(-1)

ckpt   = torch.load("model_v2.pth", map_location="cpu", weights_only=False)
model  = GRU_GCN_V2(
    hidden=int(ckpt["hidden_size"]),
    static_size=feats.shape[-1] - 10,
    dropout=float(ckpt["dropout"]),
    n=n_sensors,
    adj_mask=adj_mask,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Raw learned weights (before row-normalisation) — symmetric enough to plot
with torch.no_grad():
    raw_w = torch.sigmoid(model.gcn1.edge_logits) * adj_mask
    raw_w.fill_diagonal_(0)                         # hide self-loops
    learned_adj = raw_w.numpy()

# Normalise to [0,1] for colour mapping
w_max = learned_adj.max()
w_min = learned_adj[learned_adj > 0].min() if (learned_adj > 0).any() else 0
learned_norm = np.where(learned_adj > 0,
                        (learned_adj - w_min) / max(w_max - w_min, 1e-6), 0)

print(f"Learned adj  min={w_min:.4f}  max={w_max:.4f}")
print(f"Edges present (topology): {int((adj_mask.numpy() > 0).sum())}")
print(f"Edges with weight>0.05 (normalised): {int((learned_norm > 0.05).sum())}")

# ── Build Folium map ──────────────────────────────────────────────────────────
m = folium.Map(
    location=[38.85, -77.20],
    zoom_start=11,
    tiles='CartoDB positron',
)

# ── Colour scale matching paper (light grey → orange → red) ──────────────────
def weight_to_color(w_norm):
    """Map normalised weight [0,1] to hex colour."""
    # Paper uses: weak = thin grey, strong = thick red
    r = int(180 + 75 * w_norm)          # 180 → 255
    g = int(180 - 160 * w_norm)         # 180 → 20
    b = int(180 - 170 * w_norm)         # 180 → 10
    return f"#{r:02x}{g:02x}{b:02x}"

def weight_to_width(w_norm):
    return max(0.8, w_norm * 8)

# ── Draw edges ────────────────────────────────────────────────────────────────
# Sort weakest first so strong edges render on top
edges = []
for i in range(n_sensors):
    for j in range(n_sensors):
        if i != j and learned_adj[i, j] > 0:
            edges.append((i, j, learned_norm[i, j]))

edges.sort(key=lambda x: x[2])   # weakest first → drawn under strong

for i, j, wn in edges:
    wn = float(wn)
    if wn < 0.02:   # skip near-zero
        continue
    folium.PolyLine(
        locations=[[float(lat_arr[i]), float(lon_arr[i])],
                   [float(lat_arr[j]), float(lon_arr[j])]],
        weight=weight_to_width(wn),
        color=weight_to_color(wn),
        opacity=round(0.55 + 0.4 * wn, 3),
        tooltip=f"S{i}->S{j}  strength={float(learned_adj[i,j]):.3f}",
    ).add_to(m)

# ── Road colour palette (matching paper circles) ──────────────────────────────
ROAD_CIRCLE_COLOR = {
    0: "#c0392b",   # I-95 main   → dark red
    1: "#2980b9",   # I-66        → blue
    2: "#27ae60",   # I-495 NW    → green
    3: "#e67e22",   # I-495 SW    → orange
}

# ── Sensor nodes ──────────────────────────────────────────────────────────────
for i in range(n_sensors):
    ring_color = ROAD_CIRCLE_COLOR.get(int(road_id[i]), "#555555")

    # Outer coloured ring
    folium.CircleMarker(
        location=[float(lat_arr[i]), float(lon_arr[i])],
        radius=12,
        color=ring_color,
        weight=2.5,
        fill=True,
        fill_color="white",
        fill_opacity=0.92,
        tooltip=f"Sensor {i} - {ROAD_LABELS[int(road_id[i])]}",
    ).add_to(m)

    # Sensor number label
    folium.Marker(
        location=[float(lat_arr[i]), float(lon_arr[i])],
        icon=folium.DivIcon(
            html=(
                f'<div style="'
                f'font-size:9px;font-weight:bold;color:#111111;'
                f'text-align:center;line-height:24px;'
                f'width:24px;height:24px;'
                f'">{i}</div>'
            ),
            icon_size=(24, 24),
            icon_anchor=(12, 12),
        ),
    ).add_to(m)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_html = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 9999;
    background: white; border: 1px solid #ccc; border-radius: 6px;
    padding: 12px 16px; font-family: Arial, sans-serif; font-size: 12px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
  <b>Learned spatial dependency</b><br>
  <span style="color:#b4b4b4">&#9644;</span> Weak &nbsp;
  <span style="color:#e06010">&#9644;</span> Medium &nbsp;
  <span style="color:#ff1414">&#9644;</span> Strong<br><br>
  <b>Road</b><br>
  <span style="color:#c0392b">&#9679;</span> I-95 &nbsp;
  <span style="color:#2980b9">&#9679;</span> I-66 &nbsp;
  <span style="color:#27ae60">&#9679;</span> I-495 NW &nbsp;
  <span style="color:#e67e22">&#9679;</span> I-495 SW<br><br>
  <small>Model: GRU-GCN V2 (SADL-I style)</small>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

out = "learned_dependency_map.html"
m.save(out)
print(f"Saved {out}")
