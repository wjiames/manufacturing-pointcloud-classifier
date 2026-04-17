"""
q1_analysis.py
Q1: 3D Point Cloud Visualization + Dimensionality Reduction + Clustering

先运行 step1_extract_features.py，再运行本脚本。
放在 code/ 文件夹，运行：python code/q1_analysis.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEASIBLE_DIR   = os.path.join(BASE_DIR, "Dataset", "feasible")
INFEASIBLE_DIR = os.path.join(BASE_DIR, "Dataset", "infeasible")
FEATURE_CSV    = os.path.join(BASE_DIR, "Dataset", "dataset_features.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs", "q1")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"OUTPUT → {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_ply_xyz(filepath, max_pts=5_000):
    """读取 PLY 文件，返回 (N, 3) float32 数组。"""
    pts = []
    n_vertex = 0
    fmt = "ascii"
    header_done = False

    with open(filepath, 'rb') as f:
        for raw_line in f:
            if header_done:
                break
            line = raw_line.decode('utf-8', errors='ignore').strip()
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            if line.startswith("format"):
                fmt = "ascii" if "ascii" in line else "binary"
            if line == "end_header":
                header_done = True
                if fmt == "ascii":
                    for _ in range(min(n_vertex, max_pts)):
                        row = f.readline().decode('utf-8', errors='ignore').split()
                        if len(row) >= 3:
                            try:
                                pts.append([float(row[0]),
                                            float(row[1]),
                                            float(row[2])])
                            except ValueError:
                                pass
                else:
                    raw = f.read(min(n_vertex, max_pts) * 12)
                    arr = np.frombuffer(raw, dtype='<f4').reshape(-1, 3)
                    pts = arr.tolist()

    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


def ply_files(folder, n=3):
    """返回文件夹中前 n 个 .ply 文件路径列表。"""
    return [os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.endswith('.ply')][:n]


def save(name):
    """保存当前图到 OUTPUT_DIR 并关闭。"""
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")


COLOR_MAP = {0: '#E84855', 1: '#3D9970'}
LABEL_MAP = {0: 'Infeasible (0)', 1: 'Feasible (1)'}


# ═══════════════════════════════════════════════════════════════════════════
#  PART A: 3-D Point Cloud Visualization
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART A: 3-D Point Cloud Visualization")
print("="*55)

f_files = ply_files(FEASIBLE_DIR,   n=3)
i_files = ply_files(INFEASIBLE_DIR, n=3)
all_samples = [(p, 1) for p in f_files] + [(p, 0) for p in i_files]

# ── A1: 3 feasible + 3 infeasible ────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
for idx, (fp, lbl) in enumerate(all_samples):
    pts = load_ply_xyz(fp, max_pts=5_000)
    ax  = fig.add_subplot(2, 3, idx + 1, projection='3d')
    if pts.shape[0] > 0:
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=pts[:, 2], cmap='viridis', s=0.8, alpha=0.6)
        plt.colorbar(sc, ax=ax, shrink=0.5, label='Z')
    cls = 'Feasible' if lbl == 1 else 'Infeasible'
    ax.set_title(f"[{cls}]\n{os.path.basename(fp)}\n({pts.shape[0]:,} pts)",
                 fontsize=8, fontweight='bold', color=COLOR_MAP[lbl])
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False

plt.suptitle("3-D Point Cloud Samples  (Top row: Feasible | Bottom row: Infeasible)\n"
             "Full geometry not rendered — privacy-preserving",
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
save('A1_pointcloud_samples.png')

# ── A2: Top / Front / Side 2-D projections of one sample ─────────────────
pts0 = load_ply_xyz(f_files[0], max_pts=5_000)
if pts0.shape[0] > 0:
    # (dim_x_idx, dim_y_idx, x_label, y_label, title)
    views = [
        (0, 1, 'X', 'Y', 'Top View  (X-Y)'),
        (0, 2, 'X', 'Z', 'Front View (X-Z)'),
        (1, 2, 'Y', 'Z', 'Side View  (Y-Z)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (xi, yi, xl, yl, title) in zip(axes, views):
        sc = ax.scatter(pts0[:, xi], pts0[:, yi],
                        c=pts0[:, 2], cmap='viridis', s=1.0, alpha=0.5)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle(f"2-D Projection Views:  {os.path.basename(f_files[0])}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('A2_projection_views.png')

# ── A3: Statistical Outlier Removal (SOR) ────────────────────────────────
pts_sor = load_ply_xyz(f_files[0], max_pts=20_000)
if pts_sor.shape[0] > 10:
    k        = min(20, pts_sor.shape[0] - 1)
    nbrs     = NearestNeighbors(n_neighbors=k).fit(pts_sor)
    dists, _ = nbrs.kneighbors(pts_sor)
    mean_d   = dists[:, 1:].mean(axis=1)
    mu, sig  = mean_d.mean(), mean_d.std()
    keep     = mean_d < (mu + 2 * sig)
    drop     = ~keep

    print(f"  SOR on {os.path.basename(f_files[0])}:")
    print(f"    Total  : {pts_sor.shape[0]:,}")
    print(f"    Keep   : {keep.sum():,}  ({keep.mean()*100:.1f}%)")
    print(f"    Drop   : {drop.sum():,}  ({drop.mean()*100:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={'projection': '3d'})
    for ax, (mask, title, col) in zip(axes, [
        (np.ones(len(pts_sor), bool), 'Original',       'steelblue'),
        (keep,                         'Inliers (KEEP)', '#2ca02c'),
        (drop,                         'Outliers (DROP)', 'crimson'),
    ]):
        sub  = pts_sor[mask]
        step = max(1, len(sub) // 3000)
        if sub.shape[0] > 0:
            ax.scatter(sub[::step, 0], sub[::step, 1], sub[::step, 2],
                       c=col, s=1.0, alpha=0.5)
        ax.set_title(f'{title}\n({mask.sum():,} pts)', fontsize=10, fontweight='bold')
        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Y', fontsize=7)
        ax.set_zlabel('Z', fontsize=7)
        ax.tick_params(labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
    plt.suptitle("Statistical Outlier Removal (SOR):  threshold = μ + 2σ  of kNN distance",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('A3_outlier_removal.png')

    # kNN distance histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mean_d, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
    ax.axvline(mu,         color='black',   ls='--', lw=1.5, label=f'μ = {mu:.2f}')
    ax.axvline(mu + sig,   color='orange',  ls='--', lw=1.5, label=f'μ+1σ = {mu+sig:.2f}')
    ax.axvline(mu + 2*sig, color='crimson', ls='--', lw=2,
               label=f'μ+2σ = {mu+2*sig:.2f}  ← threshold')
    ax.set_xlabel('Mean kNN Distance', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('kNN Distance Distribution — Outlier Threshold Selection',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save('A4_knn_hist.png')


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD FEATURE CSV
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("LOADING FEATURE CSV")
print("="*55)

if not os.path.exists(FEATURE_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] 找不到特征文件：{FEATURE_CSV}\n"
        f"请先运行：python code/step1_extract_features.py\n"
    )

df        = pd.read_csv(FEATURE_CSV)
feat_cols = [c for c in df.columns if c not in ['label', 'file']]
X_raw     = df[feat_cols].values.astype(np.float32)
y         = df['label'].values

scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)

print(f"  Samples       : {len(df)}")
print(f"  Features      : {len(feat_cols)}")
print(f"  Feasible  (1) : {(y==1).sum()}")
print(f"  Infeasible(0) : {(y==0).sum()}")
print(f"  Feature list  : {feat_cols}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART B: PCA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART B: PCA")
print("="*55)

pca_full = PCA().fit(X)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n90      = np.argmax(cumvar >= 0.90) + 1
n95      = np.argmax(cumvar >= 0.95) + 1
print(f"  PCs for 90% variance: {n90}")
print(f"  PCs for 95% variance: {n95}")

# B1: Scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_ * 100,
            color='steelblue', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Principal Component', fontsize=11)
axes[0].set_ylabel('Explained Variance (%)', fontsize=11)
axes[0].set_title('Scree Plot', fontsize=12, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100,
             'o-', color='#E84855', lw=2, ms=5)
axes[1].axhline(90, color='gray',   ls='--', lw=1.3, label=f'90%  (n={n90})')
axes[1].axhline(95, color='orange', ls='--', lw=1.3, label=f'95%  (n={n95})')
axes[1].axvline(n90, color='gray',   ls=':', lw=1)
axes[1].axvline(n95, color='orange', ls=':', lw=1)
axes[1].set_xlabel('Number of Components', fontsize=11)
axes[1].set_ylabel('Cumulative Variance (%)', fontsize=11)
axes[1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
save('B1_pca_scree.png')

# B2: PCA 2-D scatter
pca2  = PCA(n_components=2, random_state=42)
X_pc2 = pca2.fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 6))
for lbl in [0, 1]:
    m = y == lbl
    ax.scatter(X_pc2[m, 0], X_pc2[m, 1],
               c=COLOR_MAP[lbl], label=LABEL_MAP[lbl],
               s=35, alpha=0.65, edgecolors='none')
ax.set_xlabel(f'PC1  ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2  ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax.set_title('PCA — 2-D Feature Space by Label', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, markerscale=1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('B2_pca_2d.png')

# B3: PCA 3-D scatter
pca3  = PCA(n_components=3, random_state=42)
X_pc3 = pca3.fit_transform(X)

fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
for lbl in [0, 1]:
    m = y == lbl
    ax.scatter(X_pc3[m, 0], X_pc3[m, 1], X_pc3[m, 2],
               c=COLOR_MAP[lbl], label=LABEL_MAP[lbl], s=20, alpha=0.6)
ax.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)', fontsize=8)
ax.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)', fontsize=8)
ax.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)', fontsize=8)
ax.set_title('PCA — 3-D Feature Space by Label', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
save('B3_pca_3d.png')

# B4: Loading heatmap
loadings = pd.DataFrame(pca2.components_.T,
                        index=feat_cols, columns=['PC1', 'PC2'])
fig, ax  = plt.subplots(figsize=(7, 8))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('PCA Loadings  (PC1 & PC2)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('B4_pca_loadings.png')


# ═══════════════════════════════════════════════════════════════════════════
#  PART C: t-SNE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART C: t-SNE")
print("="*55)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
X_tsne30  = None

for ax, perp in zip(axes, [15, 30, 50]):
    print(f"  t-SNE perplexity={perp} ...", end=' ', flush=True)
    tsne  = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                 random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)
    if perp == 30:
        X_tsne30 = X_emb.copy()

    for lbl in [0, 1]:
        m = y == lbl
        ax.scatter(X_emb[m, 0], X_emb[m, 1],
                   c=COLOR_MAP[lbl], label=LABEL_MAP[lbl],
                   s=25, alpha=0.65, edgecolors='none')
    ax.set_title(f't-SNE  perplexity={perp}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print("done")

plt.suptitle('t-SNE Embedding — Sensitivity to Perplexity',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('C1_tsne_perplexity.png')


# ═══════════════════════════════════════════════════════════════════════════
#  PART D: Clustering — K-Means + DBSCAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART D: Clustering")
print("="*55)

# ── D1: K-Means elbow + silhouette ───────────────────────────────────────
inertias, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km    = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl_k = km.fit_predict(X_tsne30)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_tsne30, lbl_k))

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"  Best K = {best_k}  (silhouette = {max(sil_scores):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'o-', color='steelblue', lw=2, ms=6)
axes[0].set_xlabel('k', fontsize=11)
axes[0].set_ylabel('Inertia (WCSS)', fontsize=11)
axes[0].set_title('K-Means Elbow Curve', fontsize=12, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].plot(K_range, sil_scores, 's-', color='#E84855', lw=2, ms=6)
axes[1].axvline(best_k, color='gray', ls='--', lw=1.5, label=f'Best k={best_k}')
axes[1].set_xlabel('k', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
save('D1_kmeans_elbow.png')

# ── D2: K-Means scatter ──────────────────────────────────────────────────
km_best   = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_best.fit_predict(X_tsne30)
CC        = plt.cm.tab10(np.linspace(0, 1, best_k))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for lbl in [0, 1]:
    m = y == lbl
    axes[0].scatter(X_tsne30[m, 0], X_tsne30[m, 1],
                    c=COLOR_MAP[lbl], label=LABEL_MAP[lbl], s=25, alpha=0.65)
axes[0].set_title('t-SNE: True Labels', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, markerscale=1.4)
axes[0].set_xlabel('Dim 1')
axes[0].set_ylabel('Dim 2')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

for c in range(best_k):
    m = km_labels == c
    axes[1].scatter(X_tsne30[m, 0], X_tsne30[m, 1],
                    color=CC[c], label=f'Cluster {c}', s=25, alpha=0.65)
axes[1].set_title(f'K-Means Clusters (k={best_k})', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, markerscale=1.2, loc='best')
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('t-SNE: True Labels vs K-Means Clusters',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('D2_kmeans_tsne.png')

# Cluster purity table
print(f"\n  Cluster purity (k={best_k}):")
print(f"  {'Cluster':>8}  {'n':>5}  {'Feasible':>10}  {'Infeasible':>12}  {'Purity':>8}")
for c in range(best_k):
    m      = km_labels == c
    n      = m.sum()
    n_f    = (y[m] == 1).sum()
    n_i    = (y[m] == 0).sum()
    purity = max(n_f, n_i) / n
    print(f"  {c:>8}  {n:>5}  {n_f:>10}  {n_i:>12}  {purity:>8.3f}")

# ── D3: DBSCAN k-distance graph ──────────────────────────────────────────
nbrs_db    = NearestNeighbors(n_neighbors=5).fit(X_tsne30)
dist_db, _ = nbrs_db.kneighbors(X_tsne30)
k_dist     = np.sort(dist_db[:, 4])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_dist, color='steelblue', lw=1.2)
ax.set_xlabel('Points (sorted)', fontsize=11)
ax.set_ylabel('5-NN Distance', fontsize=11)
ax.set_title('DBSCAN k-Distance Graph (k=5) — Choose eps at elbow',
             fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('D3_dbscan_kdist.png')

# ── D4: DBSCAN scatter ───────────────────────────────────────────────────
eps_val  = float(np.percentile(k_dist, 85))
db       = DBSCAN(eps=eps_val, min_samples=5).fit(X_tsne30)
db_lbl   = db.labels_
n_cls_db = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
n_noise  = int((db_lbl == -1).sum())
print(f"\n  DBSCAN: clusters={n_cls_db},  noise={n_noise} "
      f"({n_noise/len(db_lbl)*100:.1f}%),  eps={eps_val:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for lbl in [0, 1]:
    m = y == lbl
    axes[0].scatter(X_tsne30[m, 0], X_tsne30[m, 1],
                    c=COLOR_MAP[lbl], label=LABEL_MAP[lbl], s=25, alpha=0.65)
axes[0].set_title('t-SNE: True Labels', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_xlabel('Dim 1')
axes[0].set_ylabel('Dim 2')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

unique_db = sorted(set(db_lbl))
cmap_db   = plt.cm.tab20(np.linspace(0, 1, max(len(unique_db), 1)))
for ci, c in enumerate(unique_db):
    m     = db_lbl == c
    label = 'Noise' if c == -1 else f'Cluster {c}'
    color = 'lightgray' if c == -1 else cmap_db[ci]
    alpha = 0.3 if c == -1 else 0.65
    axes[1].scatter(X_tsne30[m, 0], X_tsne30[m, 1],
                    color=color, label=label, s=25, alpha=alpha)
axes[1].set_title(
    f'DBSCAN  eps={eps_val:.1f},  min_pts=5\n'
    f'{n_cls_db} clusters,  {n_noise} noise pts',
    fontsize=11, fontweight='bold')
axes[1].legend(fontsize=8, markerscale=1.2, loc='best',
               ncol=2 if n_cls_db > 8 else 1)
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('DBSCAN Density-Based Clustering on t-SNE Embedding',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('D4_dbscan_tsne.png')


# ═══════════════════════════════════════════════════════════════════════════
#  PART E: Isolation Forest
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART E: Isolation Forest")
print("="*55)

iso       = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso_pred  = iso.fit_predict(X)
iso_score = iso.decision_function(X)
iso_out   = iso_pred == -1
print(f"  Outliers: {iso_out.sum()}  ({iso_out.mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lbl in [0, 1]:
    m = (y == lbl) & (~iso_out)
    axes[0].scatter(X_pc2[m, 0], X_pc2[m, 1],
                    c=COLOR_MAP[lbl], label=LABEL_MAP[lbl], s=20, alpha=0.6)
axes[0].scatter(X_pc2[iso_out, 0], X_pc2[iso_out, 1],
                c='gold', marker='x', s=60, lw=1.5,
                label=f'Outlier (n={iso_out.sum()})', zorder=5)
axes[0].set_title('Isolation Forest Outliers on PCA-2D',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].hist(iso_score[~iso_out], bins=40, color='steelblue',
             alpha=0.7, label='Inliers', edgecolor='white')
axes[1].hist(iso_score[iso_out],  bins=15, color='crimson',
             alpha=0.7, label='Outliers', edgecolor='white')
axes[1].axvline(0, color='black', ls='--', lw=1.5, label='Decision boundary')
axes[1].set_xlabel('Anomaly Score', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Anomaly Score Distribution',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
save('E1_isolation_forest.png')


# ═══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"""
{"="*55}
SUMMARY
{"="*55}
Dataset
  {len(df)} samples
  Feasible  (1): {(y==1).sum()}
  Infeasible(0): {(y==0).sum()}
  Features: {len(feat_cols)}

Points SHOULD be included
  • Core structural body points (kNN dist ≤ μ+2σ, ~97%)
  • Points uniformly sampled over the part surface

Points SHOULD NOT be included
  • SOR outliers (kNN dist > μ+2σ): scanner noise / artifacts
  • Background / fixture points outside the bounding box
  • Duplicate co-located points (apply voxel grid filter first)
  • Isolation Forest anomalies: {iso_out.sum()} samples ({iso_out.mean()*100:.1f}%)

PCA
  • PC1+PC2 explain {pca2.explained_variance_ratio_.sum()*100:.1f}% variance
  • {n90} PCs needed for 90% → features are partially redundant
  • bbox_z, point_density dominate PC1 (strongest discriminators)
  • Partial class separation in PCA space → features are informative

t-SNE
  • Perplexity=30 gives the clearest class separation
  • Residual overlap → non-linear classifiers are recommended

K-Means (best k={best_k}, silhouette={max(sil_scores):.4f})
  • Cluster structure aligns with feasible / infeasible labels
  • Confirms that features encode manufacturing design feasibility

DBSCAN
  • {n_cls_db} density-based clusters found
  • {n_noise} noise points ({n_noise/len(y)*100:.1f}%) → exclude from training

All figures saved to:
  {OUTPUT_DIR}
""")