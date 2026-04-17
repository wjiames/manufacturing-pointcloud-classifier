"""
q3_feature_engineering.py
Q3: Literature Review + Unsupervised Feature Engineering + Model Comparison

放在 code/ 文件夹，运行：python code/q3_feature_engineering.py
先确保已运行 step1_extract_features.py 生成 dataset_features.csv
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEASIBLE_DIR   = os.path.join(BASE_DIR, "Dataset", "feasible")
INFEASIBLE_DIR = os.path.join(BASE_DIR, "Dataset", "infeasible")
FEATURE_CSV = os.path.join(BASE_DIR, "Dataset", "dataset_features.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "q3")
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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, classification_report)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")


def evaluate(y_true, y_pred, name=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    if name:
        print(f"  {name:<45} Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {'name': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1}


def load_ply_xyz(filepath, max_pts=10_000):
    pts = []
    n_vertex, fmt, done = 0, "ascii", False
    with open(filepath, 'rb') as f:
        for raw in f:
            if done: break
            line = raw.decode('utf-8', errors='ignore').strip()
            if line.startswith("element vertex"): n_vertex = int(line.split()[-1])
            if line.startswith("format"):         fmt = "ascii" if "ascii" in line else "binary"
            if line == "end_header":
                done = True
                if fmt == "ascii":
                    for _ in range(min(n_vertex, max_pts)):
                        r = f.readline().decode('utf-8', errors='ignore').split()
                        if len(r) >= 3:
                            try: pts.append([float(r[0]), float(r[1]), float(r[2])])
                            except ValueError: pass
                else:
                    buf = f.read(min(n_vertex, max_pts) * 12)
                    pts = np.frombuffer(buf, dtype='<f4').reshape(-1, 3).tolist()
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ORIGINAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING ORIGINAL FEATURES")
print("="*60)

if not os.path.exists(FEATURE_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] 找不到: {FEATURE_CSV}\n"
        f"请先运行: python code/step1_extract_features.py\n"
    )

df        = pd.read_csv(FEATURE_CSV)
feat_cols = [c for c in df.columns if c not in ['label', 'file']]
X_orig    = df[feat_cols].values.astype(np.float32)
y         = df['label'].values
files     = df['file'].values

print(f"  Samples  : {len(df)}")
print(f"  Features : {len(feat_cols)}  →  {feat_cols}")
print(f"  Class 1  : {(y==1).sum()}   Class 0: {(y==0).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN / TEST SPLIT  (70 / 30 stratified)
# ─────────────────────────────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(sss.split(X_orig, y))

X_tr_orig, y_train = X_orig[train_idx], y[train_idx]
X_te_orig, y_test  = X_orig[test_idx],  y[test_idx]

scaler_orig = StandardScaler().fit(X_tr_orig)
X_tr_s = scaler_orig.transform(X_tr_orig)
X_te_s = scaler_orig.transform(X_te_orig)

print(f"\n  Train: {len(X_tr_orig)}  |  Test: {len(X_te_orig)}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART A: ADDITIONAL HAND-CRAFTED GEOMETRIC FEATURES
#  来源：Rusu et al. (2009) FPFH; Weinmann et al. (2015) 3D点云特征综述
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART A: Additional Hand-Crafted Geometric Features")
print("="*60)

def extract_extra_features(pts):
    """
    从点云提取额外几何特征（文献方法）：
    - 凸包特征 (Convex Hull)
    - 惯性矩 (Inertia Moments)
    - 表面粗糙度 (Surface Roughness)
    - 重心偏移 (Centroid Shift)
    - 高度分层密度 (Layer Density)
    """
    feat = {}

    if pts.shape[0] < 10:
        return {k: 0.0 for k in [
            'convex_hull_volume', 'convex_hull_area', 'convexity_ratio',
            'inertia_x', 'inertia_y', 'inertia_z',
            'roughness_mean', 'roughness_std',
            'centroid_shift_xy', 'centroid_shift_z',
            'layer_density_bottom', 'layer_density_mid', 'layer_density_top',
            'z_skewness', 'z_kurtosis'
        ]}

    # 随机采样加速
    sub = pts
    if pts.shape[0] > 5000:
        idx = np.random.choice(pts.shape[0], 5000, replace=False)
        sub = pts[idx]

    centroid = sub.mean(axis=0)

    # ── 凸包特征 ─────────────────────────────────────────────────────────
    try:
        hull = ConvexHull(sub)
        feat['convex_hull_volume'] = hull.volume
        feat['convex_hull_area']   = hull.area
        # 凸包体积 / 边界框体积 → 描述形状紧凑程度
        bbox_vol = max(np.prod(sub.max(axis=0) - sub.min(axis=0)), 1e-9)
        feat['convexity_ratio'] = hull.volume / bbox_vol
    except Exception:
        feat['convex_hull_volume'] = 0.0
        feat['convex_hull_area']   = 0.0
        feat['convexity_ratio']    = 0.0

    # ── 惯性矩 ───────────────────────────────────────────────────────────
    # 对每个轴：sum((距离该轴的垂直距离)^2)
    feat['inertia_x'] = float(np.mean((sub[:, 1] - centroid[1])**2 +
                                       (sub[:, 2] - centroid[2])**2))
    feat['inertia_y'] = float(np.mean((sub[:, 0] - centroid[0])**2 +
                                       (sub[:, 2] - centroid[2])**2))
    feat['inertia_z'] = float(np.mean((sub[:, 0] - centroid[0])**2 +
                                       (sub[:, 1] - centroid[1])**2))

    # ── 表面粗糙度 ───────────────────────────────────────────────────────
    # 每个点到其局部平面的偏差
    k_rough = min(15, sub.shape[0] - 1)
    nbrs    = NearestNeighbors(n_neighbors=k_rough).fit(sub)
    _, idx_n = nbrs.kneighbors(sub)

    roughness = []
    for i in range(min(500, len(sub))):   # 只算500个点加速
        neighbors = sub[idx_n[i]]
        # 用PCA拟合局部平面，计算到平面的距离
        local_cov  = np.cov(neighbors.T)
        eig_vals   = np.linalg.eigvalsh(local_cov)
        roughness.append(float(eig_vals[0]))  # 最小特征值 = 平面拟合残差

    feat['roughness_mean'] = float(np.mean(roughness))
    feat['roughness_std']  = float(np.std(roughness))

    # ── 重心偏移 ─────────────────────────────────────────────────────────
    bbox_center = (sub.max(axis=0) + sub.min(axis=0)) / 2
    shift = centroid - bbox_center
    feat['centroid_shift_xy'] = float(np.sqrt(shift[0]**2 + shift[1]**2))
    feat['centroid_shift_z']  = float(abs(shift[2]))

    # ── 高度分层密度 ─────────────────────────────────────────────────────
    z_min, z_max = sub[:, 2].min(), sub[:, 2].max()
    z_range = z_max - z_min if z_max > z_min else 1e-9
    z_norm  = (sub[:, 2] - z_min) / z_range   # 归一化到 [0,1]
    feat['layer_density_bottom'] = float((z_norm < 0.33).mean())
    feat['layer_density_mid']    = float(((z_norm >= 0.33) & (z_norm < 0.67)).mean())
    feat['layer_density_top']    = float((z_norm >= 0.67).mean())

    # ── Z轴分布形状 ──────────────────────────────────────────────────────
    from scipy.stats import skew, kurtosis
    feat['z_skewness'] = float(skew(sub[:, 2]))
    feat['z_kurtosis'] = float(kurtosis(sub[:, 2]))

    return feat


# 从 PLY 文件提取额外特征
print("  Extracting extra geometric features from PLY files ...")
extra_records = []

feasible_files   = sorted([f for f in os.listdir(FEASIBLE_DIR)   if f.endswith('.ply')])
infeasible_files = sorted([f for f in os.listdir(INFEASIBLE_DIR) if f.endswith('.ply')])

file_to_label = {}
for f in feasible_files:   file_to_label[f] = 1
for f in infeasible_files: file_to_label[f] = 0

all_ply_files = [(f, 1, FEASIBLE_DIR)   for f in feasible_files] + \
                [(f, 0, INFEASIBLE_DIR) for f in infeasible_files]

for i, (fname, lbl, folder) in enumerate(all_ply_files):
    fpath = os.path.join(folder, fname)
    pts   = load_ply_xyz(fpath, max_pts=10_000)
    feat  = extract_extra_features(pts)
    feat['file']  = fname
    feat['label'] = lbl
    extra_records.append(feat)
    if (i+1) % 100 == 0 or (i+1) == len(all_ply_files):
        print(f"    {i+1}/{len(all_ply_files)} done ...", flush=True)

df_extra     = pd.DataFrame(extra_records)
extra_cols   = [c for c in df_extra.columns if c not in ['file', 'label']]

# 按 file 列对齐，合并到主 df
df_merged = df.merge(df_extra[extra_cols + ['file']], on='file', how='left')
X_extra   = df_merged[extra_cols].values.astype(np.float32)

# 填充 NaN
X_extra = np.nan_to_num(X_extra, nan=0.0)

print(f"  Extra features added: {len(extra_cols)}  →  {extra_cols}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART B: PCA-BASED UNSUPERVISED FEATURES
#  来源：Jolliffe (2002) Principal Component Analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART B: PCA Unsupervised Features")
print("="*60)

scaler_pca = StandardScaler()
X_scaled   = scaler_pca.fit_transform(X_orig)

# 保留解释90%方差的主成分
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA components kept (90% variance): {pca.n_components_}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

pca_cols = [f'pca_{i+1}' for i in range(X_pca.shape[1])]


# ═══════════════════════════════════════════════════════════════════════════
#  PART C: AUTOENCODER UNSUPERVISED FEATURES
#  来源：Hinton & Salakhutdinov (2006) Reducing Dimensionality with Neural Networks
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART C: Autoencoder Unsupervised Features")
print("="*60)

class Autoencoder(nn.Module):
    """
    全连接自编码器
    输入 → 编码器（压缩）→ 瓶颈层（8维）→ 解码器（重建）→ 输出
    只用重建损失训练，不用标签 → 无监督
    """
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


# 训练 Autoencoder
input_dim  = X_scaled.shape[1]
latent_dim = 8
ae         = Autoencoder(input_dim, latent_dim)
optimizer  = optim.Adam(ae.parameters(), lr=1e-3)
criterion  = nn.MSELoss()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset  = TensorDataset(X_tensor)
loader   = DataLoader(dataset, batch_size=32, shuffle=True)

print("  Training Autoencoder ...")
ae_losses = []
n_epochs  = 100

for epoch in range(n_epochs):
    ae.train()
    total_loss = 0
    for (batch,) in loader:
        optimizer.zero_grad()
        recon, _ = ae(batch)
        loss     = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    ae_losses.append(avg_loss)
    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}")

# 提取潜在向量（8维）作为新特征
ae.eval()
with torch.no_grad():
    X_ae = ae.encode(X_tensor).numpy()

ae_cols = [f'ae_{i+1}' for i in range(latent_dim)]
print(f"  Autoencoder latent features: {latent_dim} dims")

# 训练损失曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, n_epochs+1), ae_losses, color='steelblue', lw=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=11)
ax.set_title('Autoencoder Training Loss', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); save('01_autoencoder_loss.png')


# ═══════════════════════════════════════════════════════════════════════════
#  PART D: K-MEANS DISTANCE FEATURES (Unsupervised)
#  来源：Coates & Ng (2012) Learning Feature Representations with K-Means
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART D: K-Means Distance Features (Unsupervised)")
print("="*60)

N_CLUSTERS = 6
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
km.fit(X_scaled)

# 每个样本到每个聚类中心的距离 → 作为新特征
X_km_dist  = km.transform(X_scaled)          # (N, K)
km_labels  = km.labels_
km_cols    = [f'km_dist_{i}' for i in range(N_CLUSTERS)]

print(f"  K-Means clusters: {N_CLUSTERS}")
print(f"  New distance features: {len(km_cols)}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART E: GMM PROBABILITY FEATURES (Unsupervised)
#  来源：Reynolds (2009) Gaussian Mixture Models
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART E: GMM Probability Features (Unsupervised)")
print("="*60)

N_COMPONENTS = 4
gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42,
                       covariance_type='full', max_iter=200)
gmm.fit(X_scaled)

# 每个样本属于每个高斯分布的概率 → 作为新特征
X_gmm     = gmm.predict_proba(X_scaled)      # (N, K)
gmm_cols  = [f'gmm_prob_{i}' for i in range(N_COMPONENTS)]

print(f"  GMM components: {N_COMPONENTS}")
print(f"  New probability features: {len(gmm_cols)}")


# ═══════════════════════════════════════════════════════════════════════════
#  ASSEMBLE FEATURE SETS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ASSEMBLING FEATURE SETS")
print("="*60)

# 原始特征（标准化）
F_orig = X_scaled.copy()

# 额外手工特征（标准化）
scaler_extra = StandardScaler()
F_extra      = scaler_extra.fit_transform(X_extra)

# 全部特征拼接
F_all = np.hstack([F_orig, F_extra, X_pca, X_ae, X_km_dist, X_gmm])

feature_sets = {
    'F1: Original (20 feats)':
        F_orig,
    'F2: Original + Geometric (20+15 feats)':
        np.hstack([F_orig, F_extra]),
    'F3: Original + PCA (20+PCA feats)':
        np.hstack([F_orig, X_pca]),
    'F4: Original + Autoencoder (20+8 feats)':
        np.hstack([F_orig, X_ae]),
    'F5: Original + KMeans-dist (20+6 feats)':
        np.hstack([F_orig, X_km_dist]),
    'F6: Original + GMM-prob (20+4 feats)':
        np.hstack([F_orig, X_gmm]),
    'F7: All Features Combined':
        F_all,
}

for name, feat in feature_sets.items():
    print(f"  {name:<45}: {feat.shape[1]} features")


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN & EVALUATE ALL FEATURE SETS × MULTIPLE MODELS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING & EVALUATING")
print("="*60)

models = {
    'Random Forest':          RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=42),
    'SVM (RBF)':              SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
}

all_results = []

for feat_name, F in feature_sets.items():
    F_train = F[train_idx]
    F_test  = F[test_idx]

    # 重新标准化（每个特征集独立）
    sc = StandardScaler()
    F_train = sc.fit_transform(F_train)
    F_test  = sc.transform(F_test)

    for model_name, clf in models.items():
        clf_copy = type(clf)(**clf.get_params())   # 重新实例化，避免状态污染
        clf_copy.fit(F_train, y_train)
        y_pred = clf_copy.predict(F_test)

        res = evaluate(y_test, y_pred,
                       name=f"{feat_name[:25]:<26} | {model_name}")
        res['feature_set'] = feat_name
        res['model']       = model_name
        res['n_features']  = F.shape[1]
        all_results.append(res)

results_df = pd.DataFrame(all_results)
print(f"\n  Total experiments: {len(results_df)}")


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

# ── Fig 2: F1 heatmap (feature set × model) ──────────────────────────────
pivot = results_df.pivot_table(
    index='feature_set', columns='model', values='f1', aggfunc='mean')

# 简化行标签
short_idx = [f.split('(')[0].strip() for f in pivot.index]
pivot.index = short_idx

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
            ax=ax, linewidths=0.5, vmin=0.5, vmax=1.0,
            cbar_kws={'shrink': 0.8})
ax.set_title('F1 Score Heatmap: Feature Set × Model',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Feature Set', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
save('02_f1_heatmap.png')

# ── Fig 3: F1 by feature set (best model per set) ────────────────────────
best_per_set = results_df.groupby('feature_set')['f1'].max().reset_index()
best_per_set = best_per_set.sort_values('f1', ascending=True)
short_labels = [f.split('(')[0].strip() for f in best_per_set['feature_set']]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#95a5a6' if 'Original' in l and '+' not in l
          else '#3D9970' for l in short_labels]
bars = ax.barh(short_labels, best_per_set['f1'].values,
               color=colors, edgecolor='white', height=0.6)
for bar, v in zip(bars, best_per_set['f1'].values):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
ax.axvline(best_per_set[best_per_set['feature_set'].str.contains(
    'Original \(20')]['f1'].values[0],
    color='#95a5a6', ls='--', lw=1.5, label='Baseline (Original features)')
ax.set_xlabel('Best F1 Score (across models)', fontsize=11)
ax.set_title('Best F1 per Feature Set\n(Feature Engineering Improvement)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1.05)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
save('03_f1_by_feature_set.png')

# ── Fig 4: F1 by model (averaged across feature sets) ────────────────────
avg_by_model = results_df.groupby('model')['f1'].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(avg_by_model.index, avg_by_model.values,
               color='steelblue', alpha=0.85, edgecolor='white', height=0.5)
for bar, v in zip(bars, avg_by_model.values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Average F1 Score', fontsize=11)
ax.set_title('Average F1 by Model (across all feature sets)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
save('04_f1_by_model.png')

# ── Fig 5: Feature importance (RF on full feature set) ───────────────────
F_all_train = F_all[train_idx]
F_all_test  = F_all[test_idx]
sc_all      = StandardScaler()
F_all_train = sc_all.fit_transform(F_all_train)
F_all_test  = sc_all.transform(F_all_test)

rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_full.fit(F_all_train, y_train)

all_feat_names = (feat_cols + extra_cols + pca_cols +
                  ae_cols + km_cols + gmm_cols)
importances    = rf_full.feature_importances_

# Top 20 features
top20_idx  = np.argsort(importances)[::-1][:20]
top20_imp  = importances[top20_idx]
top20_name = [all_feat_names[i] if i < len(all_feat_names)
              else f'feat_{i}' for i in top20_idx]

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = []
for n in top20_name:
    if n.startswith('pca_'):       colors_imp.append('#3498db')
    elif n.startswith('ae_'):      colors_imp.append('#e74c3c')
    elif n.startswith('km_'):      colors_imp.append('#2ecc71')
    elif n.startswith('gmm_'):     colors_imp.append('#f39c12')
    elif n in extra_cols:          colors_imp.append('#9b59b6')
    else:                          colors_imp.append('#95a5a6')

ax.barh(range(20), top20_imp[::-1], color=colors_imp[::-1],
        edgecolor='white', height=0.7)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_name[::-1], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Top 20 Feature Importances (RF on All Features)',
             fontsize=12, fontweight='bold')
# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(color='#95a5a6', label='Original features'),
    Patch(color='#9b59b6', label='Extra geometric features'),
    Patch(color='#3498db', label='PCA features'),
    Patch(color='#e74c3c', label='Autoencoder features'),
    Patch(color='#2ecc71', label='K-Means dist features'),
    Patch(color='#f39c12', label='GMM prob features'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
save('05_feature_importance.png')

# ── Fig 6: Correlation heatmap of new features ────────────────────────────
# 展示各类特征之间的相关性
sample_feats = pd.DataFrame(
    np.hstack([X_pca[:, :3], X_ae[:, :3], X_km_dist[:, :3], X_gmm]),
    columns=['PCA1','PCA2','PCA3',
             'AE1','AE2','AE3',
             'KM_d1','KM_d2','KM_d3',
             'GMM_p1','GMM_p2','GMM_p3','GMM_p4']
)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sample_feats.corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, linewidths=0.5,
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation of Unsupervised Features\n'
             '(PCA / Autoencoder / K-Means / GMM)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save('06_unsupervised_feature_corr.png')

# ── Fig 7: t-SNE comparison (original vs all features) ───────────────────
from sklearn.manifold import TSNE

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
COLOR_MAP = {0: '#E84855', 1: '#3D9970'}
LABEL_MAP = {0: 'Infeasible', 1: 'Feasible'}

for ax, (X_viz, title) in zip(axes, [
    (F_orig, 'Original 20 Features'),
    (F_all,  'All Augmented Features'),
]):
    sc_viz  = StandardScaler()
    X_viz_s = sc_viz.fit_transform(X_viz)
    X_tsne  = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=42, init='pca',
                   learning_rate='auto').fit_transform(X_viz_s)
    for lbl in [0, 1]:
        m = y == lbl
        ax.scatter(X_tsne[m, 0], X_tsne[m, 1],
                   c=COLOR_MAP[lbl], label=LABEL_MAP[lbl],
                   s=25, alpha=0.65, edgecolors='none')
    ax.set_title(f't-SNE: {title}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, markerscale=1.5)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.suptitle('t-SNE: Original Features vs Augmented Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('07_tsne_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

summary = results_df.groupby(['feature_set', 'model']).agg(
    f1=('f1', 'mean'),
    accuracy=('accuracy', 'mean'),
).reset_index().sort_values('f1', ascending=False)

print(summary[['feature_set', 'model', 'f1', 'accuracy']].head(15).to_string(index=False))

baseline_f1 = results_df[
    results_df['feature_set'].str.contains('Original \(20')
]['f1'].max()

best_f1 = results_df['f1'].max()
best_row = results_df.loc[results_df['f1'].idxmax()]

print(f"""
{"="*60}
KEY FINDINGS
{"="*60}
  Baseline F1 (Original 20 features, best model) : {baseline_f1:.4f}
  Best F1     (All augmented features)            : {best_f1:.4f}
  Improvement                                     : +{best_f1-baseline_f1:.4f}

  Best configuration:
    Feature set : {best_row['feature_set']}
    Model       : {best_row['model']}
    F1          : {best_row['f1']:.4f}

Feature Engineering Summary:
  Original stats features  : {len(feat_cols)} features
  + Extra geometric (A)    : {len(extra_cols)} features  (convex hull, inertia, roughness...)
  + PCA (B)                : {X_pca.shape[1]} features  (unsupervised, 90% variance)
  + Autoencoder (C)        : {latent_dim} features  (unsupervised, learned representation)
  + K-Means distances (D)  : {N_CLUSTERS} features  (unsupervised, cluster distances)
  + GMM probabilities (E)  : {N_COMPONENTS} features  (unsupervised, soft assignments)
  Total                    : {F_all.shape[1]} features

Role of Unsupervised Learning in Feature Engineering:
  PCA        → removes redundancy, keeps orthogonal directions of variance
  Autoencoder → learns non-linear compact representation without labels
  K-Means    → encodes global cluster membership as continuous distances
  GMM        → soft probabilistic cluster assignment, richer than K-Means

All figures saved to: {OUTPUT_DIR}
""")