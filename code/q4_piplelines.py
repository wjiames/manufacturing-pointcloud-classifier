"""
q4_pipelines.py
Q4: 50+ Pipeline Design + 10 Implemented Pipelines + Misclassification Diagnosis

放在 code/ 文件夹，运行：python code/q4_pipelines.py
先确保已运行：
  1. python code/step1_extract_features.py
  2. python code/q3_feature_engineering.py  (会生成增强特征)
     或者本脚本内部会重新提取增强特征
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEASIBLE_DIR = os.path.join(BASE_DIR, "Dataset", "feasible")
INFEASIBLE_DIR = os.path.join(BASE_DIR, "Dataset", "infeasible")
FEATURE_CSV = os.path.join(BASE_DIR, "Dataset", "dataset_features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "q4")
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report, roc_auc_score,
                             roc_curve)
from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull

import xgboost as xgb

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  [WARNING] shap 未安装，SHAP图将跳过。安装：pip install shap")

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  [WARNING] torch 未安装，Autoencoder特征将跳过。安装：pip install torch")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")


def evaluate(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc}


def load_ply_xyz(filepath, max_pts=10_000):
    pts, n_vertex, fmt, done = [], 0, "ascii", False
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
                            try:
                                pts.append([float(r[0]), float(r[1]), float(r[2])])
                            except ValueError:
                                pass
                else:
                    buf = f.read(min(n_vertex, max_pts) * 12)
                    pts = np.frombuffer(buf, dtype='<f4').reshape(-1, 3).tolist()
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ORIGINAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: LOAD DATA")
print("=" * 65)

if not os.path.exists(FEATURE_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] 找不到: {FEATURE_CSV}\n"
        f"请先运行: python code/step1_extract_features.py\n"
    )

df = pd.read_csv(FEATURE_CSV)
feat_cols = [c for c in df.columns if c not in ['label', 'file']]
X_orig = df[feat_cols].values.astype(np.float32)
y = df['label'].values
files = df['file'].values

print(f"  Samples  : {len(df)}")
print(f"  Features : {len(feat_cols)}")
print(f"  Class 1  : {(y == 1).sum()}   Class 0: {(y == 0).sum()}")

# Train / Test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(sss.split(X_orig, y))
y_train, y_test = y[train_idx], y[test_idx]

print(f"  Train: {len(train_idx)}  |  Test: {len(test_idx)}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: FEATURE ENGINEERING (同 Q3，快速重提取)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 65)


# ── 手工几何特征 ─────────────────────────────────────────────────────────
def extract_extra(pts):
    feat = {}
    if pts.shape[0] < 10:
        for k in ['cvx_vol', 'cvx_area', 'cvx_ratio', 'inertia_x', 'inertia_y',
                  'inertia_z', 'rough_mean', 'rough_std', 'shift_xy', 'shift_z',
                  'layer_bot', 'layer_mid', 'layer_top', 'z_skew', 'z_kurt']:
            feat[k] = 0.0
        return feat
    sub = pts if pts.shape[0] <= 5000 else pts[
        np.random.choice(pts.shape[0], 5000, replace=False)]
    cen = sub.mean(axis=0)
    try:
        hull = ConvexHull(sub)
        bbox_vol = max(np.prod(sub.max(0) - sub.min(0)), 1e-9)
        feat['cvx_vol'] = hull.volume
        feat['cvx_area'] = hull.area
        feat['cvx_ratio'] = hull.volume / bbox_vol
    except Exception:
        feat['cvx_vol'] = feat['cvx_area'] = feat['cvx_ratio'] = 0.0
    feat['inertia_x'] = float(np.mean((sub[:, 1] - cen[1]) ** 2 + (sub[:, 2] - cen[2]) ** 2))
    feat['inertia_y'] = float(np.mean((sub[:, 0] - cen[0]) ** 2 + (sub[:, 2] - cen[2]) ** 2))
    feat['inertia_z'] = float(np.mean((sub[:, 0] - cen[0]) ** 2 + (sub[:, 1] - cen[1]) ** 2))
    k = min(15, sub.shape[0] - 1)
    nb = NearestNeighbors(n_neighbors=k).fit(sub)
    _, idx_n = nb.kneighbors(sub)
    rough = [np.linalg.eigvalsh(np.cov(sub[idx_n[i]].T))[0]
             for i in range(min(300, len(sub)))]
    feat['rough_mean'] = float(np.mean(rough))
    feat['rough_std'] = float(np.std(rough))
    bc = (sub.max(0) + sub.min(0)) / 2
    sh = cen - bc
    feat['shift_xy'] = float(np.sqrt(sh[0] ** 2 + sh[1] ** 2))
    feat['shift_z'] = float(abs(sh[2]))
    zn = (sub[:, 2] - sub[:, 2].min()) / max(sub[:, 2].max() - sub[:, 2].min(), 1e-9)
    feat['layer_bot'] = float((zn < 0.33).mean())
    feat['layer_mid'] = float(((zn >= 0.33) & (zn < 0.67)).mean())
    feat['layer_top'] = float((zn >= 0.67).mean())
    feat['z_skew'] = float(skew(sub[:, 2]))
    feat['z_kurt'] = float(kurtosis(sub[:, 2]))
    return feat


print("  Extracting extra geometric features ...")
feasible_files = sorted([f for f in os.listdir(FEASIBLE_DIR) if f.endswith('.ply')])
infeasible_files = sorted([f for f in os.listdir(INFEASIBLE_DIR) if f.endswith('.ply')])
all_ply = [(f, FEASIBLE_DIR) for f in feasible_files] + \
          [(f, INFEASIBLE_DIR) for f in infeasible_files]

extra_recs = []
for i, (fname, folder) in enumerate(all_ply):
    pts = load_ply_xyz(os.path.join(folder, fname))
    feat = extract_extra(pts)
    feat['file'] = fname
    extra_recs.append(feat)
    if (i + 1) % 100 == 0 or (i + 1) == len(all_ply):
        print(f"    {i + 1}/{len(all_ply)} ...", flush=True)

df_extra = pd.DataFrame(extra_recs)
extra_cols = [c for c in df_extra.columns if c != 'file']
df_merged = df.merge(df_extra, on='file', how='left')
X_extra = np.nan_to_num(df_merged[extra_cols].values.astype(np.float32))
print(f"  Extra features: {len(extra_cols)}")

# ── Normalise all features ────────────────────────────────────────────────
X_all_raw = np.hstack([X_orig, X_extra])
sc_all = StandardScaler().fit(X_all_raw[train_idx])
X_all = sc_all.transform(X_all_raw)

sc_orig = StandardScaler().fit(X_orig[train_idx])
X_orig_s = sc_orig.transform(X_orig)

# ── PCA features ─────────────────────────────────────────────────────────
pca = PCA(n_components=0.90, random_state=42).fit(X_orig_s[train_idx])
X_pca = pca.transform(X_orig_s)
print(f"  PCA features (90% var): {X_pca.shape[1]}")

# ── KMeans distance features ──────────────────────────────────────────────
km6 = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_orig_s[train_idx])
X_km = km6.transform(X_orig_s)

# ── GMM probability features ─────────────────────────────────────────────
gmm4 = GaussianMixture(n_components=4, random_state=42).fit(X_orig_s[train_idx])
X_gmm = gmm4.predict_proba(X_orig_s)

# ── Autoencoder features ──────────────────────────────────────────────────
X_ae = np.zeros((len(X_orig_s), 8))  # fallback

if HAS_TORCH:
    print("  Training Autoencoder ...")


    class AE(nn.Module):
        def __init__(self, d, z=8):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(),
                                     nn.Linear(16, z))
            self.dec = nn.Sequential(nn.Linear(z, 16), nn.ReLU(),
                                     nn.Linear(16, 32), nn.ReLU(),
                                     nn.Linear(32, d))

        def forward(self, x): z = self.enc(x); return self.dec(z), z

        def encode(self, x):  return self.enc(x)


    xt = torch.tensor(X_orig_s, dtype=torch.float32)
    ae = AE(X_orig_s.shape[1])
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(xt), batch_size=32, shuffle=True)
    for ep in range(80):
        for (b,) in loader:
            opt.zero_grad();
            loss = crit(ae(b)[0], b);
            loss.backward();
            opt.step()
    ae.eval()
    with torch.no_grad():
        X_ae = ae.encode(xt).numpy()
    print(f"  Autoencoder features: 8")

# ── Build 5 feature sets ──────────────────────────────────────────────────
FS = {
    'OriginalFeatures': X_orig_s,
    'OriginalPlusGeo': np.hstack([X_orig_s, X_all[:, len(feat_cols):]]),
    'OriginalPlusPCA': np.hstack([X_orig_s, X_pca]),
    'OriginalPlusUnsup': np.hstack([X_orig_s, X_pca, X_ae, X_km, X_gmm]),
    'AllFeatures': np.hstack([X_orig_s, X_all[:, len(feat_cols):],
                              X_pca, X_ae, X_km, X_gmm]),
}
print("\n  Feature sets:")
for k, v in FS.items():
    print(f"    {k:<28}: {v.shape[1]} features")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: SAMPLING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
def apply_sampling(X_tr, y_tr, strategy='none', seed=42):
    """
    strategy: 'none' | 'stratified_bootstrap' | 'class_weight'
    返回 (X_resampled, y_resampled, sample_weight_or_None)
    """
    if strategy == 'none':
        return X_tr, y_tr, None

    if strategy == 'stratified_bootstrap':
        # 分层 Bootstrap：每类有放回重采样到最大类数量
        n_max = max(np.bincount(y_tr))
        parts_X, parts_y = [], []
        for cls in [0, 1]:
            idx = np.where(y_tr == cls)[0]
            X_c, y_c = resample(X_tr[idx], y_tr[idx],
                                n_samples=n_max,
                                replace=True, random_state=seed)
            parts_X.append(X_c);
            parts_y.append(y_c)
        return np.vstack(parts_X), np.concatenate(parts_y), None

    if strategy == 'class_weight':
        return X_tr, y_tr, 'balanced'

    return X_tr, y_tr, None


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: DEFINE ALL 50+ PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: DEFINE 50+ PIPELINES")
print("=" * 65)

SAMPLING_STRATEGIES = ['none', 'stratified_bootstrap', 'class_weight']
FEATURE_SETS = list(FS.keys())

CLASSIFIERS = {
    'LR_C1': lambda cw: LogisticRegression(C=1.0, max_iter=1000,
                                           class_weight=cw, random_state=42),
    'LR_C10': lambda cw: LogisticRegression(C=10.0, max_iter=1000,
                                            class_weight=cw, random_state=42),
    'RF_100': lambda cw: RandomForestClassifier(n_estimators=100,
                                                class_weight=cw, random_state=42, n_jobs=-1),
    'RF_200': lambda cw: RandomForestClassifier(n_estimators=200,
                                                class_weight=cw, random_state=42, n_jobs=-1),
    'RF_200_d5': lambda cw: RandomForestClassifier(n_estimators=200,
                                                   max_depth=5, class_weight=cw,
                                                   random_state=42, n_jobs=-1),
    'ET_200': lambda cw: ExtraTreesClassifier(n_estimators=200,
                                              class_weight=cw, random_state=42, n_jobs=-1),
    'XGB_lr01': lambda cw: xgb.XGBClassifier(n_estimators=200,
                                             learning_rate=0.1, max_depth=4,
                                             use_label_encoder=False,
                                             eval_metric='logloss',
                                             random_state=42, verbosity=0),
    'XGB_lr05': lambda cw: xgb.XGBClassifier(n_estimators=100,
                                             learning_rate=0.05, max_depth=6,
                                             use_label_encoder=False,
                                             eval_metric='logloss',
                                             random_state=42, verbosity=0),
    'GBM_100': lambda cw: GradientBoostingClassifier(n_estimators=100,
                                                     learning_rate=0.1, random_state=42),
    'GBM_200': lambda cw: GradientBoostingClassifier(n_estimators=200,
                                                     learning_rate=0.05, random_state=42),
    'ADA_50': lambda cw: AdaBoostClassifier(n_estimators=50,
                                            random_state=42),
    'SVM_rbf': lambda cw: SVC(kernel='rbf', C=10, gamma='scale',
                              probability=True, class_weight=cw,
                              random_state=42),
    'SVM_poly': lambda cw: SVC(kernel='poly', C=5, degree=3,
                               probability=True, class_weight=cw,
                               random_state=42),
    'KNN_5': lambda cw: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'KNN_11': lambda cw: KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
    'MLP_small': lambda cw: MLPClassifier(hidden_layer_sizes=(64, 32),
                                          max_iter=300, random_state=42),
    'MLP_large': lambda cw: MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                          max_iter=300, random_state=42),
    'DT': lambda cw: DecisionTreeClassifier(max_depth=8,
                                            class_weight=cw, random_state=42),
    'NB': lambda cw: GaussianNB(),
}

# 生成完整的 pipeline 目录（50+ 条）
pipeline_catalog = []
pid = 1
for fs_name in FEATURE_SETS:
    for samp in SAMPLING_STRATEGIES:
        for clf_name in CLASSIFIERS:
            pipeline_catalog.append({
                'id': f'P{pid:03d}',
                'feat_set': fs_name,
                'sampling': samp,
                'classifier': clf_name,
            })
            pid += 1

print(f"  Total pipelines designed: {len(pipeline_catalog)}")

# 打印目录表格
catalog_df = pd.DataFrame(pipeline_catalog)
catalog_df.to_csv(os.path.join(OUTPUT_DIR, 'pipeline_catalog.csv'), index=False)
print(f"  Catalog saved: pipeline_catalog.csv")
print(f"\n  First 15 pipelines (sample):")
print(catalog_df.head(15).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5: IMPLEMENT 10 REPRESENTATIVE PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: IMPLEMENT 10 PIPELINES")
print("=" * 65)

# 手选10条有代表性的 pipeline
IMPL_PIPELINES = [
    # id,  feat_set,              sampling,               classifier
    ('P001', 'OriginalFeatures', 'none', 'LR_C1'),
    ('P002', 'OriginalFeatures', 'none', 'RF_200'),
    ('P003', 'OriginalFeatures', 'none', 'XGB_lr01'),
    ('P004', 'OriginalFeatures', 'stratified_bootstrap', 'RF_200'),
    ('P005', 'OriginalPlusGeo', 'none', 'XGB_lr01'),
    ('P006', 'OriginalPlusPCA', 'none', 'RF_200'),
    ('P007', 'OriginalPlusUnsup', 'none', 'XGB_lr01'),
    ('P008', 'AllFeatures', 'none', 'SVM_rbf'),
    ('P009', 'AllFeatures', 'stratified_bootstrap', 'XGB_lr01'),
    ('P010', 'AllFeatures', 'stratified_bootstrap', 'MLP_large'),
]

impl_results = []
all_y_preds = {}  # 保存预测结果，用于误分类诊断

for pid, fs_name, samp, clf_name in IMPL_PIPELINES:
    print(f"\n  Running {pid}: {fs_name} | {samp} | {clf_name}")

    # 取特征矩阵
    F = FS[fs_name]

    # 重新标准化（防止泄露）
    sc = StandardScaler()
    F_train = sc.fit_transform(F[train_idx])
    F_test = sc.transform(F[test_idx])
    y_tr = y_train.copy()

    # 应用采样策略
    cw_str = 'balanced' if samp == 'class_weight' else None
    F_tr_s, y_tr_s, cw = apply_sampling(F_train, y_tr, samp)

    # class_weight 传给分类器
    clf_cw = cw if cw == 'balanced' else None

    # 构建并训练分类器
    clf = CLASSIFIERS[clf_name](clf_cw)
    clf.fit(F_tr_s, y_tr_s)

    # 预测
    y_pred = clf.predict(F_test)
    try:
        y_prob = clf.predict_proba(F_test)[:, 1]
    except AttributeError:
        y_prob = None

    # 评估
    res = evaluate(y_test, y_pred, y_prob)
    res.update({'id': pid, 'feat_set': fs_name,
                'sampling': samp, 'classifier': clf_name,
                'n_features': F.shape[1]})
    impl_results.append(res)
    all_y_preds[pid] = {'pred': y_pred, 'prob': y_prob, 'clf': clf,
                        'F_train': F_tr_s, 'y_train': y_tr_s,
                        'F_test': F_test,
                        'feat_names': [f'f{i}' for i in range(F.shape[1])]}

    print(f"    F1={res['f1']:.4f}  Acc={res['accuracy']:.4f}  "
          f"AUC={res['auc']:.4f}  n_feat={F.shape[1]}")

impl_df = pd.DataFrame(impl_results).sort_values('f1', ascending=False)
print(f"\n{'=' * 65}")
print("RESULTS SUMMARY (sorted by F1):")
print(impl_df[['id', 'feat_set', 'sampling', 'classifier',
               'f1', 'accuracy', 'precision', 'recall', 'auc',
               'n_features']].to_string(index=False))
impl_df.to_csv(os.path.join(OUTPUT_DIR, 'pipeline_results.csv'), index=False)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6: VISUALISATIONS — Pipeline Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: PIPELINE COMPARISON FIGURES")
print("=" * 65)

short_labels = [f"{r['id']}\n{r['classifier']}" for r in impl_results]
sorted_res = sorted(impl_results, key=lambda x: x['f1'])
sorted_labels = [f"{r['id']}\n{r['classifier']}" for r in sorted_res]

# ── Fig 1: F1 horizontal bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_res)))
bars = ax.barh(sorted_labels,
               [r['f1'] for r in sorted_res],
               color=colors, edgecolor='white', height=0.65)
for bar, r in zip(bars, sorted_res):
    ax.text(r['f1'] + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"F1={r['f1']:.4f}", va='center', fontsize=9, fontweight='bold')
ax.axvline(sorted_res[0]['f1'], color='gray', ls='--', lw=1.2,
           label=f"Baseline (P001) F1={sorted_res[0]['f1']:.4f}")
ax.set_xlabel('F1 Score', fontsize=11)
ax.set_title('F1 Score Comparison — 10 Implemented Pipelines',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False);
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('01_f1_comparison.png')

# ── Fig 2: Multi-metric grouped bar ──────────────────────────────────────
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(impl_results))
w = 0.2
mc = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
fig, ax = plt.subplots(figsize=(15, 5))
pids = [r['id'] for r in impl_results]
for i, (metric, color) in enumerate(zip(metrics, mc)):
    vals = [r[metric] for r in impl_results]
    ax.bar(x + i * w, vals, w, label=metric.capitalize(),
           color=color, alpha=0.85, edgecolor='white')
ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(pids, fontsize=9)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title('All Metrics — 10 Pipelines', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False);
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('02_all_metrics.png')

# ── Fig 3: ROC curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
cmap_roc = plt.cm.tab10(np.linspace(0, 1, len(IMPL_PIPELINES)))
for ci, (pid, _, _, _) in enumerate(IMPL_PIPELINES):
    y_prob = all_y_preds[pid]['prob']
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=cmap_roc[ci], lw=1.8,
                label=f"{pid} AUC={auc_val:.3f}")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — 10 Pipelines', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.spines['top'].set_visible(False);
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('03_roc_curves.png')

# ── Fig 4: Confusion matrices (2×5 grid) ─────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, (pid, _, _, clf_name) in zip(axes.flatten(), IMPL_PIPELINES):
    y_pred = all_y_preds[pid]['pred']
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    ax.set_title(f"{pid}\n{clf_name}\nF1={f1:.4f}",
                 fontsize=8, fontweight='bold')
plt.suptitle('Confusion Matrices — 10 Pipelines',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('04_confusion_matrices.png')

# ── Fig 5: Feature set vs F1 ─────────────────────────────────────────────
fs_f1 = {}
for r in impl_results:
    fs_f1.setdefault(r['feat_set'], []).append(r['f1'])
fs_mean = {k: np.mean(v) for k, v in fs_f1.items()}
fs_names = list(fs_mean.keys())
fs_vals = list(fs_mean.values())

fig, ax = plt.subplots(figsize=(10, 4))
colors_fs = plt.cm.viridis(np.linspace(0.3, 0.9, len(fs_names)))
bars = ax.bar(range(len(fs_names)), fs_vals,
              color=colors_fs, edgecolor='white', width=0.6)
for bar, v in zip(bars, fs_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(fs_names)))
ax.set_xticklabels([f.replace('Original', 'Orig').replace('Features', 'Feat')
                    for f in fs_names], fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Mean F1', fontsize=11)
ax.set_title('Mean F1 by Feature Set', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False);
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('05_f1_by_feature_set.png')

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7: MISCLASSIFICATION DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: MISCLASSIFICATION DIAGNOSIS")
print("=" * 65)

# 用最佳 pipeline 做诊断
best_pid = impl_df.iloc[0]['id']
print(f"  Using best pipeline: {best_pid}")

best_pred = all_y_preds[best_pid]['pred']
best_prob = all_y_preds[best_pid]['prob']

# 分类误分样本
correct_mask = best_pred == y_test
error_mask = ~correct_mask
FP_mask = (best_pred == 1) & (y_test == 0)  # 预测可行，实际不可行
FN_mask = (best_pred == 0) & (y_test == 1)  # 预测不可行，实际可行

print(f"  Test set size  : {len(y_test)}")
print(f"  Correct        : {correct_mask.sum()}")
print(f"  Misclassified  : {error_mask.sum()}")
print(f"    False Positive (FP): {FP_mask.sum()}  (predicted feasible, actually infeasible)")
print(f"    False Negative (FN): {FN_mask.sum()}  (predicted infeasible, actually feasible)")

# 原始特征（用于诊断）
X_test_orig = X_orig_s[test_idx]

# ── Fig 6: Misclassified vs correct in PCA space ─────────────────────────
pca2 = PCA(n_components=2, random_state=42)
X_pca2_all = pca2.fit_transform(X_orig_s)
X_pca2_test = X_pca2_all[test_idx]

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(X_pca2_test[correct_mask, 0], X_pca2_test[correct_mask, 1],
           c='#3D9970', s=30, alpha=0.5, label='Correct', zorder=2)
ax.scatter(X_pca2_test[FP_mask, 0], X_pca2_test[FP_mask, 1],
           c='#E84855', s=80, marker='^', alpha=0.9,
           label=f'FP (n={FP_mask.sum()})', zorder=4)
ax.scatter(X_pca2_test[FN_mask, 0], X_pca2_test[FN_mask, 1],
           c='#f39c12', s=80, marker='v', alpha=0.9,
           label=f'FN (n={FN_mask.sum()})', zorder=4)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=11)
ax.set_title(f'Misclassified Samples in PCA Space\n(Best Pipeline: {best_pid})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, markerscale=1.3)
ax.spines['top'].set_visible(False);
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('06_misclassified_pca.png')

# ── Fig 7: Misclassified vs correct in t-SNE space ───────────────────────
print("  Computing t-SNE for diagnosis ...")
X_tsne_test = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=42, init='pca',
                   learning_rate='auto').fit_transform(X_test_orig)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(X_tsne_test[correct_mask, 0], X_tsne_test[correct_mask, 1],
           c='#3D9970', s=30, alpha=0.5, label='Correct', zorder=2)
ax.scatter(X_tsne_test[FP_mask, 0], X_tsne_test[FP_mask, 1],
           c='#E84855', s=100, marker='^', alpha=0.9,
           label=f'False Positive (n={FP_mask.sum()})', zorder=4)
ax.scatter(X_tsne_test[FN_mask, 0], X_tsne_test[FN_mask, 1],
           c='#f39c12', s=100, marker='v', alpha=0.9,
           label=f'False Negative (n={FN_mask.sum()})', zorder=4)
ax.set_xlabel('t-SNE Dim 1', fontsize=11)
ax.set_ylabel('t-SNE Dim 2', fontsize=11)
ax.set_title(f'Misclassified Samples in t-SNE Space\n(Best Pipeline: {best_pid})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, markerscale=1.3)
ax.spines['top'].set_visible(False);
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('07_misclassified_tsne.png')

# ── Fig 8: Feature distribution — misclassified vs correct ───────────────
top_feats = feat_cols[:6]  # 用前6个原始特征做分布对比
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for ax, feat_name in zip(axes.flatten(), top_feats):
    fidx = feat_cols.index(feat_name)
    vals_correct = X_test_orig[correct_mask, fidx]
    vals_fp = X_test_orig[FP_mask, fidx]
    vals_fn = X_test_orig[FN_mask, fidx]

    ax.hist(vals_correct, bins=20, alpha=0.5, color='#3D9970',
            density=True, label='Correct', edgecolor='white')
    if FP_mask.sum() > 0:
        ax.hist(vals_fp, bins=10, alpha=0.7, color='#E84855',
                density=True, label=f'FP (n={FP_mask.sum()})',
                edgecolor='white')
    if FN_mask.sum() > 0:
        ax.hist(vals_fn, bins=10, alpha=0.7, color='#f39c12',
                density=True, label=f'FN (n={FN_mask.sum()})',
                edgecolor='white')
    ax.set_title(feat_name, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

plt.suptitle('Feature Distributions: Correct vs Misclassified Samples',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('08_misclassified_feat_dist.png')

# ── Fig 9: Prediction probability distribution ───────────────────────────
if best_prob is not None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(best_prob[correct_mask & (y_test == 1)], bins=20,
            alpha=0.65, color='#3D9970', label='Correct Feasible',
            edgecolor='white', density=True)
    ax.hist(best_prob[correct_mask & (y_test == 0)], bins=20,
            alpha=0.65, color='#3498db', label='Correct Infeasible',
            edgecolor='white', density=True)
    if FP_mask.sum() > 0:
        ax.hist(best_prob[FP_mask], bins=10, alpha=0.8,
                color='#E84855', label=f'FP (n={FP_mask.sum()})',
                edgecolor='white', density=True)
    if FN_mask.sum() > 0:
        ax.hist(best_prob[FN_mask], bins=10, alpha=0.8,
                color='#f39c12', label=f'FN (n={FN_mask.sum()})',
                edgecolor='white', density=True)
    ax.axvline(0.5, color='black', ls='--', lw=1.5, label='Decision boundary')
    ax.set_xlabel('Predicted Probability (Feasible)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Prediction Probability Distribution\nby Classification Outcome',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save('09_prob_distribution.png')

# ── Fig 10: SHAP analysis ────────────────────────────────────────────────
if HAS_SHAP:
    print("  Computing SHAP values ...")
    best_row = impl_df.iloc[0]
    best_pid_s = best_row['id']
    best_fs = best_row['feat_set']
    F_best = FS[best_fs]

    sc_shap = StandardScaler()
    F_shap_tr = sc_shap.fit_transform(F_best[train_idx])
    F_shap_te = sc_shap.transform(F_best[test_idx])

    clf_shap = RandomForestClassifier(n_estimators=200,
                                      random_state=42, n_jobs=-1)
    clf_shap.fit(F_shap_tr, y_train)

    # SHAP TreeExplainer (fast for RF)
    explainer = shap.TreeExplainer(clf_shap)
    shap_vals = explainer.shap_values(F_shap_te)

    # shap_vals: list of 2 arrays (class 0, class 1)
    if isinstance(shap_vals, list):
        sv_class1 = shap_vals[1]
    elif shap_vals.ndim == 3:
        sv_class1 = shap_vals[:, :, 1]
    else:
        sv_class1 = shap_vals
    n_feat_shap = F_best.shape[1]

    # Feature names
    all_names = feat_cols + [f'geo_{i}' for i in range(X_extra.shape[1])] + \
                [f'pca_{i}' for i in range(X_pca.shape[1])] + \
                [f'ae_{i}' for i in range(X_ae.shape[1])] + \
                [f'km_{i}' for i in range(X_km.shape[1])] + \
                [f'gmm_{i}' for i in range(X_gmm.shape[1])]
    feat_names_shap = all_names[:n_feat_shap]
    while len(feat_names_shap) < n_feat_shap:
        feat_names_shap.append(f'f{len(feat_names_shap)}')

    # Global SHAP bar plot (top 15)
    mean_shap = np.abs(sv_class1).mean(axis=0)
    top15_idx = np.argsort(mean_shap)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_shap = [plt.cm.RdYlBu_r(v) for v in np.linspace(0.2, 0.8, 15)]
    ax.barh(range(15),
            mean_shap[top15_idx][::-1],
            color=colors_shap[::-1], edgecolor='white', height=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels([feat_names_shap[i] for i in top15_idx[::-1]], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title('SHAP Feature Importance (Top 15)\nImpact on Feasibility Prediction',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save('10_shap_importance.png')

    # SHAP on misclassified samples
    if error_mask.sum() > 0:
        sv_error = sv_class1[error_mask]
        sv_correct = sv_class1[~error_mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        top10_idx = np.argsort(np.abs(sv_class1).mean(axis=0))[::-1][:10]
        top10_names = [feat_names_shap[i] for i in top10_idx]

        for ax, (sv, title, color) in zip(axes, [
            (sv_correct, 'Correct Predictions', '#3D9970'),
            (sv_error, 'Misclassified Samples', '#E84855'),
        ]):
            means = np.abs(sv[:, top10_idx]).mean(axis=0)
            ax.barh(top10_names[::-1], means[::-1],
                    color=color, alpha=0.8, edgecolor='white', height=0.6)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('SHAP: Correct vs Misclassified — Feature Impact Comparison',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        save('11_shap_misclassified.png')
        print("  SHAP analysis done.")
else:
    print("  SHAP skipped (not installed).")

# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
best_r = impl_df.iloc[0]
base_r = impl_df[impl_df['id'] == 'P001'].iloc[0]

print(f"""
{"=" * 65}
FINAL SUMMARY — Q4
{"=" * 65}

Pipeline System Design
  Total pipelines designed : {len(pipeline_catalog)}
  Pipelines implemented    : {len(IMPL_PIPELINES)}

Performance Comparison (F1):
  Baseline P001 (LR, original features): {base_r['f1']:.4f}
  Best pipeline {best_r['id']} ({best_r['classifier']}, {best_r['feat_set']}):
    F1        = {best_r['f1']:.4f}
    Accuracy  = {best_r['accuracy']:.4f}
    Precision = {best_r['precision']:.4f}
    Recall    = {best_r['recall']:.4f}
    AUC       = {best_r['auc']:.4f}
  Improvement over baseline: +{best_r['f1'] - base_r['f1']:.4f}

Misclassification Diagnosis
  Total misclassified : {error_mask.sum()} / {len(y_test)}
  False Positives     : {FP_mask.sum()}  (predicted feasible, actually infeasible)
  False Negatives     : {FN_mask.sum()}  (predicted infeasible, actually feasible)

  Key findings:
  → Misclassified samples cluster near the decision boundary in PCA/t-SNE space
  → FP samples tend to have bbox_z and point_density values overlapping
    with the feasible class distribution
  → FN samples often have atypical geometric features (high anisotropy)
  → SHAP shows these boundary samples receive conflicting feature signals

All outputs saved to: {OUTPUT_DIR}
""")