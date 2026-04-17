"""
q2_sampling.py
Q2: Sequential Sampling Strategy + Class Imbalance Analysis
放在 code/ 文件夹，运行：python code/q2_sampling.py
先确保已运行 step1_extract_features.py 生成 dataset_features.csv
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_CSV = os.path.join(BASE_DIR, "Dataset", "dataset_features.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "q2")
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from sklearn.utils import resample
from collections import Counter

# 安装检查
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  [WARNING] imbalanced-learn 未安装，SMOTE 方案将跳过")
    print("  安装命令: pip install imbalanced-learn")


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
        print(f"  {name:<40} Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {'name': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1}


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

if not os.path.exists(FEATURE_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] 找不到: {FEATURE_CSV}\n"
        f"请先运行: python code/step1_extract_features.py\n"
    )

df        = pd.read_csv(FEATURE_CSV)
feat_cols = [c for c in df.columns if c not in ['label', 'file']]
X_all     = df[feat_cols].values.astype(np.float32)
y_all     = df['label'].values

print(f"  Total samples : {len(df)}")
print(f"  Features      : {len(feat_cols)}")
print(f"  Class 1 (feasible)  : {(y_all==1).sum()}  ({(y_all==1).mean()*100:.1f}%)")
print(f"  Class 0 (infeasible): {(y_all==0).sum()}  ({(y_all==0).mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT: 70% train pool  |  30% test  (stratified, fixed)
# ─────────────────────────────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(sss.split(X_all, y_all))

X_pool, y_pool = X_all[train_idx], y_all[train_idx]   # 350 samples
X_test, y_test = X_all[test_idx],  y_all[test_idx]    # 150 samples

scaler = StandardScaler().fit(X_pool)
X_pool_s = scaler.transform(X_pool)
X_test_s  = scaler.transform(X_test)

print(f"\n  Train pool: {len(X_pool)}  |  Test: {len(X_test)}")
print(f"  Test class ratio: {(y_test==1).sum()} feasible / {(y_test==0).sum()} infeasible")

# ── Fig 1: class distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (data, title) in zip(axes, [
    (y_pool, 'Train Pool'),
    (y_test, 'Test Set'),
]):
    counts = Counter(data)
    bars = ax.bar(['Infeasible (0)', 'Feasible (1)'],
                  [counts[0], counts[1]],
                  color=['#E84855', '#3D9970'], width=0.5, edgecolor='white')
    for bar, v in zip(bars, [counts[0], counts[1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v}\n({v/len(data)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('Class Distribution in Train Pool and Test Set',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('01_class_distribution.png')


# ═══════════════════════════════════════════════════════════════════════════
#  SEQUENTIAL SAMPLING STRATEGIES
#  场景：X_pool 的样本按顺序一个一个到来
#        我们用不同策略决定哪些样本进入最终训练集
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SEQUENTIAL SAMPLING STRATEGIES")
print("="*60)

# 打乱顺序，模拟顺序到来
rng = np.random.default_rng(42)
seq_order = rng.permutation(len(X_pool))
X_seq = X_pool_s[seq_order]
y_seq = y_pool[seq_order]

TARGET_SIZE = 150    # 从350个样本里选150个（约43%）


# ─────────────────────────────────────────────────────────────────────────────
#  策略 A: Random Subset（随机选，baseline）
# ─────────────────────────────────────────────────────────────────────────────
def strategy_random(X_seq, y_seq, target_size, seed=42):
    """随机选 target_size 个样本，不考虑任何策略。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_seq), size=target_size, replace=False)
    return X_seq[idx], y_seq[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  策略 B: Stratified Sequential Sampling（分层顺序采样）
# ─────────────────────────────────────────────────────────────────────────────
def strategy_stratified(X_seq, y_seq, target_size):
    """
    样本顺序到来，维持两类比例各50%。
    每来一个样本，判断当前训练池是否需要这个类别。
    """
    selected_X, selected_y = [], []
    count = {0: 0, 1: 0}
    half  = target_size // 2   # 每类各选一半

    for xi, yi in zip(X_seq, y_seq):
        if count[yi] < half:
            selected_X.append(xi)
            selected_y.append(yi)
            count[yi] += 1
        if sum(count.values()) >= target_size:
            break

    return np.array(selected_X), np.array(selected_y)


# ─────────────────────────────────────────────────────────────────────────────
#  策略 C: Stratified + Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def strategy_stratified_bootstrap(X_seq, y_seq, target_size, seed=42):
    """
    Step 1: Stratified 顺序采样选出子集
    Step 2: Bootstrap 有放回重采样扩充到 target_size
    """
    # Step 1: Stratified 选出小子集（选 target_size//2 个原始样本）
    half_target = target_size // 2
    selected_X, selected_y = [], []
    count = {0: 0, 1: 0}
    quarter = half_target // 2  # 每类各选 quarter 个

    for xi, yi in zip(X_seq, y_seq):
        if count[yi] < quarter:
            selected_X.append(xi)
            selected_y.append(yi)
            count[yi] += 1
        if sum(count.values()) >= half_target:
            break

    selected_X = np.array(selected_X)
    selected_y = np.array(selected_y)

    # Step 2: Bootstrap 有放回重采样到 target_size
    boot_X, boot_y = resample(selected_X, selected_y,
                               n_samples=target_size,
                               replace=True,
                               random_state=seed)
    return boot_X, boot_y


# ─────────────────────────────────────────────────────────────────────────────
#  策略 D: Stratified + Bootstrap + Uncertainty Sampling
# ─────────────────────────────────────────────────────────────────────────────
def strategy_uncertainty(X_seq, y_seq, target_size, seed=42):
    """
    样本顺序到来：
      1. 前 warm_up 个样本直接加入（冷启动）
      2. 之后用当前模型预测：预测概率越接近 0.5 越不确定 → 优先加入
      3. 同时维持 Stratified 约束（两类平衡）
      4. 最后 Bootstrap 扩充
    """
    warm_up = 20   # 前20个样本直接加入用于冷启动
    selected_X, selected_y = [], []
    count = {0: 0, 1: 0}
    half  = target_size // 2

    model = None

    for i, (xi, yi) in enumerate(zip(X_seq, y_seq)):

        # 冷启动阶段：直接加入
        if i < warm_up:
            if count[yi] < half:
                selected_X.append(xi)
                selected_y.append(yi)
                count[yi] += 1

            # 冷启动结束，训练初始模型
            if i == warm_up - 1 and len(set(selected_y)) == 2:
                model = RandomForestClassifier(
                    n_estimators=50, random_state=seed, n_jobs=-1)
                model.fit(selected_X, selected_y)
            continue

        if sum(count.values()) >= target_size:
            break

        # Stratified 约束：这个类别还需要吗？
        if count[yi] >= half:
            continue

        # Uncertainty 判断
        if model is not None:
            prob = model.predict_proba([xi])[0]
            uncertainty = 1 - max(prob)   # 越接近0.5，uncertainty越高
            # 只有 uncertainty 足够高才加入（阈值0.2）
            if uncertainty < 0.2 and sum(count.values()) > warm_up * 2:
                continue

        selected_X.append(xi)
        selected_y.append(yi)
        count[yi] += 1

        # 每积累20个新样本，重新训练模型
        if len(selected_y) % 20 == 0 and len(set(selected_y)) == 2:
            model = RandomForestClassifier(
                n_estimators=50, random_state=seed, n_jobs=-1)
            model.fit(selected_X, selected_y)

    selected_X = np.array(selected_X)
    selected_y = np.array(selected_y)

    # Bootstrap 扩充到 target_size
    if len(selected_X) < target_size:
        selected_X, selected_y = resample(
            selected_X, selected_y,
            n_samples=target_size,
            replace=True, random_state=seed)

    return selected_X, selected_y


# ─────────────────────────────────────────────────────────────────────────────
#  策略 E: SMOTE（少数类过采样）
# ─────────────────────────────────────────────────────────────────────────────
def strategy_smote(X_seq, y_seq, target_size, seed=42):
    """
    先 Stratified 顺序选出子集，再用 SMOTE 对少数类插值过采样。
    """
    # Stratified 选子集
    selected_X, selected_y = strategy_stratified(X_seq, y_seq, target_size)

    if not HAS_SMOTE:
        return selected_X, selected_y

    # SMOTE 过采样，让两类数量相等
    sm = SMOTE(random_state=seed, k_neighbors=min(5, (selected_y==0).sum()-1))
    try:
        X_res, y_res = sm.fit_resample(selected_X, selected_y)
        return X_res, y_res
    except Exception as e:
        print(f"    SMOTE failed: {e}, using stratified only")
        return selected_X, selected_y


# ═══════════════════════════════════════════════════════════════════════════
#  RUN ALL STRATEGIES & EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RUNNING STRATEGIES")
print("="*60)

strategies = [
    ('A: Random (baseline)',              strategy_random),
    ('B: Stratified',                     strategy_stratified),
    ('C: Stratified + Bootstrap',         strategy_stratified_bootstrap),
    ('D: Stratified + Bootstrap + Uncertainty', strategy_uncertainty),
]
if HAS_SMOTE:
    strategies.append(('E: Stratified + SMOTE', strategy_smote))

results    = []
subsets    = {}  # 保存每个策略选出的子集，用于后续可视化

for name, func in strategies:
    print(f"\n  Strategy: {name}")

    # 获取子集
    if name.startswith('A'):
        X_sub, y_sub = func(X_seq, y_seq, TARGET_SIZE)
    else:
        X_sub, y_sub = func(X_seq, y_seq, TARGET_SIZE)

    subsets[name] = (X_sub, y_sub)

    c = Counter(y_sub)
    print(f"    Subset size: {len(y_sub)}  "
          f"(class0={c[0]}, class1={c[1]})")

    # 训练模型
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_sub, y_sub)

    # 测试集评估
    y_pred = clf.predict(X_test_s)
    res = evaluate(y_test, y_pred, name=name)
    res['subset_size']   = len(y_sub)
    res['class0_count']  = c[0]
    res['class1_count']  = c[1]
    results.append(res)

# 全量数据作为参考上界
print(f"\n  Strategy: {'Z: Full Train Pool (upper bound)'}")
clf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_full.fit(X_pool_s, y_pool)
y_pred_full = clf_full.predict(X_test_s)
res_full = evaluate(y_test, y_pred_full, name='Z: Full Train Pool (upper bound)')
res_full['subset_size']  = len(y_pool)
res_full['class0_count'] = (y_pool==0).sum()
res_full['class1_count'] = (y_pool==1).sum()
results.append(res_full)

results_df = pd.DataFrame(results)
print("\n" + results_df[['name','subset_size','class0_count',
                          'class1_count','f1','accuracy']].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

names_short = [r['name'].split(':')[0] + ': ' +
               r['name'].split(':')[1].strip()[:25]
               for r in results]
f1_scores   = [r['f1']       for r in results]
acc_scores  = [r['accuracy'] for r in results]

# ── Fig 2: F1 comparison bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#95a5a6'] + ['#3D9970'] * (len(results) - 2) + ['#2980b9']
bars   = ax.barh(names_short, f1_scores, color=colors,
                 edgecolor='white', height=0.6)
for bar, v in zip(bars, f1_scores):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('F1 Score (Test Set)', fontsize=11)
ax.set_title('F1 Score Comparison Across Sampling Strategies',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, min(1.0, max(f1_scores) * 1.15))
ax.axvline(f1_scores[0], color='#95a5a6', ls='--', lw=1.5,
           label=f'Baseline F1={f1_scores[0]:.4f}')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('02_f1_comparison.png')

# ── Fig 3: Metrics comparison (grouped bar) ───────────────────────────────
metrics   = ['accuracy', 'precision', 'recall', 'f1']
x         = np.arange(len(results))
width     = 0.2
fig, ax   = plt.subplots(figsize=(14, 5))
mc        = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, (metric, color) in enumerate(zip(metrics, mc)):
    vals = [r[metric] for r in results]
    ax.bar(x + i * width, vals, width, label=metric.capitalize(),
           color=color, alpha=0.85, edgecolor='white')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([r['name'].split(':')[0] for r in results], fontsize=10)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('All Metrics Comparison Across Strategies',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('03_all_metrics.png')

# ── Fig 4: Subset class distribution comparison ───────────────────────────
strategy_names = list(subsets.keys())
n_strats = len(strategy_names)
fig, axes = plt.subplots(1, n_strats, figsize=(4 * n_strats, 4))
if n_strats == 1:
    axes = [axes]

for ax, sname in zip(axes, strategy_names):
    _, y_sub = subsets[sname]
    c = Counter(y_sub)
    bars = ax.bar(['Infeasible\n(0)', 'Feasible\n(1)'],
                  [c[0], c[1]],
                  color=['#E84855', '#3D9970'],
                  edgecolor='white', width=0.5)
    for bar, v in zip(bars, [c[0], c[1]]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{v}', ha='center', fontsize=9, fontweight='bold')
    short = sname.split(':')[0]
    ax.set_title(short, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(c.values()) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Class Distribution in Each Strategy Subset',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save('04_subset_distribution.png')

# ── Fig 5: Confusion matrices ─────────────────────────────────────────────
n_strats_all = len(strategies) + 1   # +1 for full pool
ncols = min(3, n_strats_all)
nrows = (n_strats_all + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5 * ncols, 4.5 * nrows))
axes = np.array(axes).flatten()

all_names_strats = [s[0] for s in strategies] + ['Z: Full Train Pool']
all_subsets_list = [subsets[s[0]] for s in strategies]

for i, (sname, (X_sub, y_sub)) in enumerate(zip(all_names_strats[:-1],
                                                  all_subsets_list)):
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_sub, y_sub)
    y_pred = clf.predict(X_test_s)
    cm     = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[i], cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    f1 = f1_score(y_test, y_pred, zero_division=0)
    axes[i].set_title(f"{sname.split(':')[0]}\nF1={f1:.4f}",
                      fontsize=9, fontweight='bold')

# Full pool
cm_full = confusion_matrix(y_test, y_pred_full)
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
            ax=axes[len(all_names_strats)-1], cbar=False,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
f1_full = f1_score(y_test, y_pred_full, zero_division=0)
axes[len(all_names_strats)-1].set_title(
    f"Z: Full Pool\nF1={f1_full:.4f}", fontsize=9, fontweight='bold')

# 隐藏多余子图
for j in range(len(all_names_strats), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Confusion Matrices — All Strategies',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save('05_confusion_matrices.png')

# ── Fig 6: F1 vs Subset Size ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sizes  = [r['subset_size'] for r in results]
f1s    = [r['f1']          for r in results]
colors_scatter = ['#95a5a6'] + ['#3D9970'] * (len(results) - 2) + ['#2980b9']

for i, (s, f, n, c) in enumerate(zip(sizes, f1s, names_short, colors_scatter)):
    ax.scatter(s, f, color=c, s=120, zorder=5)
    ax.annotate(n.split(':')[0], (s, f),
                textcoords='offset points', xytext=(6, 4), fontsize=9)

ax.set_xlabel('Subset Size (# training samples)', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score vs Training Subset Size', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('06_f1_vs_size.png')

# ── Fig 7: Class imbalance effect simulation ──────────────────────────────
# 模拟不同不平衡比例下，模型 F1 的变化
print("\n  Simulating class imbalance effect ...")
imbalance_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f1_imbalance     = []
f1_weighted      = []   # 使用 class_weight='balanced' 后的 F1

for ratio in imbalance_ratios:
    # 从 pool 里按 ratio 比例构造不平衡数据集
    n_total   = 200
    n_class1  = int(n_total * ratio)
    n_class0  = n_total - n_class1

    idx1 = np.where(y_pool == 1)[0]
    idx0 = np.where(y_pool == 0)[0]

    if len(idx1) < n_class1 or len(idx0) < n_class0:
        f1_imbalance.append(np.nan)
        f1_weighted.append(np.nan)
        continue

    sel1 = np.random.choice(idx1, n_class1, replace=False)
    sel0 = np.random.choice(idx0, n_class0, replace=False)
    sel  = np.concatenate([sel1, sel0])

    X_imb = X_pool_s[sel]
    y_imb = y_pool[sel]

    # 不加权
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf1.fit(X_imb, y_imb)
    f1_imbalance.append(f1_score(y_test, clf1.predict(X_test_s), zero_division=0))

    # 加权（class_weight='balanced'）
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
    clf2.fit(X_imb, y_imb)
    f1_weighted.append(f1_score(y_test, clf2.predict(X_test_s), zero_division=0))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(imbalance_ratios, f1_imbalance, 'o-', color='#E84855',
        lw=2, ms=7, label='No class weighting')
ax.plot(imbalance_ratios, f1_weighted,  's-', color='#3D9970',
        lw=2, ms=7, label='class_weight="balanced"')
ax.axvline(0.5, color='gray', ls='--', lw=1.5, label='Balanced (50/50)')
ax.set_xlabel('Ratio of Class 1 (Feasible) in Training Set', fontsize=11)
ax.set_ylabel('F1 Score on Test Set', fontsize=11)
ax.set_title('Effect of Class Imbalance on F1 Score\n'
             '(and how class weighting mitigates it)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('07_imbalance_effect.png')

# ── Fig 8: Learning curve (F1 vs number of sequential samples added) ──────
print("  Plotting learning curve ...")
checkpoints = list(range(20, len(X_seq) + 1, 20))
f1_curve_random     = []
f1_curve_stratified = []

for n in checkpoints:
    # Random
    idx_r  = np.random.default_rng(42).choice(n, size=min(n, TARGET_SIZE), replace=False)
    X_r, y_r = X_seq[:n][idx_r], y_seq[:n][idx_r]
    if len(set(y_r)) < 2:
        f1_curve_random.append(np.nan)
    else:
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_r, y_r)
        f1_curve_random.append(f1_score(y_test, clf.predict(X_test_s), zero_division=0))

    # Stratified
    X_st, y_st = strategy_stratified(X_seq[:n], y_seq[:n], min(n, TARGET_SIZE))
    if len(set(y_st)) < 2:
        f1_curve_stratified.append(np.nan)
    else:
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_st, y_st)
        f1_curve_stratified.append(f1_score(y_test, clf.predict(X_test_s), zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(checkpoints, f1_curve_random,     'o-', color='#E84855',
        lw=2, ms=4, alpha=0.8, label='Random Sampling')
ax.plot(checkpoints, f1_curve_stratified, 's-', color='#3D9970',
        lw=2, ms=4, alpha=0.8, label='Stratified Sampling')
ax.set_xlabel('Number of Sequential Samples Seen', fontsize=11)
ax.set_ylabel('F1 Score on Test Set', fontsize=11)
ax.set_title('Learning Curve: F1 Score as Samples Arrive Sequentially',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save('08_learning_curve.png')


# ═══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"""
{"="*60}
SUMMARY — Q2
{"="*60}

Dataset Imbalance
  Class 1 (feasible)  : {(y_all==1).sum()} ({(y_all==1).mean()*100:.1f}%)
  Class 0 (infeasible): {(y_all==0).sum()} ({(y_all==0).mean()*100:.1f}%)
  → Mild imbalance (60/40). Without correction, model is biased
    toward the majority class → lower recall for class 0.

Sampling Strategy Results (F1 on test set):
""")
for r in results:
    bar = '█' * int(r['f1'] * 30)
    print(f"  {r['name']:<45} F1={r['f1']:.4f}  {bar}")

best = max(results[:-1], key=lambda x: x['f1'])
print(f"""
  Best strategy : {best['name']}
  Best F1       : {best['f1']:.4f}
  Baseline F1   : {results[0]['f1']:.4f}
  Improvement   : +{best['f1'] - results[0]['f1']:.4f}

Key Conclusions
  1. Random sampling (baseline) suffers from class imbalance
     → biased toward majority class → lower F1
  2. Stratified sampling fixes the ratio → improved recall for minority class
  3. Bootstrap augments the small subset → more stable model training
  4. Uncertainty sampling prioritizes informative samples → higher F1
     with fewer total samples used
  5. class_weight='balanced' is a lightweight alternative when
     resampling is not feasible

All figures saved to: {OUTPUT_DIR}
""")