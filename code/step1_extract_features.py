"""
step1_extract_features.py
从 Dataset/feasible/ 和 Dataset/infeasible/ 读取所有 .ply 文件，
提取几何特征，生成 Dataset/dataset_features.csv。

放在 code/ 文件夹，运行：python code/step1_extract_features.py
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  （脚本在 code/，BASE_DIR 自动跳到项目根目录）
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEASIBLE_DIR   = os.path.join(BASE_DIR, "Dataset", "feasible")
INFEASIBLE_DIR = os.path.join(BASE_DIR, "Dataset", "infeasible")
OUTPUT_CSV     = os.path.join(BASE_DIR, "Dataset", "dataset_features.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: 读取 PLY 文件
# ─────────────────────────────────────────────────────────────────────────────
def load_ply_xyz(filepath, max_pts=50_000):
    """返回 (N, 3) float32 数组，支持 ASCII 和 Binary PLY。"""
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


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: 提取特征
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(pts, filename):
    """
    从点云提取几何特征，返回 dict。
    特征包括：
      num_points, bbox_x/y/z, bbox_volume, point_density,
      dist_mean/std/min/max, dist_p10/p25/p50/p75/p90,
      cov_eig_1/2/3, anisotropy_31/32, planarity, sphericity, linearity
    """
    if pts.shape[0] < 10:
        return None

    feat = {'file': filename}

    # 点数量
    feat['num_points'] = float(pts.shape[0])

    # 边界框
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rng = mx - mn
    feat['bbox_x'] = float(rng[0])
    feat['bbox_y'] = float(rng[1])
    feat['bbox_z'] = float(rng[2])
    vol = max(float(rng[0]) * float(rng[1]) * float(rng[2]), 1e-9)
    feat['bbox_volume']   = vol
    feat['point_density'] = feat['num_points'] / vol

    # 到质心距离分布
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    feat['dist_mean'] = float(dists.mean())
    feat['dist_std']  = float(dists.std())
    feat['dist_min']  = float(dists.min())
    feat['dist_max']  = float(dists.max())
    for p in [10, 25, 50, 75, 90]:
        feat[f'dist_p{p}'] = float(np.percentile(dists, p))

    # 协方差特征值
    sub = pts
    if pts.shape[0] > 10_000:
        idx = np.random.choice(pts.shape[0], 10_000, replace=False)
        sub = pts[idx]

    cov  = np.cov(sub.T)
    eigs = np.sort(np.linalg.eigvalsh(cov))
    e1, e2, e3 = float(eigs[0]), float(eigs[1]), float(eigs[2])
    feat['cov_eig_1']     = e1
    feat['cov_eig_2']     = e2
    feat['cov_eig_3']     = e3
    denom = e3 if e3 > 1e-9 else 1e-9
    feat['anisotropy_31'] = e3 / (e1 + 1e-9)
    feat['anisotropy_32'] = e3 / (e2 + 1e-9)
    feat['planarity']     = (e2 - e1) / denom
    feat['sphericity']    = e1 / denom
    feat['linearity']     = (e3 - e2) / denom

    return feat


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def process_folder(folder, label):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.ply')])
    print(f"\n  {len(files)} files  ←  {folder}")
    records = []
    for i, fname in enumerate(files):
        pts  = load_ply_xyz(os.path.join(folder, fname))
        feat = extract_features(pts, fname)
        if feat is not None:
            feat['label'] = label
            records.append(feat)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"    {i+1}/{len(files)} done ...", flush=True)
    return records


print("=" * 55)
print("FEATURE EXTRACTION")
print("=" * 55)

records  = process_folder(FEASIBLE_DIR,   label=1)
records += process_folder(INFEASIBLE_DIR, label=0)

df   = pd.DataFrame(records)
cols = [c for c in df.columns if c not in ['label', 'file']]
df   = df[cols + ['label', 'file']]

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*55}")
print(f"DONE")
print(f"  Total     : {len(df)}")
print(f"  Feasible  : {(df['label']==1).sum()}")
print(f"  Infeasible: {(df['label']==0).sum()}")
print(f"  Features  : {cols}")
print(f"  Saved to  : {OUTPUT_CSV}")
print(f"{'='*55}")