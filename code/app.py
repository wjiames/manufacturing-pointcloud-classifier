"""
app.py  —  Manufacturing Part Feasibility Classifier
Streamlit Online Application

运行方式：streamlit run code/app.py
"""

import os
import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull
from collections import Counter
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3D Point Cloud Feasibility Classifier",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_CSV = os.path.join(BASE_DIR, "Dataset", "dataset_features.csv")
FEASIBLE_DIR   = os.path.join(BASE_DIR, "Dataset", "feasible")
INFEASIBLE_DIR = os.path.join(BASE_DIR, "Dataset", "infeasible")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_ply_xyz(filepath_or_bytes, max_pts=10_000):
    """读取 PLY 文件，返回 (N, 3) float32 数组。"""
    pts = []
    n_vertex, fmt, done = 0, "ascii", False

    if isinstance(filepath_or_bytes, (str, bytes, os.PathLike)):
        f_obj = open(filepath_or_bytes, 'rb')
    else:
        f_obj = io.BytesIO(filepath_or_bytes)

    with f_obj as f:
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


def extract_features(pts, filename="uploaded.ply"):
    """从点云提取完整特征向量。"""
    feat = {}
    if pts.shape[0] < 10:
        return None

    feat['num_points']    = float(pts.shape[0])
    mn, mx = pts.min(0), pts.max(0)
    rng = mx - mn
    feat['bbox_x'] = float(rng[0])
    feat['bbox_y'] = float(rng[1])
    feat['bbox_z'] = float(rng[2])
    vol = max(float(np.prod(rng)), 1e-9)
    feat['bbox_volume']   = vol
    feat['point_density'] = feat['num_points'] / vol

    centroid = pts.mean(0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    feat['dist_mean'] = float(dists.mean())
    feat['dist_std']  = float(dists.std())
    feat['dist_min']  = float(dists.min())
    feat['dist_max']  = float(dists.max())
    for p in [10, 25, 50, 75, 90]:
        feat[f'dist_p{p}'] = float(np.percentile(dists, p))

    sub = pts if pts.shape[0] <= 10_000 else pts[
        np.random.choice(pts.shape[0], 10_000, replace=False)]
    eigs = np.sort(np.linalg.eigvalsh(np.cov(sub.T)))
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


def extract_extra_features(pts):
    """提取额外几何特征。"""
    feat = {}
    zero_feat = {k: 0.0 for k in ['cvx_vol','cvx_area','cvx_ratio',
                                    'inertia_x','inertia_y','inertia_z',
                                    'rough_mean','rough_std','shift_xy',
                                    'shift_z','layer_bot','layer_mid',
                                    'layer_top','z_skew','z_kurt']}
    if pts.shape[0] < 10:
        return zero_feat

    sub = pts if pts.shape[0] <= 5000 else pts[
        np.random.choice(pts.shape[0], 5000, replace=False)]
    cen = sub.mean(0)

    try:
        hull = ConvexHull(sub)
        bbox_vol = max(np.prod(sub.max(0) - sub.min(0)), 1e-9)
        feat['cvx_vol']   = hull.volume
        feat['cvx_area']  = hull.area
        feat['cvx_ratio'] = hull.volume / bbox_vol
    except Exception:
        feat['cvx_vol'] = feat['cvx_area'] = feat['cvx_ratio'] = 0.0

    feat['inertia_x'] = float(np.mean((sub[:,1]-cen[1])**2+(sub[:,2]-cen[2])**2))
    feat['inertia_y'] = float(np.mean((sub[:,0]-cen[0])**2+(sub[:,2]-cen[2])**2))
    feat['inertia_z'] = float(np.mean((sub[:,0]-cen[0])**2+(sub[:,1]-cen[1])**2))

    k = min(15, sub.shape[0]-1)
    nb = NearestNeighbors(n_neighbors=k).fit(sub)
    _, idx_n = nb.kneighbors(sub)
    rough = [np.linalg.eigvalsh(np.cov(sub[idx_n[i]].T))[0]
             for i in range(min(200, len(sub)))]
    feat['rough_mean'] = float(np.mean(rough))
    feat['rough_std']  = float(np.std(rough))

    bc = (sub.max(0) + sub.min(0)) / 2
    sh = cen - bc
    feat['shift_xy'] = float(np.sqrt(sh[0]**2 + sh[1]**2))
    feat['shift_z']  = float(abs(sh[2]))

    z_range = max(sub[:,2].max() - sub[:,2].min(), 1e-9)
    zn = (sub[:,2] - sub[:,2].min()) / z_range
    feat['layer_bot'] = float((zn < 0.33).mean())
    feat['layer_mid'] = float(((zn >= 0.33) & (zn < 0.67)).mean())
    feat['layer_top'] = float((zn >= 0.67).mean())
    feat['z_skew']    = float(skew(sub[:,2]))
    feat['z_kurt']    = float(kurtosis(sub[:,2]))

    return feat


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training models on dataset...")
def train_all_pipelines():
    """训练全部10条pipeline，缓存模型。"""
    if not os.path.exists(FEATURE_CSV):
        return None, None, None, None

    df        = pd.read_csv(FEATURE_CSV)
    feat_cols = [c for c in df.columns if c not in ['label', 'file']]
    X_orig    = df[feat_cols].values.astype(np.float32)
    y         = df['label'].values

    sc = StandardScaler()
    X_s = sc.fit_transform(X_orig)

    pca   = PCA(n_components=0.90, random_state=42).fit(X_s)
    X_pca = pca.transform(X_s)
    km6   = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_s)
    X_km  = km6.transform(X_s)
    gmm4  = GaussianMixture(n_components=4, random_state=42).fit(X_s)
    X_gmm = gmm4.predict_proba(X_s)

    X_all = np.hstack([X_s, X_pca, X_km, X_gmm])
    sc_all = StandardScaler().fit(X_all)
    X_all_s = sc_all.transform(X_all)

    from sklearn.utils import resample as skresample
    def boot_resample(X, y, seed=42):
        n = max(np.bincount(y))
        parts = []
        for c in [0,1]:
            idx = np.where(y==c)[0]
            Xc, yc = skresample(X[idx], y[idx], n_samples=n,
                                 replace=True, random_state=seed)
            parts.append((Xc, yc))
        return (np.vstack([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]))

    PIPELINES = {
        'P001 — Logistic Regression (Original)':
            (LogisticRegression(C=1.0, max_iter=1000, random_state=42), X_s, sc, False),
        'P002 — Random Forest (Original)':
            (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X_s, sc, False),
        'P003 — XGBoost (Original)':
            (xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                use_label_encoder=False, eval_metric='logloss',
                                random_state=42, verbosity=0), X_s, sc, False),
        'P004 — RF + Stratified Bootstrap (Original)':
            (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X_s, sc, True),
        'P005 — XGBoost (All Features)':
            (xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                use_label_encoder=False, eval_metric='logloss',
                                random_state=42, verbosity=0), X_all_s, sc_all, False),
        'P006 — Random Forest (All Features)':
            (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X_all_s, sc_all, False),
        'P007 — XGBoost + Bootstrap (All Features)':
            (xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                use_label_encoder=False, eval_metric='logloss',
                                random_state=42, verbosity=0), X_all_s, sc_all, True),
        'P008 — SVM RBF (All Features)':
            (SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                 random_state=42), X_all_s, sc_all, False),
        'P009 — Gradient Boosting (All Features)':
            (GradientBoostingClassifier(n_estimators=100, random_state=42), X_all_s, sc_all, False),
        'P010 — MLP Neural Network (All Features)':
            (MLPClassifier(hidden_layer_sizes=(128,64,32),
                           max_iter=300, random_state=42), X_all_s, sc_all, True),
    }

    trained = {}
    for name, (clf, X_feat, _, do_boot) in PIPELINES.items():
        if do_boot:
            X_b, y_b = boot_resample(X_feat, y)
            clf.fit(X_b, y_b)
        else:
            clf.fit(X_feat, y)
        trained[name] = clf

    transformers = {
        'sc_orig': sc, 'sc_all': sc_all,
        'pca': pca, 'km6': km6, 'gmm4': gmm4,
        'feat_cols': feat_cols,
    }

    pipeline_uses_all = {
        'P001 — Logistic Regression (Original)':       False,
        'P002 — Random Forest (Original)':             False,
        'P003 — XGBoost (Original)':                   False,
        'P004 — RF + Stratified Bootstrap (Original)': False,
        'P005 — XGBoost (All Features)':               True,
        'P006 — Random Forest (All Features)':         True,
        'P007 — XGBoost + Bootstrap (All Features)':   True,
        'P008 — SVM RBF (All Features)':               True,
        'P009 — Gradient Boosting (All Features)':     True,
        'P010 — MLP Neural Network (All Features)':    True,
    }

    return trained, transformers, pipeline_uses_all, feat_cols


def predict_single(pts_feat_dict, extra_feat_dict,
                   pipeline_name, trained, transformers, uses_all):
    """对单个样本运行指定 pipeline 并返回预测结果。"""
    feat_cols = transformers['feat_cols']
    sc_orig   = transformers['sc_orig']
    sc_all    = transformers['sc_all']
    pca       = transformers['pca']
    km6       = transformers['km6']
    gmm4      = transformers['gmm4']

    # 原始特征向量
    x_orig = np.array([[pts_feat_dict.get(c, 0.0) for c in feat_cols]],
                       dtype=np.float32)
    x_orig_s = sc_orig.transform(x_orig)

    if uses_all[pipeline_name]:
        x_pca = pca.transform(x_orig_s)
        x_km  = km6.transform(x_orig_s)
        x_gmm = gmm4.predict_proba(x_orig_s)
        x_in  = sc_all.transform(np.hstack([x_orig_s, x_pca, x_km, x_gmm]))
    else:
        x_in = x_orig_s

    clf   = trained[pipeline_name]
    pred  = clf.predict(x_in)[0]
    prob  = clf.predict_proba(x_in)[0] if hasattr(clf, 'predict_proba') else None

    return int(pred), prob


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/"
                 "Virginia_Tech_Hokies_logo.svg/200px-Virginia_Tech_Hokies_logo.svg.png",
                 width=80)
st.sidebar.title("⚙️ Control Panel")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home",
     "📁 Upload & Predict",
     "📊 Dataset Overview",
     "🔬 Pipeline Comparison",
     "ℹ️ About"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏭 Manufacturing Part Feasibility Classifier")
    st.markdown("### 3D Point Cloud Based Machine Learning Pipeline System")

    st.markdown("""
    This application classifies 3D point cloud data of manufacturing parts as
    **✅ Feasible** or **❌ Infeasible** using multiple ML pipelines.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Samples",  "500")
    col2.metric("✅ Feasible",        "300  (60%)")
    col3.metric("❌ Infeasible",      "200  (40%)")
    col4.metric("🔧 Pipelines",       "10 implemented\n(50+ designed)")

    st.markdown("---")

    st.markdown("### 🗺️ System Architecture")
    st.code("""
    .ply Files
        ↓
    Feature Extraction (20 original + 15 geometric + PCA + AE + KMeans + GMM)
        ↓
    Sampling Strategy (Random / Stratified+Bootstrap / Class Weighting)
        ↓
    Classification Pipeline (LR / RF / XGBoost / SVM / GBM / MLP)
        ↓
    Prediction: Feasible ✅  or  Infeasible ❌
        ↓
    SHAP Explanation + Diagnosis
    """, language="text")

    st.markdown("### 📋 How to Use")
    st.info("""
    1. Go to **📁 Upload & Predict** → upload your `.ply` file → get prediction
    2. Go to **📊 Dataset Overview** → explore the training data
    3. Go to **🔬 Pipeline Comparison** → compare all 10 pipelines
    """)


# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD & PREDICT PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Upload & Predict":
    st.title("📁 Upload Point Cloud & Predict")

    # Load models
    trained, transformers, uses_all, feat_cols = train_all_pipelines()

    if trained is None:
        st.error(f"❌ Dataset not found at `{FEATURE_CSV}`.\n"
                 "Please run `python code/step1_extract_features.py` first.")
        st.stop()

    st.success("✅ All 10 pipelines loaded and ready!")

    # Pipeline selector
    st.markdown("### 1️⃣ Select Pipeline")
    pipeline_name = st.selectbox(
        "Choose a classification pipeline:",
        list(trained.keys()),
        index=6,   # default to P007 (best)
    )
    st.caption(f"Selected: **{pipeline_name}**")

    # File uploader
    st.markdown("### 2️⃣ Upload Your .ply File")
    uploaded = st.file_uploader(
        "Upload a 3D point cloud file (.ply format)",
        type=['ply'],
        help="Upload a .ply file from your manufacturing design",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()

        with st.spinner("Processing point cloud ..."):
            pts = load_ply_xyz(file_bytes, max_pts=10_000)

        if pts.shape[0] < 10:
            st.error("❌ Could not read point cloud. Please check the file format.")
            st.stop()

        st.success(f"✅ Loaded **{pts.shape[0]:,}** points from `{uploaded.name}`")

        # ── 3D Visualization ──────────────────────────────────────────────
        st.markdown("### 3️⃣ Point Cloud Visualization")
        tab1, tab2, tab3 = st.tabs(["3D View", "Top View (X-Y)", "Side View (Y-Z)"])

        with tab1:
            step = max(1, len(pts) // 3000)
            pts_sub = pts[::step]
            fig3d = go.Figure(data=[go.Scatter3d(
                x=pts_sub[:, 0], y=pts_sub[:, 1], z=pts_sub[:, 2],
                mode='markers',
                marker=dict(size=1.5, color=pts_sub[:, 2],
                            colorscale='Viridis', opacity=0.7,
                            colorbar=dict(title='Z'))
            )])
            fig3d.update_layout(
                title=f"3D Point Cloud: {uploaded.name}",
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                height=450, margin=dict(l=0, r=0, b=0, t=40),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with tab2:
            fig_top = px.scatter(x=pts_sub[:, 0], y=pts_sub[:, 1],
                                  color=pts_sub[:, 2],
                                  color_continuous_scale='Viridis',
                                  labels={'x': 'X', 'y': 'Y', 'color': 'Z'},
                                  title='Top View (X-Y Plane)')
            fig_top.update_traces(marker=dict(size=2))
            st.plotly_chart(fig_top, use_container_width=True)

        with tab3:
            fig_side = px.scatter(x=pts_sub[:, 1], y=pts_sub[:, 2],
                                   color=pts_sub[:, 2],
                                   color_continuous_scale='Viridis',
                                   labels={'x': 'Y', 'y': 'Z', 'color': 'Z'},
                                   title='Side View (Y-Z Plane)')
            fig_side.update_traces(marker=dict(size=2))
            st.plotly_chart(fig_side, use_container_width=True)

        # ── Feature Extraction ────────────────────────────────────────────
        st.markdown("### 4️⃣ Extracted Features")
        with st.spinner("Extracting features ..."):
            feat_dict  = extract_features(pts, uploaded.name)
            extra_dict = extract_extra_features(pts)

        if feat_dict is None:
            st.error("❌ Feature extraction failed.")
            st.stop()

        feat_df = pd.DataFrame([feat_dict]).T.reset_index()
        feat_df.columns = ['Feature', 'Value']
        feat_df['Value'] = feat_df['Value'].round(4)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(feat_df, use_container_width=True, height=400)
        with col2:
            fig_feat = px.bar(feat_df, x='Value', y='Feature',
                               orientation='h',
                               title='Feature Values',
                               color='Value',
                               color_continuous_scale='RdYlGn')
            fig_feat.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_feat, use_container_width=True)

        # ── Prediction ────────────────────────────────────────────────────
        st.markdown("### 5️⃣ Prediction Result")

        with st.spinner("Running classification pipeline ..."):
            pred, prob = predict_single(feat_dict, extra_dict,
                                         pipeline_name, trained,
                                         transformers, uses_all)

        if pred == 1:
            st.success("## ✅ FEASIBLE DESIGN")
            st.balloons()
        else:
            st.error("## ❌ INFEASIBLE DESIGN")

        if prob is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction",    "Feasible" if pred==1 else "Infeasible")
            col2.metric("P(Feasible)",   f"{prob[1]*100:.1f}%")
            col3.metric("P(Infeasible)", f"{prob[0]*100:.1f}%")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[1] * 100,
                title={'text': "Feasibility Confidence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "#3D9970" if pred==1 else "#E84855"},
                    'steps': [
                        {'range': [0,   40], 'color': '#fde8e8'},
                        {'range': [40,  60], 'color': '#fef9e7'},
                        {'range': [60, 100], 'color': '#e8f8f0'},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.8, 'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.info(f"**Pipeline used**: {pipeline_name}")

    else:
        st.info("👆 Please upload a `.ply` file to get started.")
        st.markdown("""
        **Don't have a .ply file?** You can use any sample from the `Dataset/feasible/`
        or `Dataset/infeasible/` folders.
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET OVERVIEW PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    if not os.path.exists(FEATURE_CSV):
        st.error(f"Dataset CSV not found: `{FEATURE_CSV}`")
        st.stop()

    df        = pd.read_csv(FEATURE_CSV)
    feat_cols = [c for c in df.columns if c not in ['label', 'file']]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples",  len(df))
    col2.metric("Feasible (1)",   (df['label']==1).sum())
    col3.metric("Infeasible (0)", (df['label']==0).sum())
    col4.metric("Features",       len(feat_cols))

    st.markdown("---")

    # Class distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        counts = df['label'].value_counts().reset_index()
        counts.columns = ['Label', 'Count']
        counts['Class'] = counts['Label'].map({1: 'Feasible', 0: 'Infeasible'})
        fig_pie = px.pie(counts, values='Count', names='Class',
                          color='Class',
                          color_discrete_map={'Feasible':'#3D9970',
                                              'Infeasible':'#E84855'},
                          title='Class Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Feature Statistics")
        st.dataframe(df[feat_cols].describe().round(4), use_container_width=True)

    # Feature distributions
    st.markdown("---")
    st.subheader("Feature Distributions by Class")
    feat_choice = st.selectbox("Select feature to visualize:", feat_cols)

    fig_hist = go.Figure()
    for lbl, color, name in [(0, '#E84855', 'Infeasible'), (1, '#3D9970', 'Feasible')]:
        vals = df[df['label']==lbl][feat_choice]
        fig_hist.add_trace(go.Histogram(
            x=vals, name=name, opacity=0.65,
            marker_color=color, nbinsx=30,
        ))
    fig_hist.update_layout(
        barmode='overlay',
        title=f'Distribution of {feat_choice} by Class',
        xaxis_title=feat_choice, yaxis_title='Count',
        legend_title='Class',
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    st.markdown("---")
    st.subheader("Feature Correlation Matrix")
    corr = df[feat_cols].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1,
                          title='Feature Correlation Matrix',
                          aspect='auto')
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE COMPARISON PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Pipeline Comparison":
    st.title("🔬 Pipeline Comparison")

    result_csv = os.path.join(BASE_DIR, "outputs", "q4", "pipeline_results.csv")

    if os.path.exists(result_csv):
        results_df = pd.read_csv(result_csv)
        results_df = results_df.sort_values('f1', ascending=False)

        st.subheader("📈 F1 Score Comparison")
        fig_f1 = px.bar(
            results_df, x='id', y='f1',
            color='f1', color_continuous_scale='RdYlGn',
            title='F1 Score by Pipeline (Test Set)',
            labels={'id': 'Pipeline', 'f1': 'F1 Score'},
            text=results_df['f1'].round(4),
        )
        fig_f1.update_traces(textposition='outside')
        fig_f1.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 All Metrics")
        metrics_cols = ['id', 'feat_set', 'sampling', 'classifier',
                        'f1', 'accuracy', 'precision', 'recall', 'auc']
        available = [c for c in metrics_cols if c in results_df.columns]
        st.dataframe(
            results_df[available].style.background_gradient(
                subset=['f1', 'accuracy'], cmap='RdYlGn'),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🎯 Multi-Metric Radar Chart")
        best_row = results_df.iloc[0]
        base_row = results_df[results_df['id']=='P001'].iloc[0]

        categories = ['F1', 'Accuracy', 'Precision', 'Recall']
        fig_radar  = go.Figure()
        for row, name, color in [
            (best_row, f"Best ({best_row['id']})", '#3D9970'),
            (base_row, 'Baseline (P001)',           '#E84855'),
        ]:
            vals = [row['f1'], row['accuracy'], row['precision'], row['recall']]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill='toself', name=name,
                line_color=color,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Best Pipeline vs Baseline',
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 Best Pipeline", best_row['id'])
            st.metric("Best F1", f"{best_row['f1']:.4f}")
        with col2:
            st.metric("📌 Baseline (P001)", "Logistic Regression")
            st.metric("Baseline F1", f"{base_row['f1']:.4f}")
            st.metric("Improvement", f"+{best_row['f1']-base_row['f1']:.4f}")

    else:
        st.warning(f"⚠️ Results not found at `{result_csv}`.\n"
                   "Please run `python code/q4_pipelines.py` first.")
        st.info("The comparison table will appear here after running Q4.")


# ─────────────────────────────────────────────────────────────────────────────
#  ABOUT PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Manufacturing Part Feasibility Classifier



    ### 📚 Methods Used

    | Question | Methods |
    |----------|---------|
    | Q1 | PCA, t-SNE, K-Means, DBSCAN, SOR Outlier Removal |
    | Q2 | Sequential Sampling, Bootstrap, Stratified Sampling, Class Imbalance |
    | Q3 | Hand-crafted features, PCA, Autoencoder, K-Means dist, GMM |
    | Q4 | 50+ Pipeline Design, XGBoost, RF, SVM, MLP, SHAP Diagnosis |
    | Q5 | Streamlit App, GitHub, Generative AI |

    ### 🔧 Pipeline System

    **50+ pipelines** designed from combinations of:
    - **5 feature sets**: Original / +Geometric / +PCA / +Unsupervised / All
    - **3 sampling strategies**: None / Stratified+Bootstrap / Class Weight
    - **19 classifiers**: LR, RF, XGBoost, SVM, GBM, MLP, KNN, NB, DT, ...

    ### 📖 References

    1. Rusu & Cousins (2011). 3D is here: Point Cloud Library. *ICRA*
    2. Weinmann et al. (2015). Semantic point cloud interpretation. *ISPRS*
    3. Hinton & Salakhutdinov (2006). Reducing dimensionality. *Science*
    4. Lundberg & Lee (2017). SHAP unified model interpretation. *NeurIPS*

    ### 🤖 AI Assistance

    Code, README, and requirements generated with assistance from **Claude AI (Anthropic)**.

    ---
    **Virginia Tech** | ISE 5764 Applied Regression Analysis
    """)
