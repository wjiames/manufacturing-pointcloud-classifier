# Manufacturing Part Feasibility Classifier
### 3D Point Cloud Based ML Pipeline System

---

## 📌 Project Overview

This project builds a **complete machine learning pipeline system** for classifying 3D point cloud data of manufacturing parts as **feasible** or **infeasible** designs.

The dataset contains **500 manufacturing parts** (300 feasible, 200 infeasible), represented as 3D point clouds in `.ply` format. Features are extracted from the raw point clouds and used to train multiple classification models.

---

## 🗂️ Project Structure

```
assignment/
├── code/
│   ├── step1_extract_features.py   # Step 1: Extract features from .ply files
│   ├── q1_analysis.py              # Q1: Visualization + PCA + t-SNE + Clustering
│   ├── q2_sampling.py              # Q2: Sequential sampling + class imbalance
│   ├── q3_feature_engineering.py  # Q3: Advanced feature engineering
│   ├── q4_pipelines.py            # Q4: 50+ pipelines + misclassification diagnosis
│   └── app.py                     # Q5: Streamlit online application
├── Dataset/
│   ├── feasible/                  # .ply files for feasible designs
│   ├── infeasible/                # .ply files for infeasible designs
│   └── dataset_features.csv      # Extracted features (auto-generated)
├── outputs/                       # Generated figures and results
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Features from Point Clouds

```bash
python code/step1_extract_features.py
```

### 3. Run Analysis Scripts

```bash
# Q1: Visualization + Dimensionality Reduction + Clustering
python code/q1_analysis.py

# Q2: Sampling Strategy + Class Imbalance Analysis
python code/q2_sampling.py

# Q3: Feature Engineering
python code/q3_feature_engineering.py

# Q4: 50+ Pipelines + Misclassification Diagnosis
python code/q4_pipelines.py
```

### 4. Launch Streamlit App

```bash
streamlit run code/app.py
```

---

## 🔬 Methodology

### Q1 — Data Visualization
- **3D Point Cloud Visualization**: Direct rendering of `.ply` files
- **Outlier Removal**: Statistical Outlier Removal (SOR) using kNN distance
- **Dimensionality Reduction**: PCA (linear) + t-SNE (non-linear)
- **Clustering**: K-Means (elbow + silhouette) + DBSCAN

### Q2 — Sequential Sampling Strategy
| Strategy | Description |
|----------|-------------|
| Random (baseline) | Random subset selection |
| Stratified | Maintain class ratio 50:50 |
| Stratified + Bootstrap | Stratified + resampling with replacement |
| Uncertainty Sampling | Prioritize samples near decision boundary |

### Q3 — Feature Engineering
| Method | Type | Features Added |
|--------|------|---------------|
| Hand-crafted geometric | Supervised design | +15 |
| PCA components | **Unsupervised** | +5~8 |
| Autoencoder latent | **Unsupervised** | +8 |
| K-Means distances | **Unsupervised** | +6 |
| GMM probabilities | **Unsupervised** | +4 |

### Q4 — Pipeline System
- **50+ pipelines** designed: 5 feature sets × 3 sampling strategies × 19 classifiers
- **10 pipelines implemented** and compared by F1 score
- **Misclassification diagnosis**: PCA/t-SNE visualization + SHAP analysis

---

## 📊 Results Summary

| Pipeline | Feature Set | Sampling | Classifier | F1 |
|----------|-------------|----------|------------|----|
| P001 (baseline) | Original | None | Logistic Regression | ~0.85 |
| P009 (best) | All Features | Stratified+Bootstrap | XGBoost | ~0.95 |

---

## 🌐 Online App

The Streamlit app provides:
- **Upload** your own `.ply` point cloud file
- **Select** from 10 pre-trained pipelines
- **View** 3D visualization of the uploaded point cloud
- **Get** feasibility prediction with confidence score
- **Explore** feature values and SHAP explanations

**Live Demo**: [Deploy on Streamlit Cloud](https://streamlit.io/cloud)

---

## 🛠️ Dependencies

- `scikit-learn` — ML models and evaluation
- `xgboost` — Gradient boosting classifier
- `torch` — Autoencoder feature extraction
- `shap` — Model explainability
- `streamlit` — Web application framework
- `plotly` — Interactive visualizations
- `open3d` — Point cloud processing

---

## 📚 References

1. Rusu, R.B. & Cousins, S. (2011). 3D is here: Point Cloud Library. *ICRA*.
2. Weinmann, M. et al. (2015). Semantic point cloud interpretation. *ISPRS*.
3. Hinton, G. & Salakhutdinov, R. (2006). Reducing dimensionality with neural networks. *Science*.
4. Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
5. Coates, A. & Ng, A. (2012). Learning feature representations with K-Means. *ICML*.

---

## 👤 Author

Generated with assistance from Claude AI (Anthropic).  
Course: ISE 5764 / CS 5805 — Virginia Tech
