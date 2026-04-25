"""
Kenya Coffee Sector - Advanced ML Analysis Pipeline
=====================================================
Author: Chege-N / Claude Analysis
Date: April 2026
Description:
    Full end-to-end ML pipeline for Kenya Coffee Growers & Dealers datasets.
    Includes: EDA, Feature Engineering, Clustering (KMeans, DBSCAN),
    Classification (Random Forest, Gradient Boosting, Logistic Regression),
    Anomaly Detection (Isolation Forest), and Forecasting projections.
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ========== CONFIG ==========
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ========== 1. DATA LOADING ==========
print("=" * 60)
print("KENYA COFFEE SECTOR — ML ANALYSIS PIPELINE")
print("=" * 60)

growers = pd.read_csv(os.path.join(DATA_DIR, 'coffeegrowers_raw.csv'), encoding='latin1')
dealers = pd.read_csv(os.path.join(DATA_DIR, 'coffeedealers_raw.csv'), encoding='latin1')
print(f"\n✓ Loaded growers: {growers.shape[0]:,} rows × {growers.shape[1]} cols")
print(f"✓ Loaded dealers: {dealers.shape[0]:,} rows × {dealers.shape[1]} cols")

# ========== 2. FEATURE ENGINEERING — GROWERS ==========
print("\n--- FEATURE ENGINEERING (Growers) ---")

growers['created_dt'] = pd.to_datetime(growers['created'], utc=True, errors='coerce')
growers['updated_dt'] = pd.to_datetime(growers['updated'], utc=True, errors='coerce')
growers['days_since_update'] = (growers['updated_dt'] - growers['created_dt']).dt.days.fillna(-1)

# Binary presence flags
growers['has_geo']          = growers['lat'].notna().astype(int)
growers['has_website']      = growers['website'].notna().astype(int)
growers['has_email']        = growers['email'].notna().astype(int)
growers['has_affiliation']  = growers['affiliation_name'].notna().astype(int)
growers['has_notes']        = growers['notes'].notna().astype(int)
growers['loc_verified_int'] = growers['location_verified'].astype(int)

# Normalize actor labels
actor_norm_map = {'factory': 'Factory', 'estate_small': 'Small Estates', 'Not applicable': 'Other'}
growers['actor_norm'] = growers['actor'].replace(actor_norm_map)

# Completeness score (0–1)
growers['completeness'] = (
    growers['has_geo'] + growers['has_website'] + growers['has_email'] +
    growers['has_affiliation'] + growers['loc_verified_int']
) / 5.0

# Label encode actor
le = LabelEncoder()
growers['actor_enc'] = le.fit_transform(growers['actor_norm'])

print(f"✓ Feature engineering complete. Columns: {growers.shape[1]}")
print(f"  Active growers: {growers['active'].sum():,} / {len(growers):,} ({growers['active'].mean()*100:.1f}%)")
print(f"  With geo coords: {growers['has_geo'].sum():,} / {len(growers):,} ({growers['has_geo'].mean()*100:.1f}%)")

# ========== 3. FEATURE ENGINEERING — DEALERS ==========
print("\n--- FEATURE ENGINEERING (Dealers) ---")
dealers['has_website']      = dealers['website'].notna().astype(int)
dealers['has_nce']          = dealers['nce_ref'].notna().astype(int)
dealers['has_notes']        = dealers['notes'].notna().astype(int)
dealers['completeness']     = (dealers['has_website'] + dealers['has_nce'] + dealers['has_notes']) / 3.0
print(f"✓ Dealers with website: {dealers['has_website'].sum()} / {len(dealers)} ({dealers['has_website'].mean()*100:.1f}%)")
print(f"✓ Avg dealer completeness: {dealers['completeness'].mean():.2f}")

# ========== 4. CLUSTERING — KMeans ==========
print("\n--- CLUSTERING: KMeans (k=4) ---")

feature_cols = ['actor_enc', 'has_geo', 'has_website', 'has_email',
                'has_affiliation', 'loc_verified_int', 'completeness']
X_cluster = growers[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
growers['cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = growers.groupby('cluster').agg(
    size=('cluster', 'count'),
    active_rate=('active', lambda x: x.astype(int).mean()),
    geo_rate=('has_geo', 'mean'),
    completeness_mean=('completeness', 'mean'),
    dominant_actor=('actor_norm', lambda x: x.mode()[0])
).reset_index()
print(cluster_summary.to_string(index=False))

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
growers['pca1'] = pca_coords[:, 0]
growers['pca2'] = pca_coords[:, 1]
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ========== 5. CLASSIFICATION — Activity Prediction ==========
print("\n--- CLASSIFICATION: Predicting Grower Active Status ---")

clf_features = ['actor_enc', 'has_geo', 'has_website', 'has_email',
                'has_affiliation', 'loc_verified_int', 'completeness', 'days_since_update']
X_clf = growers[clf_features].fillna(-1)
y_clf = growers['active'].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

results = {}
for name, model in models.items():
    f1_scores = cross_val_score(model, X_clf, y_clf, cv=cv, scoring='f1')
    auc_scores = cross_val_score(model, X_clf, y_clf, cv=cv, scoring='roc_auc')
    results[name] = {'f1_mean': f1_scores.mean(), 'f1_std': f1_scores.std(), 'auc_mean': auc_scores.mean()}
    print(f"  {name:25s} F1={f1_scores.mean():.3f}±{f1_scores.std():.3f}  AUC={auc_scores.mean():.3f}")

# Fit best model for predictions
best_model = models['Random Forest']
best_model.fit(X_clf, y_clf)
growers['active_prob'] = best_model.predict_proba(X_clf)[:, 1]

feat_imp = pd.Series(best_model.feature_importances_, index=clf_features).sort_values(ascending=False)
print("\n  Feature Importance (Random Forest):")
for f, v in feat_imp.items():
    bar = '█' * int(v * 50)
    print(f"    {f:30s} {bar} {v*100:.1f}%")

# ========== 6. ANOMALY DETECTION ==========
print("\n--- ANOMALY DETECTION: Isolation Forest ---")
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
growers['anomaly'] = iso_forest.fit_predict(X_scaled)
growers['anomaly_score'] = iso_forest.decision_function(X_scaled)
n_anomalies = (growers['anomaly'] == -1).sum()
print(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(growers)*100:.1f}%)")
print(f"  Anomaly score range: [{growers['anomaly_score'].min():.3f}, {growers['anomaly_score'].max():.3f}]")

# Anomaly breakdown by actor
anom_by_actor = growers[growers['anomaly'] == -1]['actor_norm'].value_counts()
print("  Anomalies by actor type:")
print(anom_by_actor.to_string())

# ========== 7. TIME SERIES & PROJECTIONS ==========
print("\n--- TIME SERIES ANALYSIS & PROJECTION ---")

growers['year_month'] = growers['created_dt'].dt.to_period('M')
ts = growers.groupby('year_month').size().reset_index(name='count')
ts['period_str'] = ts['year_month'].astype(str)

# Exclude bulk import (April 2018)
ts_monthly = ts[ts['period_str'] >= '2018-05'].copy()
ts_monthly['t'] = range(len(ts_monthly))

lr = LinearRegression()
lr.fit(ts_monthly[['t']], ts_monthly['count'])
slope = lr.coef_[0]
intercept = lr.intercept_
print(f"  Linear trend: {slope:.3f} registrations/month (slope)")
print(f"  R² = {lr.score(ts_monthly[['t']], ts_monthly['count']):.3f}")

# Project 12 months
last_t = len(ts_monthly)
proj_months = pd.period_range(start='2020-10', periods=12, freq='M')
proj_baseline = [max(0, slope * (last_t + i) + intercept) for i in range(1, 13)]
proj_optimistic = [v * 1.2 + i * 0.3 for i, v in enumerate(proj_baseline)]
proj_pessimistic = [max(0, v * 0.75 - i * 0.1) for i, v in enumerate(proj_baseline)]

print("\n  12-Month Projection (Baseline):")
for m, v in zip(proj_months, proj_baseline):
    print(f"    {m}: {v:.1f} registrations expected")

# ========== 8. COUNTY ANALYSIS ==========
print("\n--- COUNTY ANALYSIS ---")
county_id_map = {
    14.0: 'Kiambu', 30.0: 'Nyeri', 37.0: 'Muranga', 27.0: 'Nakuru',
    13.0: 'Kirinyaga', 16.0: 'Meru', 17.0: 'Tharaka-Nithi',
    23.0: 'Embu', 7.0: 'Kisii', 42.0: 'Nyandarua'
}
growers['county_name'] = growers['county_name_id'].map(county_id_map).fillna('Unknown')
county_stats = growers.groupby('county_name').agg(
    total=('id', 'count'),
    active=('active', 'sum'),
    geo_count=('has_geo', 'sum'),
    active_rate=('active', lambda x: x.astype(int).mean() * 100)
).sort_values('total', ascending=False).head(10)
print(county_stats.to_string())

# ========== 9. SAVE OUTPUTS ==========
print("\n--- SAVING OUTPUTS ---")
growers.to_csv(os.path.join(DATA_DIR, 'growers_ml_output.csv'), index=False)
dealers.to_csv(os.path.join(DATA_DIR, 'dealers_ml_output.csv'), index=False)

# Save summary JSON
summary = {
    'dataset': {
        'total_growers': int(len(growers)),
        'total_dealers': int(len(dealers)),
        'active_growers': int(growers['active'].sum()),
        'geo_referenced': int(growers['has_geo'].sum()),
        'data_date_range': {'start': '2018-04-28', 'end': '2020-09-18'}
    },
    'ml_results': {
        'rf_f1': float(results['Random Forest']['f1_mean']),
        'rf_auc': float(results['Random Forest']['auc_mean']),
        'gb_f1': float(results['Gradient Boosting']['f1_mean']),
        'n_clusters': 4,
        'n_anomalies': int(n_anomalies),
        'top_feature': str(feat_imp.index[0]),
        'top_feature_importance': float(feat_imp.iloc[0])
    },
    'projections': {
        'months': [str(m) for m in proj_months],
        'baseline': [round(v, 1) for v in proj_baseline],
        'optimistic': [round(v, 1) for v in proj_optimistic],
        'pessimistic': [round(v, 1) for v in proj_pessimistic]
    }
}
with open(os.path.join(OUTPUT_DIR, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Saved growers_ml_output.csv")
print("✓ Saved dealers_ml_output.csv")
print("✓ Saved analysis_summary.json")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
