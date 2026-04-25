# Kenya Coffee Sector — Advanced ML Analysis Project

**Author:** Chege-N | 
**Dataset:** Kenya Coffee Growers (2,784) + Dealers (157) | **Period:** 2018–2020

---

## Project Structure

```
kenya_coffee_project/
├── data/
│   ├── coffeedealers_raw.csv        # Original dealer data (157 records)
│   ├── coffeegrowers_raw.csv        # Original grower data (2,784 records)
│   ├── growers_processed.csv        # Feature-engineered growers
│   ├── dealers_processed.csv        # Feature-engineered dealers
│   └── geo_data.json                # 897 georeferenced growers (JSON)
├── analysis/
│   └── kenya_coffee_analysis.py     # Full ML pipeline script
├── models/
│   └── (model artifacts if saved)
└── outputs/
    ├── kenya_coffee_dashboard.html  # Interactive dashboard (open in browser)
    └── analysis_summary.json        # Machine-readable results summary
```

---

## Datasets

### Coffee Growers (`coffeegrowers.csv`)
- **2,784 records**, 25 columns
- Date range: 2018-04-28 → 2020-09-18
- Key fields: `id`, `title`, `actor`, `lat`, `lon`, `county_name_id`, `active`, `created`, `updated`
- Actor types: Factory (983), Small Estates (984), Cooperative Society (494), Estate Producers (320)

### Coffee Dealers (`coffeedealers.csv`)
- **157 records**, 7 columns
- Key fields: `id`, `title`, `license_number`, `nce_ref`, `notes`, `website`

---

## ML Models Applied

| Model | Purpose | Key Metric |
|-------|---------|-----------|
| **Random Forest** (n=200) | Predict grower active status | F1=0.630 (5-fold CV) |
| **Gradient Boosting** | Compare classifier | F1=0.580 |
| **Logistic Regression** | Baseline classifier | F1=0.410 |
| **KMeans** (k=4) | Grower segmentation | 4 distinct clusters |
| **Isolation Forest** | Anomaly detection | 136 anomalies (4.9%) |
| **PCA** (2D) | Dimensionality reduction | 68.2% variance explained |
| **Linear Regression** | Registration trend + projection | R²=0.21 |

---

## Key Findings

### Growers
1. **Only 10.8% of growers are active** (301/2,784). Small Estates have highest activation at 31.3%.
2. **67.8% lack geo coordinates** — major gap for supply chain traceability.
3. **Actor type is the single strongest predictor** of activation (84.4% Random Forest feature importance).
4. **4 clusters identified:**
   - C0 (n=1,754): No geo data, 16.6% active — large dormant base
   - C1 (n=1,025): Geo-mapped, 0.9% active — high re-engagement potential
   - C2/C3: Micro-clusters of well-documented operators
5. **136 anomalous growers** flagged by Isolation Forest for manual review.

### Dealers
1. Only **18.5% have a website** — significant digital presence gap.
2. Dealers with websites show **3.1× higher data completeness** (0.58 vs 0.19).
3. **31 dealers** are documented Coffee Warehousemen — indicating vertical integration.
4. **3 missing license numbers** require regulatory follow-up.

### Projections
- **Baseline:** ~10–11 new grower registrations/month through 2021
- **Optimistic** (+20% stimulus): 12–13/month
- **Pessimistic** (-25%): 7–8/month
- Long-term: Estimated 3,000 growers by end of 2022 (baseline scenario)

---

## Strategic Recommendations

1. **Geo-tagging campaign** — Target Kiambu, Nyeri & Muranga (45% of registry) first.
2. **Reactivation initiative** — ML Cluster 1 (1,025 geo-mapped inactive growers) is highest-ROI.
3. **Dealer digitisation fund** — 81.5% of dealers are offline; digital presence correlates with quality.
4. **Supply chain linkage** — Build grower-dealer matching model using county + license data.
5. **Data quality investment** — Improving completeness from 20% → 60% estimated to boost model F1 to 0.82+.

---

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost
python analysis/kenya_coffee_analysis.py
```

Open `outputs/kenya_coffee_dashboard.html` in any modern browser for the interactive dashboard.

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards, actor distribution, registration trend, completeness |
| **Grower Analysis** | County breakdown, active rates, location accuracy, completeness profiles |
| **Dealer Analysis** | Role categories, digital presence, key insights |
| **Geo Intelligence** | Canvas map of 897 georeferenced growers across Kenya coffee belt |
| **ML Models** | Feature importance, cluster profiles, model comparison radar chart |
| **Projections** | 3-scenario 12-month forecast with table and strategy recommendations |
