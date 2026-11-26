# 📊 App User Behavior Segmentation Using Unsupervised Machine Learning

Segment 50,000 app users into behavioural groups with **K-Means** (no labels), find high-engagement and at-risk users, and map every user to a business action.

## Project structure

```
App_User_Behavior_Segmentation/
├── data/app_user_behavior_dataset.csv        # raw data (50,000 × 25)
├── notebooks/App_User_Behavior_Segmentation.ipynb   # full analysis, executed with outputs
├── notebooks/App_User_Behavior_Segmentation.html    # same notebook, viewable in a browser
├── src/segmentation_pipeline.py              # script version – reproduces every output
├── app/app.py                                # Streamlit dashboard
├── models/segmentation_model.joblib          # scaler + K-Means + PCA + label/action maps
├── figures/                                  # 10 charts (EDA, elbow, PCA, profiles)
├── outputs/clustered_users.csv               # every user with cluster, segment, action
├── outputs/cluster_user_lists.xlsx           # summary, profile, one sheet per cluster, upsell list
├── outputs/segment_summary.csv
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python src/segmentation_pipeline.py      # rebuilds model, figures and outputs (~40 s)
streamlit run app/app.py                 # dashboard: overview, profiles, user lookup, score a new user
```

## Approach

1. **Data understanding** – 50,000 users, 25 columns (demographics, device, usage, engagement, churn risk). No duplicate users.
2. **Cleaning** – `rating_given` had 5,022 missing values (10%) → median imputation + `rating_missing` flag. Categorical text normalised; range checks passed (no negatives, scores within 0–1 / 0–100). Outliers kept (real heavy users); skewed `avg_session_duration_min` log-transformed.
3. **EDA** – distributions, categorical mix, correlation matrix, bivariate plots. Key finding: features are almost uncorrelated (max |r| ≈ 0.01).
4. **Feature engineering** – `activity_index` (sessions × daily minutes × log session length), `interaction_index`, `friction_index`, `weekly_active_minutes`.
5. **Feature selection** – compared three feature sets at k = 4:

   | Feature set | Silhouette ↑ | Davies–Bouldin ↓ |
   |---|---|---|
   | All 18 numeric metrics | 0.042 | 3.79 |
   | 8 core behaviour metrics | 0.086 | 2.57 |
   | **Selected 4: engagement, churn risk, recency, activity index** | **0.184** | **1.53** |

6. **Scaling** – `StandardScaler` (mean 0, std 1).
7. **Optimal k** – Elbow (inertia) + Silhouette + Davies–Bouldin + Calinski–Harabasz over k = 2–10 → **k = 4** (elbow bend, DB improves sharply to 4 then flattens, matches business needs).
8. **Model** – K-Means (k = 4, n_init = 20, random_state = 42); fits 50k users in < 1 s.
9. **PCA** – 2-D projection to visualise segments + loadings to explain the axes.
10. **Profiling & action mapping** – per-cluster means, relative-profile heatmap, plan/device/channel mix, user lists.

## Results

| Cluster | Segment | Users | Engagement | Churn risk | Days since login | Action | Priority |
|---|---|---|---|---|---|---|---|
| 0 | High Engagement Users | 12,719 (25.4%) | 71.6 | 0.27 | 32.8 | Loyalty & premium upsell | Nurture |
| 1 | Moderate Engagement Users | 12,597 (25.2%) | 58.2 | 0.28 | 11.0 | Grow engagement | Medium |
| 2 | Low Engagement / At-Risk Users | 12,279 (24.6%) | 55.3 | 0.73 | 31.4 | Retention / win-back | High (urgent) |
| 3 | Occasional Users | 12,405 (24.8%) | 74.5 | 0.73 | 13.0 | Build habit | High |

**Business highlights**
- ~50% of users (At-Risk + Occasional) carry high churn risk → retention is the biggest lever.
- **7,744** High-Engagement users are still on the Free plan → ready Premium upsell list.
- **1,825** Premium subscribers sit in the At-Risk segment → revenue at risk, contact first.
- Every `user_id` carries a segment, recommended tactics and priority, enabling customer-level targeting.

## Honest note on cluster quality

The dataset's metrics are statistically independent, so users form a continuum rather than naturally separated groups — no algorithm can produce very high silhouette scores on it. Feature selection lifted silhouette 4× (0.04 → 0.18) and the segments differ strongly on the KPIs that drive decisions (engagement ±0.5 SD, churn ±0.8 SD), which makes them reliable and actionable as a reproducible partition of the behaviour space.

## Scalability

K-Means: 50k rows ≈ 0.2–0.7 s, 200k rows ≈ 1.2 s. MiniBatchKMeans is 5–7× faster for millions of rows. New users are scored with `scaler.transform` + `kmeans.predict` (no retraining), as used in the Streamlit "Score a New User" page.

## Next steps

Monthly refresh of segments · A/B test the playbook per segment · push `clustered_users.csv` into CRM / BI dashboards · track segment migration over time.
