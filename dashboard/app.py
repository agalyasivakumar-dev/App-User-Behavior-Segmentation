"""
App User Behavior Segmentation – Streamlit Dashboard
Run from the project root:   streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="User Segmentation", page_icon="📊", layout="wide")

PALETTE = {"High Engagement Users": "#2c6e49", "Moderate Engagement Users": "#2a6f97",
           "Low Engagement / At-Risk Users": "#c1121f", "Occasional Users": "#e09f3e"}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "segmentation_model.joblib")


@st.cache_data
def load_users():
    return pd.read_csv(ROOT / "outputs" / "clustered_users.csv")


art = load_model()
df = load_users()
SEG_ORDER = [art["segments"][i] for i in range(4)]


def score_user(sessions, duration, daily_min, engagement, churn, days):
    """Engineer the activity index exactly as in training, scale, predict."""
    s = art["activity_stats"]
    z = lambda v, k: (v - s[k][0]) / s[k][1]
    activity = (z(sessions, "sessions_per_week") + z(daily_min, "daily_active_minutes")
                + z(np.log1p(duration), "log_session_duration")) / 3
    row = pd.DataFrame([[engagement, churn, days, activity]], columns=art["features"])
    raw = int(art["kmeans"].predict(art["scaler"].transform(row))[0])
    return art["order_map"][raw]


# ---------------- Sidebar ----------------
st.sidebar.title("📊 User Segmentation")
page = st.sidebar.radio("Go to", ["Overview", "Segment Profiles", "User Lookup & Lists", "Score a New User"])
st.sidebar.markdown("---")
seg_filter = st.sidebar.multiselect("Segments", SEG_ORDER, default=SEG_ORDER)
sub_filter = st.sidebar.multiselect("Subscription", ["Free", "Basic", "Premium"], default=["Free", "Basic", "Premium"])
dev_filter = st.sidebar.multiselect("Device", sorted(df.device_type.unique()), default=sorted(df.device_type.unique()))
f = df[df.segment.isin(seg_filter) & df.subscription_type.isin(sub_filter) & df.device_type.isin(dev_filter)]

# ---------------- Overview ----------------
if page == "Overview":
    st.title("App User Behavior Segmentation")
    st.caption("K-Means (k=4) on engagement, churn risk, recency and activity · 50,000 users")
    c = st.columns(5)
    c[0].metric("Users (filtered)", f"{len(f):,}")
    c[1].metric("Avg engagement", f"{f.engagement_score.mean():.1f}")
    c[2].metric("Avg churn risk", f"{f.churn_risk_score.mean():.2f}")
    c[3].metric("At-Risk users", f"{(f.cluster == 2).sum():,}")
    c[4].metric("Upsell targets", f"{((f.cluster == 0) & (f.subscription_type == 'Free')).sum():,}",
                help="High-engagement users still on the Free plan")

    l, r = st.columns([1, 1.4])
    counts = f.segment.value_counts().reindex(SEG_ORDER).dropna().reset_index()
    counts.columns = ["segment", "users"]
    l.plotly_chart(px.pie(counts, names="segment", values="users", hole=.5, color="segment",
                          color_discrete_map=PALETTE, title="Users per segment"), use_container_width=True)
    samp = f.sample(min(6000, len(f)), random_state=1)
    r.plotly_chart(px.scatter(samp, x="pc1", y="pc2", color="segment", color_discrete_map=PALETTE, opacity=.55,
                              hover_data=["user_id", "engagement_score", "churn_risk_score"],
                              title="Segments in PCA space").update_traces(marker_size=4),
                   use_container_width=True)
    st.plotly_chart(px.scatter(samp, x="engagement_score", y="churn_risk_score", color="segment",
                               color_discrete_map=PALETTE, opacity=.5, title="Engagement vs churn risk")
                    .update_traces(marker_size=4), use_container_width=True)

# ---------------- Profiles ----------------
elif page == "Segment Profiles":
    st.title("Segment Profiles & Actions")
    cols = ["engagement_score", "churn_risk_score", "days_since_last_login", "sessions_per_week",
            "avg_session_duration_min", "daily_active_minutes", "feature_clicks_per_session",
            "content_downloads", "social_shares", "rating_given"]
    prof = f.groupby("segment")[cols].mean().reindex([s for s in SEG_ORDER if s in f.segment.unique()])
    st.dataframe(prof.round(2).T, use_container_width=True)
    metric = st.selectbox("Compare a metric", cols)
    st.plotly_chart(px.box(f, x="segment", y=metric, color="segment", color_discrete_map=PALETTE,
                           category_orders={"segment": SEG_ORDER}), use_container_width=True)
    mix = st.selectbox("Mix by", ["subscription_type", "device_type", "country", "marketing_source", "gender"])
    ct = (pd.crosstab(f.segment, f[mix], normalize="index") * 100).round(1).reset_index().melt("segment")
    st.plotly_chart(px.bar(ct, x="value", y="segment", color=mix, orientation="h",
                           labels={"value": "% of segment"}), use_container_width=True)
    st.subheader("Business action map")
    for i in range(4):
        name, tactics = art["actions"][i]
        st.markdown(f"**Cluster {i} – {art['segments'][i]}** · priority *{art['priority'][i]}*  \n"
                    f"🎯 {name}: {tactics}")

# ---------------- Lookup & lists ----------------
elif page == "User Lookup & Lists":
    st.title("Customer-Level Identification")
    uid = st.number_input("Look up a user_id", min_value=int(df.user_id.min()),
                          max_value=int(df.user_id.max()), value=int(df.user_id.min()))
    u = df[df.user_id == uid]
    if not u.empty:
        r = u.iloc[0]
        st.success(f"User {uid} → **{r.segment}** (cluster {r.cluster}) · action: **{r.business_action}** "
                   f"· priority: {r.action_priority}")
        st.dataframe(u.T.rename(columns={u.index[0]: "value"}), use_container_width=True)
    st.markdown("---")
    seg = st.selectbox("Download user list for segment", SEG_ORDER)
    out = f[f.segment == seg].sort_values("churn_risk_score", ascending=False)
    st.write(f"{len(out):,} users")
    st.dataframe(out[["user_id", "subscription_type", "engagement_score", "churn_risk_score",
                      "days_since_last_login", "business_action"]].head(500), use_container_width=True)
    st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode(),
                       file_name=f"{seg.replace(' ', '_').replace('/', '-')}.csv", mime="text/csv")

# ---------------- Score new user ----------------
else:
    st.title("Score a New User")
    a, b = st.columns(2)
    sessions = a.slider("Sessions per week", 0, 30, 8)
    duration = a.slider("Avg session duration (min)", 0.5, 90.0, 12.0)
    daily = a.slider("Daily active minutes", 0.0, 130.0, 45.0)
    engagement = b.slider("Engagement score", 0.0, 100.0, 65.0)
    churn = b.slider("Churn risk score", 0.0, 1.0, 0.5)
    days = b.slider("Days since last login", 0, 60, 20)
    c = score_user(sessions, duration, daily, engagement, churn, days)
    name, tactics = art["actions"][c]
    st.markdown(f"### Segment: <span style='color:{PALETTE[art['segments'][c]]}'>{art['segments'][c]}</span>",
                unsafe_allow_html=True)
    st.info(f"**Recommended action – {name}** ({art['priority'][c]} priority)\n\n{tactics}")
