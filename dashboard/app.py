
"""
App User Behavior Segmentation – Streamlit Dashboard
Run from the project root: streamlit run dashboard/app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="User Segmentation",
    page_icon="📊",
    layout="wide"
)

PALETTE = {
    "High Engagement Users": "#2c6e49",
    "Moderate Engagement Users": "#2a6f97",
    "Low Engagement / At-Risk Users": "#c1121f",
    "Occasional Users": "#e09f3e"
}


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
    """Engineer activity index and predict the user segment."""

    s = art["activity_stats"]

    z = lambda v, k: (v - s[k][0]) / s[k][1]

    activity = (
        z(sessions, "sessions_per_week")
        + z(daily_min, "daily_active_minutes")
        + z(np.log1p(duration), "log_session_duration")
    ) / 3

    row = pd.DataFrame(
        [[engagement, churn, days, activity]],
        columns=art["features"]
    )

    raw = int(
        art["kmeans"].predict(
            art["scaler"].transform(row)
        )[0]
    )

    return art["order_map"][raw]


# ---------------- Main Dashboard ----------------

st.title("📊 App User Behavior Segmentation")

st.caption(
    "K-Means (k=4) on engagement, churn risk, "
    "recency and activity · 50,000 users"
)


# ---------------- Create Separate Tabs ----------------

overview_tab, profiles_tab, lookup_tab, scoring_tab = st.tabs(
    [
        "Overview",
        "Segment Profiles",
        "User Lookup & Lists",
        "Score a New User"
    ]
)


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview_tab:

    st.subheader("Overall User Overview")

    # Key Metrics
    c = st.columns(5)

    c[0].metric(
        "Total Users",
        f"{len(df):,}"
    )

    c[1].metric(
        "Avg Engagement",
        f"{df.engagement_score.mean():.1f}"
    )

    c[2].metric(
        "Avg Churn Risk",
        f"{df.churn_risk_score.mean():.2f}"
    )

    c[3].metric(
        "At-Risk Users",
        f"{(df.cluster == 2).sum():,}"
    )

    c[4].metric(
        "Upsell Targets",
        f"{((df.cluster == 0) & (df.subscription_type == 'Free')).sum():,}",
        help="High-engagement users still on the Free plan"
    )

    # Segment Distribution and PCA
    left, right = st.columns([1, 1.4])

    counts = (
        df.segment.value_counts()
        .reindex(SEG_ORDER)
        .dropna()
        .reset_index()
    )

    counts.columns = ["segment", "users"]

    left.plotly_chart(
        px.pie(
            counts,
            names="segment",
            values="users",
            hole=0.5,
            color="segment",
            color_discrete_map=PALETTE,
            title="Users per Segment"
        ),
        use_container_width=True
    )

    sample = df.sample(
        min(6000, len(df)),
        random_state=1
    )

    right.plotly_chart(
        px.scatter(
            sample,
            x="pc1",
            y="pc2",
            color="segment",
            color_discrete_map=PALETTE,
            opacity=0.55,
            hover_data=[
                "user_id",
                "engagement_score",
                "churn_risk_score"
            ],
            title="Segments in PCA Space"
        ).update_traces(marker_size=4),
        use_container_width=True
    )

    # Engagement vs Churn Risk
    st.plotly_chart(
        px.scatter(
            sample,
            x="engagement_score",
            y="churn_risk_score",
            color="segment",
            color_discrete_map=PALETTE,
            opacity=0.5,
            title="Engagement vs Churn Risk"
        ).update_traces(marker_size=4),
        use_container_width=True
    )


# ==================================================
# TAB 2: SEGMENT PROFILES
# ==================================================

with profiles_tab:

    st.subheader("Segment Profiles & Actions")

    cols = [
        "engagement_score",
        "churn_risk_score",
        "days_since_last_login",
        "sessions_per_week",
        "avg_session_duration_min",
        "daily_active_minutes",
        "feature_clicks_per_session",
        "content_downloads",
        "social_shares",
        "rating_given"
    ]

    # Segment Profile Summary
    prof = (
        df.groupby("segment")[cols]
        .mean()
        .reindex(SEG_ORDER)
    )

    st.dataframe(
        prof.round(2).T,
        use_container_width=True
    )

    # Compare Metrics
    metric = st.selectbox(
        "Compare a Metric",
        cols,
        key="profile_metric"
    )

    st.plotly_chart(
        px.box(
            df,
            x="segment",
            y=metric,
            color="segment",
            color_discrete_map=PALETTE,
            category_orders={"segment": SEG_ORDER}
        ),
        use_container_width=True
    )

    # Segment Composition
    mix = st.selectbox(
        "View Segment Mix By",
        [
            "subscription_type",
            "device_type",
            "country",
            "marketing_source",
            "gender"
        ],
        key="profile_mix"
    )

    ct = (
        pd.crosstab(
            df.segment,
            df[mix],
            normalize="index"
        ) * 100
    ).round(1).reset_index().melt("segment")

    st.plotly_chart(
        px.bar(
            ct,
            x="value",
            y="segment",
            color=mix,
            orientation="h",
            labels={"value": "% of Segment"}
        ),
        use_container_width=True
    )

    # Business Action Map
    st.subheader("Business Action Map")

    for i in range(4):

        name, tactics = art["actions"][i]

        st.markdown(
            f"**Cluster {i} – {art['segments'][i]}** "
            f"· Priority *{art['priority'][i]}*  \n"
            f"🎯 {name}: {tactics}"
        )


# ==================================================
# TAB 3: USER LOOKUP & LISTS
# ==================================================

with lookup_tab:

    st.subheader("Customer-Level Identification")

    # Individual User Lookup
    uid = st.number_input(
        "Look Up a User ID",
        min_value=int(df.user_id.min()),
        max_value=int(df.user_id.max()),
        value=int(df.user_id.min()),
        step=1
    )

    user = df[df.user_id == uid]

    if not user.empty:

        details = user.iloc[0]

        st.success(
            f"User {uid} → **{details.segment}** "
            f"(Cluster {details.cluster}) · "
            f"Action: **{details.business_action}** · "
            f"Priority: {details.action_priority}"
        )

        st.dataframe(
            user.T.rename(
                columns={user.index[0]: "value"}
            ),
            use_container_width=True
        )

    else:

        st.warning("No user found with this ID.")

    st.markdown("---")

    # Download Segment-wise User List
    st.subheader("Download User List")

    segment = st.selectbox(
        "Select a Segment",
        SEG_ORDER,
        key="download_segment"
    )

    user_list = (
        df[df.segment == segment]
        .sort_values(
            "churn_risk_score",
            ascending=False
        )
    )

    st.write(
        f"{len(user_list):,} users in this segment"
    )

    st.dataframe(
        user_list[
            [
                "user_id",
                "subscription_type",
                "engagement_score",
                "churn_risk_score",
                "days_since_last_login",
                "business_action"
            ]
        ].head(500),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download CSV",
        user_list.to_csv(index=False).encode("utf-8"),
        file_name=f"{segment.replace(' ', '_').replace('/', '-')}.csv",
        mime="text/csv"
    )


# ==================================================
# TAB 4: SCORE A NEW USER
# ==================================================

with scoring_tab:

    st.subheader("Score a New User")

    st.write(
        "Enter the user's activity details "
        "to identify their segment."
    )

    left, right = st.columns(2)

    # User Activity Inputs
    sessions = left.slider(
        "Sessions per Week",
        0, 30, 8
    )

    duration = left.slider(
        "Avg Session Duration (min)",
        0.5, 90.0, 12.0
    )

    daily = left.slider(
        "Daily Active Minutes",
        0.0, 130.0, 45.0
    )

    engagement = right.slider(
        "Engagement Score",
        0.0, 100.0, 65.0
    )

    churn = right.slider(
        "Churn Risk Score",
        0.0, 1.0, 0.5
    )

    days = right.slider(
        "Days Since Last Login",
        0, 60, 20
    )

    # Predict Segment
    cluster = score_user(
        sessions,
        duration,
        daily,
        engagement,
        churn,
        days
    )

    segment_name = art["segments"][cluster]

    action_name, tactics = art["actions"][cluster]

    st.markdown(
        f"### Segment: "
        f"<span style='color:{PALETTE[segment_name]}'>"
        f"{segment_name}</span>",
        unsafe_allow_html=True
    )

    st.info(
        f"**Recommended Action – {action_name}** "
        f"({art['priority'][cluster]} Priority)\n\n"
        f"{tactics}"
    )
