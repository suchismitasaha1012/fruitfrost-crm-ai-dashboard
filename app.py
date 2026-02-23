import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="FruitFrost CRM AI Engine", layout="wide")

st.title("🍧 FruitFrost CRM AI Engine — Retention & Revenue Optimiser")
st.caption("B2B outlets | Churn risk × LTV × ROI-based intervention recommendations")

@st.cache_data
def load_data():
    df = pd.read_csv("fruitfrost_crm_action_table.csv")
    # Ensure numeric columns are numeric
    num_cols = ["revenue_last_30d","rev_trend_3m","complaints_last_60d","avg_payment_delay_days",
                "p_churn","pred_ltv_12m","revenue_at_risk","RPI","roi","expected_value_saved","expected_cost"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")

outlet_types = sorted(df["outlet_type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect("Outlet Type", outlet_types, default=outlet_types)

city_tiers = sorted(df["city_tier"].dropna().unique().tolist())
selected_tiers = st.sidebar.multiselect("City Tier", city_tiers, default=city_tiers)

p_min, p_max = float(df["p_churn"].min()), float(df["p_churn"].max())
p_range = st.sidebar.slider("Churn Probability Range", 0.0, 1.0, (max(0.0, p_min), min(1.0, p_max)))

rpi_min, rpi_max = float(df["RPI"].min()), float(df["RPI"].max())
rpi_range = st.sidebar.slider("RPI Range", float(rpi_min), float(rpi_max), (float(rpi_min), float(rpi_max)))

top_n = st.sidebar.slider("Show Top N Priority Accounts", 10, 200, 50)

filtered = df[
    (df["outlet_type"].isin(selected_types)) &
    (df["city_tier"].isin(selected_tiers)) &
    (df["p_churn"] >= p_range[0]) & (df["p_churn"] <= p_range[1]) &
    (df["RPI"] >= rpi_range[0]) & (df["RPI"] <= rpi_range[1])
].copy()

filtered = filtered.sort_values("RPI", ascending=False).head(top_n)

# ----------------------------
# KPI Row
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

total_accounts = len(filtered)
total_rev_risk = filtered["revenue_at_risk"].sum()
avg_churn = filtered["p_churn"].mean() if total_accounts > 0 else 0
total_value_saved = filtered["expected_value_saved"].sum()

col1.metric("Accounts in View", f"{total_accounts}")
col2.metric("Total Revenue at Risk (₹)", f"{total_rev_risk:,.0f}")
col3.metric("Avg Churn Probability", f"{avg_churn:.2f}")
col4.metric("Total Expected Value Saved (₹)", f"{total_value_saved:,.0f}")

st.divider()

# ----------------------------
# Charts Row
# ----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Churn vs LTV (Bubble = RPI)")
    fig = px.scatter(
        filtered,
        x="p_churn",
        y="pred_ltv_12m",
        size="RPI",
        hover_data=["outlet_id","outlet_type","city_tier","action","roi"],
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Recommended Intervention Mix")
    mix = filtered["action"].value_counts().reset_index()
    mix.columns = ["action","count"]
    fig2 = px.bar(mix, x="action", y="count")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ----------------------------
# Priority Table
# ----------------------------
st.subheader("Top Priority Accounts (Revenue-Aware Retention Queue)")
show_cols = [
    "outlet_id","outlet_type","city_tier",
    "revenue_last_30d","rev_trend_3m","complaints_last_60d","avg_payment_delay_days",
    "p_churn","pred_ltv_12m","revenue_at_risk","RPI",
    "action","roi","expected_value_saved","expected_cost"
]

# Nice formatting
display_df = filtered[show_cols].copy()
st.dataframe(display_df, use_container_width=True)

# Download button
csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered action table (CSV)", csv, "fruitfrost_filtered_action_table.csv", "text/csv")
