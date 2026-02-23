"""
FruitFrost CRM — AI-Driven Retention & Revenue Optimisation Dashboard
Run: streamlit run app.py
Requirements: pip install streamlit pandas numpy scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FruitFrost CRM Intelligence",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme & Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* Root Variables */
:root {
    --navy:    #0D1B2A;
    --navy2:   #1A2D42;
    --blue:    #1E6FD9;
    --blue2:   #4A9FFF;
    --teal:    #00C2CB;
    --green:   #22C55E;
    --amber:   #F59E0B;
    --red:     #EF4444;
    --text:    #E8F0FE;
    --muted:   #94A3B8;
    --card-bg: #162032;
    --border:  #1E3A5F;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}

.main { background-color: var(--navy) !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #162032 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric Cards */
[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--blue), var(--teal)) !important;
    color: white !important;
}

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
.stDataFrame thead th { background: var(--navy2) !important; color: var(--teal) !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }

/* Headers */
h1 { font-size: 1.9rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; color: var(--teal) !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

/* Custom cards */
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value { font-size: 2.1rem; font-weight: 700; margin: 0.3rem 0; }
.kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; }
.kpi-delta { font-size: 0.82rem; margin-top: 0.4rem; }

.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #4A9FFF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.insight-box {
    background: linear-gradient(135deg, #162032, #1A2D42);
    border-left: 3px solid var(--teal);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #CBD5E1;
}

.tier-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Selectbox / Input */
[data-testid="stSelectbox"] > div { background: var(--card-bg) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fruitfrost_outlets_scored.csv")
    return df

@st.cache_resource
def load_models():
    with open("churn_model.pkl","rb") as f:  churn_model = pickle.load(f)
    with open("ltv_model.pkl","rb") as f:    ltv_model   = pickle.load(f)
    with open("feature_list.pkl","rb") as f: features    = pickle.load(f)
    fi_churn = pd.read_csv("feature_importance_churn.csv")
    fi_ltv   = pd.read_csv("feature_importance_ltv.csv")
    return churn_model, ltv_model, features, fi_churn, fi_ltv

df = load_data()
churn_model, ltv_model, features, fi_churn, fi_ltv = load_models()

# ─── Plotly Theme ─────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0D1B2A",
    font=dict(family="DM Sans", color="#E8F0FE", size=12),
    colorway=["#1E6FD9","#00C2CB","#22C55E","#F59E0B","#EF4444","#A78BFA"],
    xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickcolor="#94A3B8"),
    yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickcolor="#94A3B8"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def style_fig(fig, title="", height=380):
    fig.update_layout(**PLOTLY_THEME, title=dict(text=title, font=dict(size=14, color="#E8F0FE")), height=height)
    return fig

COLOR_RISK = {"🔴 Critical": "#EF4444", "🟡 At Risk": "#F59E0B", "🟢 Healthy": "#22C55E"}
COLOR_IV   = {
    "Dedicated Account Manager": "#1E6FD9",
    "5% Discount Offer":          "#F59E0B",
    "Extended Credit Terms":      "#A78BFA",
    "Premium Flavour Trial":      "#00C2CB",
    "No Intervention Needed":     "#22C55E",
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.5rem;'>🧊</div>
        <div style='font-size:1.25rem; font-weight:700; color:#4A9FFF; letter-spacing:0.05em;'>FRUITFROST</div>
        <div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.12em; text-transform:uppercase;'>CRM Intelligence</div>
        <hr style='border-color:#1E3A5F; margin: 0.8rem 0;'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🗂 Navigation**")
    page = st.radio("", [
        "📊 Executive Summary",
        "⚠️ Churn Risk Analysis",
        "💰 LTV & Revenue Intel",
        "🎯 Intervention Engine",
        "🏆 Retention Priority Index",
        "🔮 Predict New Lead"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E3A5F;'>", unsafe_allow_html=True)

    st.markdown("**🔍 Filters**")
    city_filter = st.multiselect("City", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
    type_filter = st.multiselect("Outlet Type", sorted(df["outlet_type"].unique()), default=sorted(df["outlet_type"].unique()))
    risk_filter = st.multiselect("Health Tier", ["🔴 Critical","🟡 At Risk","🟢 Healthy"],
                                  default=["🔴 Critical","🟡 At Risk","🟢 Healthy"])

    st.markdown("<hr style='border-color:#1E3A5F;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#475569; text-align:center; line-height:1.6;'>
    IIM Ranchi · Entrepreneurial Marketing<br>
    AI-Driven CRM Project · Group 2<br>
    <span style='color:#1E6FD9;'>Suchismita Saha · M105-23</span>
    </div>""", unsafe_allow_html=True)

# Apply filters
dff = df[
    df["city"].isin(city_filter) &
    df["outlet_type"].isin(type_filter) &
    df["health_tier"].isin(risk_filter)
].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":

    st.markdown("## 📊 Executive Summary")
    st.markdown(f"<div style='color:#94A3B8; font-size:0.88rem; margin-bottom:1.5rem;'>Showing {len(dff)} of {len(df)} outlets · FruitFrost B2B Portfolio Overview</div>", unsafe_allow_html=True)

    # KPI Row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    total_arr     = dff["annual_revenue_inr"].sum()
    critical_n    = (dff["health_tier"] == "🔴 Critical").sum()
    at_risk_n     = (dff["health_tier"] == "🟡 At Risk").sum()
    rev_at_risk   = dff[dff["health_tier"] == "🔴 Critical"]["annual_revenue_inr"].sum()
    total_ltv     = dff["ltv_24m_inr"].sum()

    c1.metric("Total Active Outlets", f"{len(dff)}", f"+{int(len(dff)*0.08)} QoQ")
    c2.metric("Total Portfolio ARR", f"₹{total_arr/1e7:.2f} Cr", f"₹{total_arr/1e5:.0f}L")
    c3.metric("🔴 Critical Risk", f"{critical_n}", f"{critical_n/len(dff)*100:.1f}% of portfolio")
    c4.metric("⚠️ Revenue at Risk", f"₹{rev_at_risk/1e5:.1f}L", delta_color="inverse", delta=f"{rev_at_risk/total_arr*100:.1f}% of ARR")
    c5.metric("Total 24M LTV Pool", f"₹{total_ltv/1e7:.1f} Cr", "Predicted")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        # Health Distribution Donut
        health_counts = dff["health_tier"].value_counts().reset_index()
        health_counts.columns = ["tier","count"]
        fig_donut = go.Figure(go.Pie(
            labels=health_counts["tier"],
            values=health_counts["count"],
            hole=0.62,
            marker_colors=[COLOR_RISK.get(t, "#666") for t in health_counts["tier"]],
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Outlets: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(text=f"<b>{len(dff)}</b><br><span style='font-size:11px'>Outlets</span>",
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=22, color="white"))
        style_fig(fig_donut, "Portfolio Health Distribution", height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        # City-wise churn risk concentration
        city_risk = dff.groupby("city").agg(
            total=("outlet_id","count"),
            critical=("health_tier", lambda x: (x=="🔴 Critical").sum()),
            at_risk=("health_tier", lambda x: (x=="🟡 At Risk").sum()),
        ).reset_index()
        city_risk["risk_pct"] = (city_risk["critical"] + city_risk["at_risk"]) / city_risk["total"] * 100

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(name="🟢 Healthy",
            x=city_risk["city"],
            y=city_risk["total"] - city_risk["critical"] - city_risk["at_risk"],
            marker_color="#22C55E", opacity=0.85))
        fig_city.add_trace(go.Bar(name="🟡 At Risk",
            x=city_risk["city"], y=city_risk["at_risk"],
            marker_color="#F59E0B", opacity=0.85))
        fig_city.add_trace(go.Bar(name="🔴 Critical",
            x=city_risk["city"], y=city_risk["critical"],
            marker_color="#EF4444", opacity=0.85))
        fig_city.update_layout(barmode="stack", showlegend=True,
            legend=dict(orientation="h", y=-0.2, font=dict(size=11)))
        style_fig(fig_city, "Outlet Health by City", height=320)
        st.plotly_chart(fig_city, use_container_width=True)

    # Row 2
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        # Outlet type distribution
        type_rev = dff.groupby("outlet_type")["annual_revenue_inr"].sum().reset_index().sort_values("annual_revenue_inr", ascending=True)
        fig_type = px.bar(type_rev, x="annual_revenue_inr", y="outlet_type", orientation="h",
                          color="annual_revenue_inr", color_continuous_scale=["#1E3A5F","#1E6FD9","#00C2CB"],
                          labels={"annual_revenue_inr":"Annual Revenue (₹)","outlet_type":""})
        fig_type.update_coloraxes(showscale=False)
        fig_type.update_traces(hovertemplate="<b>%{y}</b><br>ARR: ₹%{x:,.0f}<extra></extra>")
        style_fig(fig_type, "ARR by Outlet Type", height=300)
        st.plotly_chart(fig_type, use_container_width=True)

    with col_b:
        # Churn risk distribution histogram
        fig_hist = px.histogram(dff, x="churn_risk_model", nbins=25,
                                 color_discrete_sequence=["#1E6FD9"],
                                 labels={"churn_risk_model":"Churn Risk Score (0–100)"})
        fig_hist.add_vline(x=70, line_dash="dash", line_color="#EF4444",
                           annotation_text="Critical (70)", annotation_position="top right",
                           annotation_font_color="#EF4444")
        fig_hist.add_vline(x=40, line_dash="dash", line_color="#F59E0B",
                           annotation_text="At Risk (40)", annotation_position="top left",
                           annotation_font_color="#F59E0B")
        style_fig(fig_hist, "Churn Risk Score Distribution", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_c:
        # Intervention split
        iv_counts = dff["recommended_intervention"].value_counts().reset_index()
        iv_counts.columns = ["intervention","count"]
        fig_iv = px.pie(iv_counts, names="intervention", values="count",
                        color="intervention",
                        color_discrete_map=COLOR_IV,
                        hole=0.45)
        fig_iv.update_traces(textinfo="percent+label", textfont_size=10)
        style_fig(fig_iv, "Recommended Interventions", height=300)
        st.plotly_chart(fig_iv, use_container_width=True)

    # Key Insights
    st.markdown("<div class='section-header'>🔑 Key Insights</div>", unsafe_allow_html=True)
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown(f"""<div class='insight-box'>
        <b>Revenue Concentration Risk</b><br>
        {critical_n} critical outlets represent ₹{rev_at_risk/1e5:.1f}L ARR ({rev_at_risk/total_arr*100:.1f}% of portfolio).
        Immediate retention action could save up to ₹{rev_at_risk*0.6/1e5:.1f}L annually.
        </div>""", unsafe_allow_html=True)
    with ins2:
        top_city = city_risk.sort_values("risk_pct", ascending=False).iloc[0]
        st.markdown(f"""<div class='insight-box'>
        <b>Highest Risk City</b><br>
        {top_city['city']} has the highest at-risk concentration at {top_city['risk_pct']:.1f}% of its outlets.
        Prioritise sales coverage in this market immediately.
        </div>""", unsafe_allow_html=True)
    with ins3:
        avg_ltv_h = dff[dff["health_tier"]=="🟢 Healthy"]["ltv_24m_inr"].mean()
        avg_ltv_c = dff[dff["health_tier"]=="🔴 Critical"]["ltv_24m_inr"].mean()
        st.markdown(f"""<div class='insight-box'>
        <b>LTV Gap Alert</b><br>
        Healthy outlets have avg 24M LTV of ₹{avg_ltv_h/1e5:.1f}L vs ₹{avg_ltv_c/1e5:.1f}L for critical accounts.
        Retaining one critical account = {avg_ltv_h/avg_ltv_c:.1f}× value uplift.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Risk Analysis":

    st.markdown("## ⚠️ Churn Risk Analysis")
    st.markdown("<div style='color:#94A3B8; font-size:0.88rem; margin-bottom:1.5rem;'>AI-predicted churn probability per outlet · Random Forest Classifier · AUC 0.962</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        # SCATTER: Churn Risk vs LTV — The Money Plot
        fig_scatter = px.scatter(
            dff, x="churn_risk_model", y="ltv_24m_inr",
            size="avg_monthly_cubes", color="health_tier",
            color_discrete_map=COLOR_RISK,
            hover_name="outlet_name",
            hover_data={"city": True, "outlet_type": True,
                        "churn_risk_model": ":.1f",
                        "ltv_24m_inr": ":,.0f",
                        "avg_monthly_cubes": ":,",
                        "health_tier": False},
            labels={"churn_risk_model": "Churn Risk Score →",
                    "ltv_24m_inr": "24-Month LTV (₹) →"},
            size_max=28,
        )
        # Quadrant lines
        fig_scatter.add_vline(x=60, line_dash="dot", line_color="#475569")
        fig_scatter.add_hline(y=dff["ltv_24m_inr"].median(), line_dash="dot", line_color="#475569")
        # Quadrant annotations
        fig_scatter.add_annotation(x=80, y=dff["ltv_24m_inr"].quantile(0.85), text="🚨 URGENT<br>High Risk + High Value",
                                    font=dict(size=10, color="#EF4444"), showarrow=False,
                                    bgcolor="#1A0A0A", bordercolor="#EF4444", borderwidth=1)
        fig_scatter.add_annotation(x=20, y=dff["ltv_24m_inr"].quantile(0.85), text="🌟 PROTECT<br>Safe + High Value",
                                    font=dict(size=10, color="#22C55E"), showarrow=False,
                                    bgcolor="#0A1A0A", bordercolor="#22C55E", borderwidth=1)
        style_fig(fig_scatter, "Churn Risk vs LTV Matrix — The Strategic Quadrant View", height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Feature Importance
        fi_top = fi_churn.head(10).copy()
        fi_top["feature_clean"] = fi_top["feature"].str.replace("_"," ").str.title()
        fig_fi = px.bar(fi_top.sort_values("importance"),
                        x="importance", y="feature_clean",
                        orientation="h",
                        color="importance",
                        color_continuous_scale=["#1E3A5F","#1E6FD9","#00C2CB"],
                        labels={"importance":"Importance Score","feature_clean":""})
        fig_fi.update_coloraxes(showscale=False)
        style_fig(fig_fi, "Top Churn Drivers (Feature Importance)", height=420)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Order Trend Analysis
    col_a, col_b = st.columns(2)

    with col_a:
        # Order trend by health tier — box plot
        order_cols = ["order_m1","order_m2","order_m3","order_m4","order_m5","order_m6"]
        months = ["M1","M2","M3","M4","M5","M6"]

        fig_trend = go.Figure()
        for tier, color in [("🔴 Critical","#EF4444"),("🟡 At Risk","#F59E0B"),("🟢 Healthy","#22C55E")]:
            subset = dff[dff["health_tier"]==tier]
            avg_per_month = [subset[c].mean() for c in order_cols]
            fig_trend.add_trace(go.Scatter(
                x=months, y=avg_per_month, name=tier,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                hovertemplate=f"<b>{tier}</b><br>Month: %{{x}}<br>Avg Cubes: %{{y:,.0f}}<extra></extra>"
            ))
        style_fig(fig_trend, "Average Order Volume Trend by Health Tier (6 Months)", height=320)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        # Payment delay vs churn risk
        fig_pay = px.scatter(dff, x="payment_delay_days", y="churn_risk_model",
                             color="health_tier", color_discrete_map=COLOR_RISK,
                             hover_name="outlet_name",
                             labels={"payment_delay_days":"Payment Delay (Days)","churn_risk_model":"Churn Risk Score"},
                             opacity=0.75)
        style_fig(fig_pay, "Payment Delay vs Churn Risk Score", height=320)
        st.plotly_chart(fig_pay, use_container_width=True)

    # At-risk table
    st.markdown("<div class='section-header'>🚨 High-Risk Outlet Watchlist (Score ≥ 60)</div>", unsafe_allow_html=True)
    watchlist = dff[dff["churn_risk_model"] >= 60].sort_values("churn_risk_model", ascending=False)[[
        "outlet_name","city","outlet_type","churn_risk_model","ltv_24m_inr",
        "days_since_last_order","complaints_last3m","payment_delay_days",
        "recommended_intervention","health_tier"
    ]].head(25)
    watchlist.columns = ["Outlet","City","Type","Risk Score","LTV (₹)","Days Inactive",
                          "Complaints","Delay (Days)","Intervention","Status"]
    watchlist["Risk Score"] = watchlist["Risk Score"].apply(lambda x: f"{x:.1f}")
    watchlist["LTV (₹)"] = watchlist["LTV (₹)"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(watchlist, use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LTV & REVENUE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 LTV & Revenue Intel":

    st.markdown("## 💰 LTV & Revenue Intelligence")
    st.markdown("<div style='color:#94A3B8; font-size:0.88rem; margin-bottom:1.5rem;'>24-month customer lifetime value predictions · Anchored to ₹6/cube @ 46% gross margin</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total 24M LTV Pool", f"₹{dff['ltv_24m_inr'].sum()/1e7:.2f} Cr")
    c2.metric("Avg LTV per Outlet", f"₹{dff['ltv_24m_inr'].mean()/1e5:.1f}L")
    c3.metric("Top 20% LTV Share", f"{dff.nlargest(int(len(dff)*0.2),'ltv_24m_inr')['ltv_24m_inr'].sum()/dff['ltv_24m_inr'].sum()*100:.1f}%")
    c4.metric("At-Risk LTV Exposure", f"₹{dff[dff['health_tier']!='🟢 Healthy']['ltv_24m_inr'].sum()/1e7:.2f} Cr")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # LTV by Outlet Type — violin / box
        fig_box = px.box(dff, x="outlet_type", y="ltv_24m_inr",
                         color="outlet_type",
                         color_discrete_sequence=["#1E6FD9","#00C2CB","#22C55E","#F59E0B","#EF4444","#A78BFA","#EC4899"],
                         labels={"ltv_24m_inr":"24M LTV (₹)","outlet_type":""},
                         points="outliers")
        fig_box.update_layout(showlegend=False)
        style_fig(fig_box, "LTV Distribution by Outlet Type", height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        # LTV by City — bar with error
        city_ltv = dff.groupby("city")["ltv_24m_inr"].agg(["mean","std","sum"]).reset_index()
        city_ltv.columns = ["city","mean_ltv","std_ltv","total_ltv"]
        city_ltv = city_ltv.sort_values("mean_ltv", ascending=False)
        fig_cltv = go.Figure()
        fig_cltv.add_trace(go.Bar(
            x=city_ltv["city"], y=city_ltv["mean_ltv"],
            error_y=dict(type="data", array=city_ltv["std_ltv"]/2, visible=True, color="#475569"),
            marker_color=["#1E6FD9","#00C2CB","#22C55E","#F59E0B","#EF4444","#A78BFA","#EC4899"][:len(city_ltv)],
            hovertemplate="<b>%{x}</b><br>Avg LTV: ₹%{y:,.0f}<extra></extra>"
        ))
        style_fig(fig_cltv, "Average 24M LTV by City", height=360)
        st.plotly_chart(fig_cltv, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # LTV vs Monthly Revenue scatter
        fig_lmr = px.scatter(dff, x="monthly_revenue_inr", y="ltv_24m_inr",
                              color="health_tier", color_discrete_map=COLOR_RISK,
                              hover_name="outlet_name",
                              trendline="ols",
                              labels={"monthly_revenue_inr":"Monthly Revenue (₹)","ltv_24m_inr":"24M LTV (₹)"})
        style_fig(fig_lmr, "Monthly Revenue vs 24M LTV (with Trend)", height=320)
        st.plotly_chart(fig_lmr, use_container_width=True)

    with col_b:
        # LTV Feature Importance
        fi_ltv_top = fi_ltv.head(10).copy()
        fi_ltv_top["feature_clean"] = fi_ltv_top["feature"].str.replace("_"," ").str.title()
        fig_lfi = px.bar(fi_ltv_top.sort_values("importance"),
                         x="importance", y="feature_clean", orientation="h",
                         color="importance",
                         color_continuous_scale=["#1E3A5F","#00C2CB","#22C55E"],
                         labels={"importance":"Importance","feature_clean":""})
        fig_lfi.update_coloraxes(showscale=False)
        style_fig(fig_lfi, "Top LTV Predictors (Feature Importance)", height=320)
        st.plotly_chart(fig_lfi, use_container_width=True)

    # Top 20 Highest LTV Accounts
    st.markdown("<div class='section-header'>🏅 Top 20 Highest LTV Accounts</div>", unsafe_allow_html=True)
    top20 = dff.nlargest(20, "ltv_24m_inr")[[
        "outlet_name","city","outlet_type","ltv_24m_inr","monthly_revenue_inr",
        "churn_risk_model","health_tier","account_manager"
    ]].copy()
    top20["ltv_24m_inr"] = top20["ltv_24m_inr"].apply(lambda x: f"₹{x:,.0f}")
    top20["monthly_revenue_inr"] = top20["monthly_revenue_inr"].apply(lambda x: f"₹{x:,.0f}")
    top20["churn_risk_model"] = top20["churn_risk_model"].apply(lambda x: f"{x:.1f}")
    top20.columns = ["Outlet","City","Type","24M LTV","Monthly Rev","Risk Score","Status","Account Manager"]
    st.dataframe(top20, use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERVENTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Intervention Engine":

    st.markdown("## 🎯 AI Retention Intervention Engine")
    st.markdown("<div style='color:#94A3B8; font-size:0.88rem; margin-bottom:1.5rem;'>Optimal intervention strategy per at-risk outlet · ROI-optimised recommendations</div>", unsafe_allow_html=True)

    # Only show at-risk and critical
    at_risk_df = dff[dff["churn_risk_model"] >= 35].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outlets Needing Intervention", len(at_risk_df))
    c2.metric("Total Intervention Budget", f"₹{at_risk_df['intervention_cost_inr'].sum()/1e5:.1f}L")
    c3.metric("Expected Revenue Saved", f"₹{at_risk_df['expected_revenue_saved_inr'].sum()/1e5:.1f}L")
    c4.metric("Avg Retention ROI", f"{at_risk_df[at_risk_df['retention_roi']>0]['retention_roi'].mean():.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Intervention distribution
        iv_dist = at_risk_df["recommended_intervention"].value_counts().reset_index()
        iv_dist.columns = ["intervention","count"]
        fig_ivd = px.bar(iv_dist.sort_values("count"),
                         x="count", y="intervention", orientation="h",
                         color="intervention", color_discrete_map=COLOR_IV,
                         labels={"count":"# Outlets","intervention":""},
                         text="count")
        fig_ivd.update_traces(textposition="outside", textfont_color="white")
        fig_ivd.update_layout(showlegend=False)
        style_fig(fig_ivd, "Intervention Type Distribution", height=320)
        st.plotly_chart(fig_ivd, use_container_width=True)

    with col2:
        # ROI by intervention type
        roi_by_iv = at_risk_df[at_risk_df["retention_roi"] > 0].groupby("recommended_intervention").agg(
            avg_roi=("retention_roi","mean"),
            total_saved=("expected_revenue_saved_inr","sum"),
            total_cost=("intervention_cost_inr","sum")
        ).reset_index().sort_values("avg_roi", ascending=False)

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            x=roi_by_iv["recommended_intervention"],
            y=roi_by_iv["avg_roi"],
            marker_color=[COLOR_IV.get(i,"#666") for i in roi_by_iv["recommended_intervention"]],
            text=[f"{v:.0f}%" for v in roi_by_iv["avg_roi"]],
            textposition="outside", textfont_color="white",
            hovertemplate="<b>%{x}</b><br>Avg ROI: %{y:.1f}%<extra></extra>"
        ))
        style_fig(fig_roi, "Average Retention ROI by Intervention Type", height=320)
        st.plotly_chart(fig_roi, use_container_width=True)

    # Revenue Waterfall
    col_a, col_b = st.columns(2)

    with col_a:
        total_portfolio_rev = dff["annual_revenue_inr"].sum()
        rev_at_risk_val     = at_risk_df["annual_revenue_inr"].sum()
        rev_saveable        = at_risk_df["expected_revenue_saved_inr"].sum()
        rev_lost_anyway     = rev_at_risk_val - rev_saveable

        fig_wf = go.Figure(go.Waterfall(
            name="Revenue", orientation="v",
            x=["Total ARR", "At-Risk ARR", "Intervention<br>Saves", "Net Retained<br>ARR"],
            measure=["absolute","relative","relative","total"],
            y=[total_portfolio_rev, -rev_at_risk_val, rev_saveable, 0],
            text=[f"₹{total_portfolio_rev/1e7:.1f}Cr",
                  f"-₹{rev_at_risk_val/1e5:.0f}L",
                  f"+₹{rev_saveable/1e5:.0f}L",
                  f"₹{(total_portfolio_rev-rev_at_risk_val+rev_saveable)/1e7:.1f}Cr"],
            textposition="outside",
            connector=dict(line=dict(color="#1E3A5F", width=1)),
            increasing=dict(marker=dict(color="#22C55E")),
            decreasing=dict(marker=dict(color="#EF4444")),
            totals=dict(marker=dict(color="#1E6FD9")),
        ))
        style_fig(fig_wf, "Revenue Waterfall: Risk vs Retention Impact", height=320)
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_b:
        # Budget vs savings bubble
        fig_bub = px.scatter(
            roi_by_iv,
            x="total_cost", y="total_saved",
            size="avg_roi", color="recommended_intervention",
            color_discrete_map=COLOR_IV,
            hover_name="recommended_intervention",
            text="recommended_intervention",
            labels={"total_cost":"Total Intervention Cost (₹)","total_saved":"Total Revenue Saved (₹)"},
            size_max=50
        )
        fig_bub.update_traces(textposition="top center", textfont_size=9)
        style_fig(fig_bub, "Intervention Cost vs Revenue Saved (Bubble = ROI)", height=320)
        st.plotly_chart(fig_bub, use_container_width=True)

    # Full intervention table
    st.markdown("<div class='section-header'>📋 Complete Intervention Recommendation Table</div>", unsafe_allow_html=True)

    show_df = at_risk_df.sort_values("retention_priority_index", ascending=False)[[
        "outlet_name","city","outlet_type","churn_risk_model","ltv_24m_inr",
        "recommended_intervention","intervention_cost_inr",
        "expected_revenue_saved_inr","retention_roi","health_tier"
    ]].head(50).copy()

    show_df.columns = ["Outlet","City","Type","Risk Score","LTV (₹)",
                        "Recommended Intervention","Cost (₹)","Revenue Saved (₹)","ROI (%)","Status"]
    show_df["Risk Score"]       = show_df["Risk Score"].apply(lambda x: f"{x:.1f}")
    show_df["LTV (₹)"]          = show_df["LTV (₹)"].apply(lambda x: f"₹{x:,.0f}")
    show_df["Cost (₹)"]         = show_df["Cost (₹)"].apply(lambda x: f"₹{x:,}")
    show_df["Revenue Saved (₹)"] = show_df["Revenue Saved (₹)"].apply(lambda x: f"₹{x:,.0f}")
    show_df["ROI (%)"]          = show_df["ROI (%)"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(show_df, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RETENTION PRIORITY INDEX
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Retention Priority Index":

    st.markdown("## 🏆 Retention Priority Index")
    st.markdown("<div style='color:#94A3B8; font-size:0.88rem; margin-bottom:1.5rem;'>Composite score = Churn Risk × LTV × Margin (46%) × Intervention Efficiency · Higher = Act Now</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        # Leaderboard bar chart
        top_rpi = dff.nlargest(20, "retention_priority_index")[
            ["outlet_name","city","retention_priority_index","churn_risk_model","ltv_24m_inr","health_tier"]
        ].copy()
        fig_rpi = px.bar(
            top_rpi.sort_values("retention_priority_index"),
            x="retention_priority_index", y="outlet_name",
            orientation="h",
            color="health_tier", color_discrete_map=COLOR_RISK,
            hover_data={"city":True,"churn_risk_model":":.1f","ltv_24m_inr":":,.0f"},
            labels={"retention_priority_index":"Retention Priority Index","outlet_name":""}
        )
        style_fig(fig_rpi, "Top 20 Outlets by Retention Priority Index", height=520)
        st.plotly_chart(fig_rpi, use_container_width=True)

    with col2:
        # Tier breakdown
        tier_counts = dff["priority_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier","count"]
        TIER_COLORS = {"Tier 1 – Urgent":"#EF4444","Tier 2 – Engage":"#F59E0B","Tier 3 – Monitor":"#22C55E"}
        fig_tier = go.Figure(go.Pie(
            labels=tier_counts["tier"], values=tier_counts["count"],
            hole=0.55,
            marker_colors=[TIER_COLORS.get(t,"#666") for t in tier_counts["tier"]],
            textinfo="label+percent+value",
            textfont=dict(size=11)
        ))
        style_fig(fig_tier, "Priority Tier Distribution", height=300)
        st.plotly_chart(fig_tier, use_container_width=True)

        # Account Manager workload
        am_load = dff[dff["priority_tier"]=="Tier 1 – Urgent"].groupby("account_manager").size().reset_index()
        am_load.columns = ["Account Manager","Urgent Accounts"]
        am_load = am_load.sort_values("Urgent Accounts", ascending=False)
        fig_am = px.bar(am_load, x="Account Manager", y="Urgent Accounts",
                        color="Urgent Accounts",
                        color_continuous_scale=["#F59E0B","#EF4444"],
                        text="Urgent Accounts")
        fig_am.update_traces(textposition="outside", textfont_color="white")
        fig_am.update_coloraxes(showscale=False)
        style_fig(fig_am, "Urgent Accounts per Account Manager", height=280)
        st.plotly_chart(fig_am, use_container_width=True)

    # RPI vs Churn vs LTV 3D-like bubble
    fig_3 = px.scatter(
        dff.nlargest(80, "retention_priority_index"),
        x="churn_risk_model", y="ltv_24m_inr",
        size="retention_priority_index",
        color="priority_tier",
        color_discrete_map=TIER_COLORS,
        hover_name="outlet_name",
        hover_data={"retention_priority_index":":.2f","city":True},
        labels={"churn_risk_model":"Churn Risk Score","ltv_24m_inr":"24M LTV (₹)"},
        size_max=35
    )
    style_fig(fig_3, "Retention Priority Index — Top 80 Accounts (Size = Priority Score)", height=380)
    st.plotly_chart(fig_3, use_container_width=True)

    # Full RPI table
    st.markdown("<div class='section-header'>📊 Full Retention Priority Leaderboard</div>", unsafe_allow_html=True)
    rpi_table = dff.sort_values("retention_priority_index", ascending=False)[[
        "outlet_name","city","outlet_type","retention_priority_index",
        "churn_risk_model","ltv_24m_inr","priority_tier",
        "recommended_intervention","account_manager"
    ]].head(50).copy()
    rpi_table.columns = ["Outlet","City","Type","Priority Index","Risk Score",
                          "LTV (₹)","Priority Tier","Intervention","Account Manager"]
    rpi_table["Priority Index"] = rpi_table["Priority Index"].apply(lambda x: f"{x:.2f}")
    rpi_table["Risk Score"]     = rpi_table["Risk Score"].apply(lambda x: f"{x:.1f}")
    rpi_table["LTV (₹)"]        = rpi_table["LTV (₹)"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(rpi_table, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT NEW LEAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict New Lead":

    st.markdown("## 🔮 Predict Churn Risk & LTV for a New Outlet")
    st.markdown("<div style='color:#94A3B8; font-size:0.88rem; margin-bottom:1.5rem;'>Enter outlet details to get AI-powered churn risk score, LTV estimate & intervention recommendation</div>", unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            outlet_name_inp = st.text_input("Outlet Name", "New Premium Café Delhi")
            city_inp        = st.selectbox("City", sorted(df["city"].unique()))
            outlet_type_inp = st.selectbox("Outlet Type", sorted(df["outlet_type"].unique()))
            cuisine_inp     = st.selectbox("Cuisine Type", sorted(df["cuisine_type"].unique()))
        with c2:
            seating_inp     = st.slider("Seating Capacity", 10, 300, 80)
            rating_inp      = st.slider("Google Rating", 1.0, 5.0, 4.2, step=0.1)
            instagram_inp   = st.number_input("Instagram Followers", 100, 100000, 5000, step=500)
            months_inp      = st.slider("Months Since Onboarding", 1, 36, 6)
        with c3:
            flavours_inp    = st.slider("Flavours Ordered", 1, 8, 3)
            complaints_inp  = st.slider("Complaints (Last 3M)", 0, 10, 1)
            delay_inp       = st.slider("Payment Delay (Days)", 0, 60, 5)
            am_inp          = st.selectbox("Account Manager Assigned?", ["Yes","No"])
            outreach_inp    = st.slider("Outreach Response Rate (0=None, 3=All)", 0, 3, 2)

        col_m, _ = st.columns([2,1])
        with col_m:
            st.markdown("**Monthly Order Volumes (Cubes)**")
            mc1,mc2,mc3 = st.columns(3)
            m1_inp = mc1.number_input("Month 1", 1000, 25000, 10000, step=500)
            m2_inp = mc1.number_input("Month 2", 1000, 25000, 9800, step=500)
            m3_inp = mc2.number_input("Month 3", 1000, 25000, 9500, step=500)
            m4_inp = mc2.number_input("Month 4", 1000, 25000, 9000, step=500)
            m5_inp = mc3.number_input("Month 5", 1000, 25000, 8200, step=500)
            m6_inp = mc3.number_input("Month 6", 1000, 25000, 7500, step=500)

        submitted = st.form_submit_button("🔮 Run AI Prediction", use_container_width=True)

    if submitted:
        with open("label_encoders.pkl","rb") as f:
            le = pickle.load(f)

        avg_vol   = int(np.mean([m1_inp,m2_inp,m3_inp,m4_inp,m5_inp,m6_inp]))
        trend_pct = round((m6_inp - m1_inp) / (m1_inp+1) * 100, 2)
        slope     = (m6_inp - m1_inp) / 5
        vol_arr   = [m1_inp,m2_inp,m3_inp,m4_inp,m5_inp,m6_inp]
        volatility= float(np.std(vol_arr))
        recent_vs = m6_inp / (avg_vol + 1)

        try: city_enc = le["city"].transform([city_inp])[0]
        except: city_enc = 0
        try: type_enc = le["type"].transform([outlet_type_inp])[0]
        except: type_enc = 0
        try: cuisine_enc = le["cuisine"].transform([cuisine_inp])[0]
        except: cuisine_enc = 0

        X_new = pd.DataFrame([[
            seating_inp, rating_inp, instagram_inp, months_inp,
            flavours_inp, complaints_inp, delay_inp,
            1 if am_inp=="Yes" else 0, outreach_inp,
            avg_vol, trend_pct, 15,
            slope, volatility, recent_vs,
            city_enc, type_enc, cuisine_enc
        ]], columns=features)

        churn_prob_new = churn_model.predict_proba(X_new)[0][1]
        churn_score_new = round(churn_prob_new * 100, 1)
        ltv_pred_new = int(ltv_model.predict(X_new)[0])
        monthly_rev_new = avg_vol * 6
        monthly_gp_new  = avg_vol * 2.76

        # Health tier
        if churn_score_new >= 70:   tier_new = "🔴 Critical"
        elif churn_score_new >= 40: tier_new = "🟡 At Risk"
        else:                       tier_new = "🟢 Healthy"

        # Intervention
        if churn_score_new < 35:       iv_new = "No Intervention Needed"
        elif delay_inp > 25:           iv_new = "Extended Credit Terms"
        elif complaints_inp > 4:       iv_new = "Dedicated Account Manager"
        elif ltv_pred_new > 500000:    iv_new = "Dedicated Account Manager"
        elif outreach_inp == 0:        iv_new = "Premium Flavour Trial"
        elif churn_score_new > 70:     iv_new = "5% Discount Offer"
        else:                          iv_new = "Premium Flavour Trial"

        iv_cost = {"No Intervention Needed":0,"5% Discount Offer":int(monthly_rev_new*0.05),
                   "Premium Flavour Trial":500,"Extended Credit Terms":int(monthly_rev_new*0.02),
                   "Dedicated Account Manager":8000}
        iv_eff  = {"No Intervention Needed":0,"5% Discount Offer":0.55,"Premium Flavour Trial":0.65,
                   "Extended Credit Terms":0.70,"Dedicated Account Manager":0.85}
        cost_new = iv_cost.get(iv_new, 0)
        saved_new = int(ltv_pred_new * (churn_prob_new) * iv_eff.get(iv_new, 0))
        roi_new   = round((saved_new - cost_new) / (cost_new+1) * 100, 1) if cost_new > 0 else 0
        rpi_new   = round(churn_prob_new * (ltv_pred_new/df["ltv_24m_inr"].max()) * 0.46 * iv_eff.get(iv_new,0) * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### Results for **{outlet_name_inp}**")
        r1,r2,r3,r4,r5 = st.columns(5)
        r1.metric("Churn Risk Score", f"{churn_score_new}", tier_new)
        r2.metric("Predicted 24M LTV", f"₹{ltv_pred_new:,.0f}")
        r3.metric("Monthly Revenue", f"₹{monthly_rev_new:,.0f}")
        r4.metric("Retention Priority", f"{rpi_new:.2f}")
        r5.metric("Recommended Action", iv_new)

        # Gauge chart
        col_g, col_info = st.columns([1, 1.5])
        with col_g:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_score_new,
                title={"text": "Churn Risk Score", "font": {"size": 16, "color":"#E8F0FE"}},
                delta={"reference": 50, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#22C55E"}},
                gauge={
                    "axis": {"range": [0,100], "tickcolor":"#94A3B8"},
                    "bar": {"color": "#EF4444" if churn_score_new>=70 else "#F59E0B" if churn_score_new>=40 else "#22C55E"},
                    "steps": [
                        {"range":[0,40],  "color":"#0D2A18"},
                        {"range":[40,70], "color":"#2A1F0D"},
                        {"range":[70,100],"color":"#2A0D0D"},
                    ],
                    "threshold": {"line":{"color":"white","width":2},"thickness":0.8,"value":churn_score_new}
                }
            ))
            fig_gauge.update_layout(**PLOTLY_THEME, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            st.markdown(f"""
            <div style='background:#162032; border:1px solid #1E3A5F; border-radius:12px; padding:1.4rem; margin-top:0.5rem;'>
            <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; color:#94A3B8; margin-bottom:0.8rem;'>AI RECOMMENDATION SUMMARY</div>

            <div style='margin-bottom:0.7rem;'>
            <span style='color:#94A3B8; font-size:0.82rem;'>Health Status</span><br>
            <span style='font-size:1.1rem; font-weight:600;'>{tier_new}</span>
            </div>

            <div style='margin-bottom:0.7rem;'>
            <span style='color:#94A3B8; font-size:0.82rem;'>Intervention Strategy</span><br>
            <span style='font-size:1.05rem; font-weight:600; color:#4A9FFF;'>{iv_new}</span>
            </div>

            <div style='display:flex; gap:1.5rem; margin-top:0.8rem; flex-wrap:wrap;'>
            <div><span style='color:#94A3B8; font-size:0.78rem;'>Intervention Cost</span><br><b>₹{cost_new:,}</b></div>
            <div><span style='color:#94A3B8; font-size:0.78rem;'>Expected Revenue Saved</span><br><b style='color:#22C55E;'>₹{saved_new:,.0f}</b></div>
            <div><span style='color:#94A3B8; font-size:0.78rem;'>Retention ROI</span><br><b style='color:#4A9FFF;'>{roi_new:.1f}%</b></div>
            </div>

            <div style='margin-top:1rem; padding:0.7rem; background:#0D1B2A; border-radius:8px; font-size:0.83rem; color:#CBD5E1;'>
            <b>Order Trend:</b> {trend_pct:+.1f}% over 6 months
            {"— ⚠️ Declining trend detected. Early intervention critical." if trend_pct < -10 else "— ✅ Stable order pattern." if abs(trend_pct) < 10 else "— 📈 Growing account."}
            </div>
            </div>
            """, unsafe_allow_html=True)

        # Order trend mini chart
        fig_ot = go.Figure(go.Scatter(
            x=["M1","M2","M3","M4","M5","M6"],
            y=[m1_inp,m2_inp,m3_inp,m4_inp,m5_inp,m6_inp],
            mode="lines+markers+text",
            text=[f"{v:,}" for v in [m1_inp,m2_inp,m3_inp,m4_inp,m5_inp,m6_inp]],
            textposition="top center", textfont=dict(size=10),
            line=dict(color="#4A9FFF" if trend_pct >= 0 else "#EF4444", width=2.5),
            marker=dict(size=8, color="#4A9FFF" if trend_pct >= 0 else "#EF4444"),
            fill="tozeroy",
            fillcolor="rgba(30,111,217,0.1)" if trend_pct >= 0 else "rgba(239,68,68,0.1)"
        ))
        style_fig(fig_ot, f"Order Volume Trend — {outlet_name_inp}", height=250)
        st.plotly_chart(fig_ot, use_container_width=True)
