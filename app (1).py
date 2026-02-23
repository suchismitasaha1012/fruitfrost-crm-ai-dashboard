"""
FruitFrost CRM — AI-Driven Retention & Revenue Optimisation Dashboard
Run: streamlit run app.py
Install: pip install streamlit pandas numpy scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FruitFrost CRM Intelligence",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0D1B2A !important;
    color: #E8F0FE !important;
}
.main { background-color: #0D1B2A !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #162032 100%) !important;
    border-right: 1px solid #1E3A5F !important;
}
[data-testid="stSidebar"] * { color: #E8F0FE !important; }

[data-testid="stMetric"] {
    background: #162032 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] { color: #E8F0FE !important; font-size: 1.7rem !important; font-weight: 700 !important; }

[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1E6FD9, #00C2CB) !important;
    color: white !important;
}

h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #E8F0FE !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #00C2CB !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #E8F0FE !important; }

.insight-box {
    background: linear-gradient(135deg, #162032, #1A2D42);
    border-left: 3px solid #00C2CB;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.87rem;
    color: #CBD5E1;
    line-height: 1.6;
}
.section-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #4A9FFF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 0.5rem;
    margin: 1.2rem 0 0.8rem 0;
}
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load Data & Models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("fruitfrost_outlets_scored.csv")

@st.cache_resource
def load_models():
    with open("churn_model.pkl", "rb") as f:   cm = pickle.load(f)
    with open("ltv_model.pkl", "rb") as f:     lm = pickle.load(f)
    with open("feature_list.pkl", "rb") as f:  fl = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f: le = pickle.load(f)
    fi_c = pd.read_csv("feature_importance_churn.csv")
    fi_l = pd.read_csv("feature_importance_ltv.csv")
    return cm, lm, fl, le, fi_c, fi_l

df = load_data()
churn_model, ltv_model, features, label_encoders, fi_churn, fi_ltv = load_models()

# ── Color Maps ────────────────────────────────────────────────────────────────
C_RISK = {"🔴 Critical": "#EF4444", "🟡 At Risk": "#F59E0B", "🟢 Healthy": "#22C55E"}
C_IV   = {
    "Dedicated Account Manager": "#1E6FD9",
    "5% Discount Offer":          "#F59E0B",
    "Extended Credit Terms":      "#A78BFA",
    "Premium Flavour Trial":      "#00C2CB",
    "No Intervention Needed":     "#22C55E",
}
C_TIER = {"Tier 1 – Urgent": "#EF4444", "Tier 2 – Engage": "#F59E0B", "Tier 3 – Monitor": "#22C55E"}

# ── Plotly Layout Helper ──────────────────────────────────────────────────────
BASE = dict(
    paper_bgcolor="#0D1B2A", plot_bgcolor="#0D1B2A",
    font=dict(family="DM Sans", color="#E8F0FE", size=12),
    margin=dict(l=40, r=20, t=45, b=40),
    xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F"),
    yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F"),
    colorway=["#1E6FD9","#00C2CB","#22C55E","#F59E0B","#EF4444","#A78BFA","#EC4899"],
)

def sfig(fig, title="", h=380):
    fig.update_layout(**BASE, title=dict(text=title, font=dict(size=13, color="#E8F0FE")), height=h)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0;'>
      <div style='font-size:2.2rem;'>🧊</div>
      <div style='font-size:1.2rem;font-weight:700;color:#4A9FFF;letter-spacing:0.05em;'>FRUITFROST</div>
      <div style='font-size:0.68rem;color:#94A3B8;letter-spacing:0.12em;'>CRM INTELLIGENCE</div>
      <hr style='border-color:#1E3A5F;margin:0.7rem 0;'>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊 Executive Summary",
        "⚠️ Churn Risk Analysis",
        "💰 LTV & Revenue Intel",
        "🎯 Intervention Engine",
        "🏆 Retention Priority Index",
        "🔮 Predict New Lead",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E3A5F;'>", unsafe_allow_html=True)
    st.markdown("**🔍 Filters**")
    city_f = st.multiselect("City", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
    type_f = st.multiselect("Outlet Type", sorted(df["outlet_type"].unique()), default=sorted(df["outlet_type"].unique()))
    risk_f = st.multiselect("Health Tier",
        ["🔴 Critical","🟡 At Risk","🟢 Healthy"],
        default=["🔴 Critical","🟡 At Risk","🟢 Healthy"])

    st.markdown("<hr style='border-color:#1E3A5F;'>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:0.68rem;color:#475569;text-align:center;line-height:1.7;'>
    IIM Ranchi · Entrepreneurial Marketing<br>
    AI-Driven CRM · Group 2<br>
    <span style='color:#1E6FD9;'>Suchismita Saha · M105-23</span>
    </div>""", unsafe_allow_html=True)

# Apply filters
dff = df[
    df["city"].isin(city_f) &
    df["outlet_type"].isin(type_f) &
    df["health_tier"].isin(risk_f)
].copy()

if len(dff) == 0:
    st.warning("No data matches current filters. Please adjust filters.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("## 📊 Executive Summary")
    st.markdown(f"<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>Showing {len(dff)} of {len(df)} outlets · FruitFrost B2B Portfolio</div>", unsafe_allow_html=True)

    total_arr   = dff["annual_revenue_inr"].sum()
    critical_n  = (dff["health_tier"] == "🔴 Critical").sum()
    atrisk_n    = (dff["health_tier"] == "🟡 At Risk").sum()
    rev_at_risk = dff[dff["health_tier"] == "🔴 Critical"]["annual_revenue_inr"].sum()
    total_ltv   = dff["ltv_predicted_inr"].sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Outlets",        f"{len(dff)}")
    c2.metric("Portfolio ARR",        f"₹{total_arr/1e7:.2f} Cr")
    c3.metric("🔴 Critical Risk",     f"{critical_n}", f"{critical_n/max(len(dff),1)*100:.1f}% of portfolio")
    c4.metric("Revenue at Risk",      f"₹{rev_at_risk/1e5:.1f}L", delta_color="inverse",
              delta=f"{rev_at_risk/max(total_arr,1)*100:.1f}% of ARR")
    c5.metric("Predicted 24M LTV",    f"₹{total_ltv/1e7:.1f} Cr")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        hc = dff["health_tier"].value_counts().reset_index()
        hc.columns = ["tier","count"]
        fig = go.Figure(go.Pie(
            labels=hc["tier"], values=hc["count"], hole=0.60,
            marker_colors=[C_RISK.get(t,"#666") for t in hc["tier"]],
            textinfo="label+percent", textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig.add_annotation(text=f"<b>{len(dff)}</b><br>Outlets",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="white"))
        sfig(fig, "Portfolio Health Distribution", 320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cr = dff.groupby("city").agg(
            total=("outlet_id","count"),
            critical=("health_tier", lambda x: (x=="🔴 Critical").sum()),
            at_risk=("health_tier",  lambda x: (x=="🟡 At Risk").sum()),
            healthy=("health_tier",  lambda x: (x=="🟢 Healthy").sum()),
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="🟢 Healthy",   x=cr["city"], y=cr["healthy"],  marker_color="#22C55E"))
        fig2.add_trace(go.Bar(name="🟡 At Risk",   x=cr["city"], y=cr["at_risk"],  marker_color="#F59E0B"))
        fig2.add_trace(go.Bar(name="🔴 Critical",  x=cr["city"], y=cr["critical"], marker_color="#EF4444"))
        fig2.update_layout(barmode="stack", legend=dict(orientation="h",y=-0.2,font=dict(size=10)))
        sfig(fig2, "Outlet Health by City", 320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4, col5 = st.columns(3)

    with col3:
        tr = dff.groupby("outlet_type")["annual_revenue_inr"].sum().reset_index().sort_values("annual_revenue_inr")
        fig3 = px.bar(tr, x="annual_revenue_inr", y="outlet_type", orientation="h",
                      color="annual_revenue_inr",
                      color_continuous_scale=["#1E3A5F","#1E6FD9","#00C2CB"],
                      labels={"annual_revenue_inr":"Annual Revenue (₹)","outlet_type":""})
        fig3.update_coloraxes(showscale=False)
        sfig(fig3, "ARR by Outlet Type", 300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.histogram(dff, x="churn_risk_model", nbins=25,
                            color_discrete_sequence=["#1E6FD9"],
                            labels={"churn_risk_model":"Churn Risk Score (0–100)"})
        fig4.add_vline(x=70, line_dash="dash", line_color="#EF4444",
                       annotation_text="Critical",annotation_font_color="#EF4444",annotation_position="top right")
        fig4.add_vline(x=40, line_dash="dash", line_color="#F59E0B",
                       annotation_text="At Risk",annotation_font_color="#F59E0B",annotation_position="top left")
        sfig(fig4, "Churn Risk Score Distribution", 300)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        iv = dff["recommended_intervention"].value_counts().reset_index()
        iv.columns = ["intervention","count"]
        fig5 = px.pie(iv, names="intervention", values="count",
                      color="intervention", color_discrete_map=C_IV, hole=0.45)
        fig5.update_traces(textinfo="percent+label", textfont_size=9)
        sfig(fig5, "Recommended Interventions", 300)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div class='section-header'>🔑 Key Insights</div>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    avg_h = dff[dff["health_tier"]=="🟢 Healthy"]["ltv_predicted_inr"].mean() if (dff["health_tier"]=="🟢 Healthy").any() else 0
    avg_c = dff[dff["health_tier"]=="🔴 Critical"]["ltv_predicted_inr"].mean() if (dff["health_tier"]=="🔴 Critical").any() else 1
    top_risk_city = cr.assign(pct=(cr["critical"]+cr["at_risk"])/cr["total"]*100).sort_values("pct",ascending=False).iloc[0]
    with i1:
        st.markdown(f"""<div class='insight-box'><b>Revenue Concentration Risk</b><br>
        {critical_n} critical outlets = ₹{rev_at_risk/1e5:.1f}L ARR at risk ({rev_at_risk/max(total_arr,1)*100:.1f}% of portfolio).
        Timely retention could save ₹{rev_at_risk*0.6/1e5:.1f}L annually.</div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class='insight-box'><b>Highest Risk City: {top_risk_city['city']}</b><br>
        {top_risk_city['pct']:.1f}% of {top_risk_city['city']} outlets are at-risk or critical.
        Prioritise sales and CRM coverage here immediately.</div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class='insight-box'><b>LTV Gap Alert</b><br>
        Healthy outlets avg ₹{avg_h/1e5:.1f}L vs ₹{avg_c/1e5:.1f}L for critical accounts.
        Every retained critical account = {avg_h/max(avg_c,1):.1f}× LTV uplift.</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Risk Analysis":
    st.markdown("## ⚠️ Churn Risk Analysis")
    st.markdown("<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>Random Forest Classifier · ROC-AUC 0.962 · Predicts churn within 60 days</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        fig = px.scatter(
            dff, x="churn_risk_model", y="ltv_predicted_inr",
            size="avg_monthly_cubes", color="health_tier",
            color_discrete_map=C_RISK,
            hover_name="outlet_name",
            hover_data={"city":True,"outlet_type":True,
                        "churn_risk_model":":.1f","ltv_predicted_inr":":,.0f",
                        "avg_monthly_cubes":":,","health_tier":False},
            labels={"churn_risk_model":"Churn Risk Score →","ltv_predicted_inr":"Predicted 24M LTV (₹) →"},
            size_max=28,
        )
        median_ltv = dff["ltv_predicted_inr"].median()
        q85_ltv    = dff["ltv_predicted_inr"].quantile(0.85)
        fig.add_vline(x=60, line_dash="dot", line_color="#475569")
        fig.add_hline(y=median_ltv, line_dash="dot", line_color="#475569")
        fig.add_annotation(x=82,y=q85_ltv,text="🚨 URGENT<br>High Risk+High Value",
                           font=dict(size=9,color="#EF4444"),showarrow=False,
                           bgcolor="#1A0A0A",bordercolor="#EF4444",borderwidth=1)
        fig.add_annotation(x=18,y=q85_ltv,text="🌟 PROTECT<br>Safe+High Value",
                           font=dict(size=9,color="#22C55E"),showarrow=False,
                           bgcolor="#0A1A0A",bordercolor="#22C55E",borderwidth=1)
        sfig(fig, "Churn Risk vs Predicted LTV — Strategic Quadrant Matrix", 420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_fi = fi_churn.head(10).copy()
        top_fi["feature_clean"] = top_fi["feature"].str.replace("_"," ").str.title()
        fig2 = px.bar(top_fi.sort_values("importance"), x="importance", y="feature_clean",
                      orientation="h", color="importance",
                      color_continuous_scale=["#1E3A5F","#1E6FD9","#00C2CB"],
                      labels={"importance":"Importance","feature_clean":""})
        fig2.update_coloraxes(showscale=False)
        sfig(fig2, "Top 10 Churn Drivers (Feature Importance)", 420)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        order_cols = ["order_m1","order_m2","order_m3","order_m4","order_m5","order_m6"]
        months     = ["M1","M2","M3","M4","M5","M6"]
        fig3 = go.Figure()
        for tier, color in [("🔴 Critical","#EF4444"),("🟡 At Risk","#F59E0B"),("🟢 Healthy","#22C55E")]:
            sub = dff[dff["health_tier"]==tier]
            if len(sub) == 0: continue
            avgs = [sub[c].mean() for c in order_cols]
            fig3.add_trace(go.Scatter(x=months, y=avgs, name=tier,
                mode="lines+markers", line=dict(color=color,width=2.5),
                marker=dict(size=7),
                hovertemplate=f"<b>{tier}</b><br>%{{x}}: %{{y:,.0f}} cubes<extra></extra>"))
        sfig(fig3, "6-Month Order Volume Trend by Health Tier", 320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.scatter(dff, x="payment_delay_days", y="churn_risk_model",
                          color="health_tier", color_discrete_map=C_RISK,
                          hover_name="outlet_name",
                          labels={"payment_delay_days":"Payment Delay (Days)","churn_risk_model":"Churn Risk Score"},
                          opacity=0.75)
        sfig(fig4, "Payment Delay vs Churn Risk", 320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>🚨 High-Risk Watchlist (Score ≥ 60)</div>", unsafe_allow_html=True)
    watchlist = dff[dff["churn_risk_model"]>=60].sort_values("churn_risk_model",ascending=False)[[
        "outlet_name","city","outlet_type","churn_risk_model","ltv_predicted_inr",
        "days_since_last_order","complaints_last3m","payment_delay_days","recommended_intervention","health_tier"
    ]].head(30).copy()
    watchlist.columns = ["Outlet","City","Type","Risk Score","LTV (₹)","Days Inactive","Complaints","Delay(d)","Intervention","Status"]
    watchlist["Risk Score"] = watchlist["Risk Score"].apply(lambda x: f"{x:.1f}")
    watchlist["LTV (₹)"]    = watchlist["LTV (₹)"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(watchlist, use_container_width=True, height=360)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LTV & REVENUE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 LTV & Revenue Intel":
    st.markdown("## 💰 LTV & Revenue Intelligence")
    st.markdown("<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>24-month LTV predictions · ₹6/cube @ 46% gross margin</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total 24M LTV Pool",  f"₹{dff['ltv_predicted_inr'].sum()/1e7:.2f} Cr")
    c2.metric("Avg LTV per Outlet",  f"₹{dff['ltv_predicted_inr'].mean()/1e5:.1f}L")
    top20_share = dff.nlargest(max(1,int(len(dff)*0.2)),"ltv_predicted_inr")["ltv_predicted_inr"].sum()
    c3.metric("Top 20% LTV Share",   f"{top20_share/max(dff['ltv_predicted_inr'].sum(),1)*100:.1f}%")
    at_risk_ltv = dff[dff["health_tier"]!="🟢 Healthy"]["ltv_predicted_inr"].sum()
    c4.metric("At-Risk LTV Exposure",f"₹{at_risk_ltv/1e7:.2f} Cr")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(dff, x="outlet_type", y="ltv_predicted_inr",
                     color="outlet_type",
                     color_discrete_sequence=["#1E6FD9","#00C2CB","#22C55E","#F59E0B","#EF4444","#A78BFA","#EC4899"],
                     labels={"ltv_predicted_inr":"Predicted 24M LTV (₹)","outlet_type":""},
                     points="outliers")
        fig.update_layout(showlegend=False)
        sfig(fig, "LTV Distribution by Outlet Type", 360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_ltv = dff.groupby("city")["ltv_predicted_inr"].agg(["mean","std"]).reset_index()
        city_ltv.columns = ["city","mean_ltv","std_ltv"]
        city_ltv = city_ltv.sort_values("mean_ltv", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=city_ltv["city"], y=city_ltv["mean_ltv"],
            error_y=dict(type="data", array=city_ltv["std_ltv"]/2, visible=True, color="#475569"),
            marker_color=["#1E6FD9","#00C2CB","#22C55E","#F59E0B","#EF4444","#A78BFA","#EC4899"][:len(city_ltv)],
            hovertemplate="<b>%{x}</b><br>Avg LTV: ₹%{y:,.0f}<extra></extra>"
        ))
        sfig(fig2, "Average Predicted 24M LTV by City", 360)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Manual trendline — no statsmodels needed
        x_vals = dff["monthly_revenue_inr"].values
        y_vals = dff["ltv_predicted_inr"].values
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

        fig3 = px.scatter(dff, x="monthly_revenue_inr", y="ltv_predicted_inr",
                          color="health_tier", color_discrete_map=C_RISK,
                          hover_name="outlet_name",
                          labels={"monthly_revenue_inr":"Monthly Revenue (₹)","ltv_predicted_inr":"Predicted 24M LTV (₹)"},
                          opacity=0.8)
        fig3.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines",
                                  line=dict(color="#94A3B8", dash="dash", width=1.5),
                                  name="Trend", showlegend=False))
        sfig(fig3, "Monthly Revenue vs Predicted 24M LTV", 320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        top_fi = fi_ltv.head(10).copy()
        top_fi["feature_clean"] = top_fi["feature"].str.replace("_"," ").str.title()
        fig4 = px.bar(top_fi.sort_values("importance"), x="importance", y="feature_clean",
                      orientation="h", color="importance",
                      color_continuous_scale=["#1E3A5F","#00C2CB","#22C55E"],
                      labels={"importance":"Importance","feature_clean":""})
        fig4.update_coloraxes(showscale=False)
        sfig(fig4, "Top 10 LTV Predictors (Feature Importance)", 320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>🏅 Top 20 Highest LTV Accounts</div>", unsafe_allow_html=True)
    top20 = dff.nlargest(20,"ltv_predicted_inr")[[
        "outlet_name","city","outlet_type","ltv_predicted_inr",
        "monthly_revenue_inr","churn_risk_model","health_tier","account_manager"
    ]].copy()
    top20["ltv_predicted_inr"]   = top20["ltv_predicted_inr"].apply(lambda x: f"₹{x:,.0f}")
    top20["monthly_revenue_inr"] = top20["monthly_revenue_inr"].apply(lambda x: f"₹{x:,.0f}")
    top20["churn_risk_model"]    = top20["churn_risk_model"].apply(lambda x: f"{x:.1f}")
    top20.columns = ["Outlet","City","Type","Predicted 24M LTV","Monthly Rev","Risk Score","Status","Acc. Manager"]
    st.dataframe(top20, use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERVENTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Intervention Engine":
    st.markdown("## 🎯 AI Retention Intervention Engine")
    st.markdown("<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>ROI-optimised retention strategy per at-risk outlet</div>", unsafe_allow_html=True)

    at_risk = dff[dff["churn_risk_model"] >= 35].copy()
    if len(at_risk) == 0:
        st.info("No at-risk outlets found with current filters.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Outlets Needing Intervention", len(at_risk))
    c2.metric("Total Intervention Budget",    f"₹{at_risk['intervention_cost_inr'].sum()/1e5:.1f}L")
    c3.metric("Expected Revenue Saved",       f"₹{at_risk['expected_revenue_saved_inr'].sum()/1e5:.1f}L")
    pos_roi = at_risk[at_risk["retention_roi"]>0]["retention_roi"]
    c4.metric("Avg Retention ROI",            f"{pos_roi.mean():.0f}%" if len(pos_roi)>0 else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        iv_dist = at_risk["recommended_intervention"].value_counts().reset_index()
        iv_dist.columns = ["intervention","count"]
        fig = px.bar(iv_dist.sort_values("count"),
                     x="count", y="intervention", orientation="h",
                     color="intervention", color_discrete_map=C_IV,
                     text="count",
                     labels={"count":"# Outlets","intervention":""})
        fig.update_traces(textposition="outside", textfont_color="white")
        fig.update_layout(showlegend=False)
        sfig(fig, "Intervention Type Distribution", 300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        roi_iv = at_risk[at_risk["retention_roi"]>0].groupby("recommended_intervention").agg(
            avg_roi=("retention_roi","mean"),
        ).reset_index().sort_values("avg_roi",ascending=False)
        if len(roi_iv) > 0:
            fig2 = go.Figure(go.Bar(
                x=roi_iv["recommended_intervention"], y=roi_iv["avg_roi"],
                marker_color=[C_IV.get(i,"#666") for i in roi_iv["recommended_intervention"]],
                text=[f"{v:.0f}%" for v in roi_iv["avg_roi"]],
                textposition="outside", textfont_color="white",
                hovertemplate="<b>%{x}</b><br>Avg ROI: %{y:.1f}%<extra></extra>"
            ))
            sfig(fig2, "Average Retention ROI by Intervention Type", 300)
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        total_portfolio_rev = dff["annual_revenue_inr"].sum()
        rev_at_risk_val     = at_risk["annual_revenue_inr"].sum()
        rev_saveable        = at_risk["expected_revenue_saved_inr"].sum()

        fig3 = go.Figure(go.Waterfall(
            orientation="v",
            x=["Total ARR","Revenue at Risk","Intervention Saves","Net Retained ARR"],
            measure=["absolute","relative","relative","total"],
            y=[total_portfolio_rev, -rev_at_risk_val, rev_saveable, 0],
            text=[f"₹{total_portfolio_rev/1e7:.1f}Cr",
                  f"-₹{rev_at_risk_val/1e5:.0f}L",
                  f"+₹{rev_saveable/1e5:.0f}L",
                  f"₹{(total_portfolio_rev-rev_at_risk_val+rev_saveable)/1e7:.1f}Cr"],
            textposition="outside",
            connector=dict(line=dict(color="#1E3A5F",width=1)),
            increasing=dict(marker=dict(color="#22C55E")),
            decreasing=dict(marker=dict(color="#EF4444")),
            totals=dict(marker=dict(color="#1E6FD9")),
        ))
        sfig(fig3, "Revenue Waterfall: Risk vs Retention Impact", 320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.scatter(dff, x="churn_risk_model", y="expected_revenue_saved_inr",
                          size="ltv_predicted_inr", color="health_tier",
                          color_discrete_map=C_RISK,
                          hover_name="outlet_name",
                          labels={"churn_risk_model":"Churn Risk Score",
                                  "expected_revenue_saved_inr":"Expected Revenue Saved (₹)"},
                          size_max=25, opacity=0.8)
        sfig(fig4, "Churn Risk vs Expected Revenue Saved (Size = LTV)", 320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>📋 Full Intervention Recommendation Table</div>", unsafe_allow_html=True)
    show = at_risk.sort_values("retention_priority_index",ascending=False)[[
        "outlet_name","city","outlet_type","churn_risk_model","ltv_predicted_inr",
        "recommended_intervention","intervention_cost_inr","expected_revenue_saved_inr","retention_roi","health_tier"
    ]].head(50).copy()
    show.columns = ["Outlet","City","Type","Risk Score","LTV (₹)","Intervention","Cost (₹)","Saved (₹)","ROI (%)","Status"]
    show["Risk Score"] = show["Risk Score"].apply(lambda x: f"{x:.1f}")
    show["LTV (₹)"]    = show["LTV (₹)"].apply(lambda x: f"₹{x:,.0f}")
    show["Cost (₹)"]   = show["Cost (₹)"].apply(lambda x: f"₹{x:,}")
    show["Saved (₹)"]  = show["Saved (₹)"].apply(lambda x: f"₹{x:,.0f}")
    show["ROI (%)"]    = show["ROI (%)"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(show, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RETENTION PRIORITY INDEX
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Retention Priority Index":
    st.markdown("## 🏆 Retention Priority Index")
    st.markdown("<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>Score = Churn Risk × LTV × Margin (46%) × Intervention Efficiency · Higher = Act Now</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        top20 = dff.nlargest(20,"retention_priority_index")[[
            "outlet_name","city","retention_priority_index","churn_risk_model","ltv_predicted_inr","health_tier"
        ]].copy()
        fig = px.bar(top20.sort_values("retention_priority_index"),
                     x="retention_priority_index", y="outlet_name",
                     orientation="h", color="health_tier", color_discrete_map=C_RISK,
                     hover_data={"city":True,"churn_risk_model":":.1f","ltv_predicted_inr":":,.0f"},
                     labels={"retention_priority_index":"Retention Priority Index","outlet_name":""})
        sfig(fig, "Top 20 Outlets by Retention Priority Index", 500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tc = dff["priority_tier"].value_counts().reset_index()
        tc.columns = ["tier","count"]
        fig2 = go.Figure(go.Pie(
            labels=tc["tier"], values=tc["count"], hole=0.55,
            marker_colors=[C_TIER.get(t,"#666") for t in tc["tier"]],
            textinfo="label+percent+value", textfont=dict(size=11)
        ))
        sfig(fig2, "Priority Tier Distribution", 300)
        st.plotly_chart(fig2, use_container_width=True)

        tier1 = dff[dff["priority_tier"]=="Tier 1 – Urgent"]
        if len(tier1) > 0:
            am = tier1.groupby("account_manager").size().reset_index()
            am.columns = ["Account Manager","Urgent Accounts"]
            am = am.sort_values("Urgent Accounts", ascending=False)
            fig3 = px.bar(am, x="Account Manager", y="Urgent Accounts",
                          color="Urgent Accounts",
                          color_continuous_scale=["#F59E0B","#EF4444"],
                          text="Urgent Accounts")
            fig3.update_traces(textposition="outside", textfont_color="white")
            fig3.update_coloraxes(showscale=False)
            sfig(fig3, "Urgent Accounts per Account Manager", 260)
            st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(
        dff.nlargest(80,"retention_priority_index"),
        x="churn_risk_model", y="ltv_predicted_inr",
        size="retention_priority_index", color="priority_tier",
        color_discrete_map=C_TIER,
        hover_name="outlet_name",
        hover_data={"retention_priority_index":":.2f","city":True},
        labels={"churn_risk_model":"Churn Risk Score","ltv_predicted_inr":"Predicted 24M LTV (₹)"},
        size_max=35
    )
    sfig(fig4, "Top 80 Accounts — Retention Priority Index (Bubble Size = Priority Score)", 380)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>📊 Full Retention Priority Leaderboard</div>", unsafe_allow_html=True)
    rpi = dff.sort_values("retention_priority_index",ascending=False)[[
        "outlet_name","city","outlet_type","retention_priority_index",
        "churn_risk_model","ltv_predicted_inr","priority_tier","recommended_intervention","account_manager"
    ]].head(50).copy()
    rpi.columns = ["Outlet","City","Type","Priority Index","Risk Score","LTV (₹)","Priority Tier","Intervention","Acc. Manager"]
    rpi["Priority Index"] = rpi["Priority Index"].apply(lambda x: f"{x:.2f}")
    rpi["Risk Score"]     = rpi["Risk Score"].apply(lambda x: f"{x:.1f}")
    rpi["LTV (₹)"]        = rpi["LTV (₹)"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(rpi, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT NEW LEAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict New Lead":
    st.markdown("## 🔮 Predict Churn Risk & LTV for a New Outlet")
    st.markdown("<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>Enter outlet details · Get AI-powered churn score, LTV estimate & intervention recommendation</div>", unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Outlet Info**")
            name_inp    = st.text_input("Outlet Name", "New Premium Café")
            city_inp    = st.selectbox("City", sorted(df["city"].unique()))
            type_inp    = st.selectbox("Outlet Type", sorted(df["outlet_type"].unique()))
            cuisine_inp = st.selectbox("Cuisine Type", sorted(df["cuisine_type"].unique()))
        with c2:
            st.markdown("**Profile**")
            seating_inp  = st.slider("Seating Capacity", 10, 300, 80)
            rating_inp   = st.slider("Google Rating", 1.0, 5.0, 4.2, step=0.1)
            insta_inp    = st.number_input("Instagram Followers", 100, 100000, 5000, step=500)
            months_inp   = st.slider("Months Since Onboarding", 1, 36, 6)
        with c3:
            st.markdown("**Risk Signals**")
            flavours_inp  = st.slider("Flavours Ordered", 1, 8, 3)
            comp_inp      = st.slider("Complaints (Last 3M)", 0, 10, 1)
            delay_inp     = st.slider("Payment Delay (Days)", 0, 60, 5)
            am_inp        = st.selectbox("Account Manager Assigned?", ["Yes","No"])
            outreach_inp  = st.slider("Outreach Response (0=None, 3=All)", 0, 3, 2)

        st.markdown("**Monthly Order Volumes (Cubes)**")
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        m1 = mc1.number_input("Month 1", 1000, 25000, 10000, step=500)
        m2 = mc2.number_input("Month 2", 1000, 25000, 9800,  step=500)
        m3 = mc3.number_input("Month 3", 1000, 25000, 9500,  step=500)
        m4 = mc4.number_input("Month 4", 1000, 25000, 9000,  step=500)
        m5 = mc5.number_input("Month 5", 1000, 25000, 8200,  step=500)
        m6 = mc6.number_input("Month 6", 1000, 25000, 7500,  step=500)

        submitted = st.form_submit_button("🔮 Run AI Prediction", use_container_width=True)

    if submitted:
        avg_vol    = int(np.mean([m1,m2,m3,m4,m5,m6]))
        trend_pct  = round((m6 - m1) / max(m1,1) * 100, 2)
        slope      = (m6 - m1) / 5
        volatility = float(np.std([m1,m2,m3,m4,m5,m6]))
        recent_vs  = m6 / max(avg_vol, 1)

        try: city_enc    = label_encoders["city"].transform([city_inp])[0]
        except: city_enc = 0
        try: type_enc    = label_encoders["type"].transform([type_inp])[0]
        except: type_enc = 0
        try: cuisine_enc = label_encoders["cuisine"].transform([cuisine_inp])[0]
        except: cuisine_enc = 0

        X_new = pd.DataFrame([[
            seating_inp, rating_inp, insta_inp, months_inp,
            flavours_inp, comp_inp, delay_inp,
            1 if am_inp=="Yes" else 0, outreach_inp,
            avg_vol, trend_pct, 15,
            slope, volatility, recent_vs,
            city_enc, type_enc, cuisine_enc
        ]], columns=features)

        churn_prob  = churn_model.predict_proba(X_new)[0][1]
        churn_score = round(churn_prob * 100, 1)
        ltv_pred    = int(ltv_model.predict(X_new)[0])
        monthly_rev = avg_vol * 6

        if churn_score >= 70:   tier = "🔴 Critical"
        elif churn_score >= 40: tier = "🟡 At Risk"
        else:                   tier = "🟢 Healthy"

        if churn_score < 35:     iv = "No Intervention Needed"
        elif delay_inp > 25:     iv = "Extended Credit Terms"
        elif comp_inp > 4:       iv = "Dedicated Account Manager"
        elif ltv_pred > 500000:  iv = "Dedicated Account Manager"
        elif outreach_inp == 0:  iv = "Premium Flavour Trial"
        elif churn_score > 70:   iv = "5% Discount Offer"
        else:                    iv = "Premium Flavour Trial"

        iv_cost_map = {"No Intervention Needed":0, "5% Discount Offer":int(monthly_rev*0.05),
                       "Premium Flavour Trial":500, "Extended Credit Terms":int(monthly_rev*0.02),
                       "Dedicated Account Manager":8000}
        iv_eff_map  = {"No Intervention Needed":0, "5% Discount Offer":0.55,
                       "Premium Flavour Trial":0.65, "Extended Credit Terms":0.70,
                       "Dedicated Account Manager":0.85}
        cost  = iv_cost_map.get(iv, 0)
        saved = int(ltv_pred * churn_prob * iv_eff_map.get(iv, 0))
        roi   = round((saved - cost) / max(cost,1) * 100, 1) if cost > 0 else 0.0
        rpi   = round(churn_prob * (ltv_pred / max(df["ltv_predicted_inr"].max(),1)) * 0.46 * iv_eff_map.get(iv,0) * 100, 2)

        st.markdown(f"### 📋 Results for **{name_inp}**")
        r1,r2,r3,r4,r5 = st.columns(5)
        r1.metric("Churn Risk Score",     f"{churn_score}", tier)
        r2.metric("Predicted 24M LTV",    f"₹{ltv_pred:,.0f}")
        r3.metric("Monthly Revenue Est.", f"₹{monthly_rev:,.0f}")
        r4.metric("Retention Priority",   f"{rpi:.2f}")
        r5.metric("Recommended Action",   iv)

        col_g, col_i = st.columns([1, 1.4])
        with col_g:
            gauge_color = "#EF4444" if churn_score>=70 else "#F59E0B" if churn_score>=40 else "#22C55E"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_score,
                title={"text":"Churn Risk Score","font":{"size":14,"color":"#E8F0FE"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#94A3B8"},
                    "bar":{"color":gauge_color},
                    "steps":[{"range":[0,40],"color":"#0D2A18"},
                              {"range":[40,70],"color":"#2A1F0D"},
                              {"range":[70,100],"color":"#2A0D0D"}],
                }
            ))
            fig_g.update_layout(**BASE, height=260)
            st.plotly_chart(fig_g, use_container_width=True)

        with col_i:
            st.markdown(f"""
            <div style='background:#162032;border:1px solid #1E3A5F;border-radius:12px;padding:1.4rem;margin-top:0.3rem;'>
            <div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#94A3B8;margin-bottom:0.8rem;'>AI RECOMMENDATION</div>
            <div style='margin-bottom:0.6rem;'><span style='color:#94A3B8;font-size:0.8rem;'>Health Status</span><br>
            <span style='font-size:1.1rem;font-weight:600;'>{tier}</span></div>
            <div style='margin-bottom:0.6rem;'><span style='color:#94A3B8;font-size:0.8rem;'>Intervention Strategy</span><br>
            <span style='font-size:1rem;font-weight:600;color:#4A9FFF;'>{iv}</span></div>
            <div style='display:flex;gap:1.5rem;margin-top:0.8rem;flex-wrap:wrap;'>
            <div><span style='color:#94A3B8;font-size:0.75rem;'>Cost</span><br><b>₹{cost:,}</b></div>
            <div><span style='color:#94A3B8;font-size:0.75rem;'>Revenue Saved</span><br><b style='color:#22C55E;'>₹{saved:,.0f}</b></div>
            <div><span style='color:#94A3B8;font-size:0.75rem;'>Retention ROI</span><br><b style='color:#4A9FFF;'>{roi:.1f}%</b></div>
            </div>
            <div style='margin-top:0.9rem;padding:0.6rem;background:#0D1B2A;border-radius:8px;font-size:0.82rem;color:#CBD5E1;'>
            <b>Order Trend:</b> {trend_pct:+.1f}% over 6 months
            {"— ⚠️ Declining. Early intervention critical." if trend_pct<-10
             else "— ✅ Stable order pattern." if abs(trend_pct)<10
             else "— 📈 Growing account."}
            </div></div>""", unsafe_allow_html=True)

        trend_color = "#4A9FFF" if trend_pct >= 0 else "#EF4444"
        fig_t = go.Figure(go.Scatter(
            x=["M1","M2","M3","M4","M5","M6"],
            y=[m1,m2,m3,m4,m5,m6],
            mode="lines+markers+text",
            text=[f"{v:,}" for v in [m1,m2,m3,m4,m5,m6]],
            textposition="top center", textfont=dict(size=10),
            line=dict(color=trend_color, width=2.5),
            marker=dict(size=8, color=trend_color),
            fill="tozeroy",
            fillcolor=f"rgba({'30,111,217' if trend_pct>=0 else '239,68,68'},0.1)"
        ))
        sfig(fig_t, f"Order Volume Trend — {name_inp}", 240)
        st.plotly_chart(fig_t, use_container_width=True)
