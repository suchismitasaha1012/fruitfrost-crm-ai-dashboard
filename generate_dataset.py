"""
FruitFrost CRM Dataset Generator
Generates 300 realistic B2B outlet records for AI-driven retention analysis
Run: python generate_dataset.py
"""

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

N = 300

# ─── Basic Info ───────────────────────────────────────────────────────────────
outlet_types = ["Premium Café", "Bar & Lounge", "5-Star Hotel", "4-Star Hotel",
                "Fine Dining Restaurant", "Cloud Kitchen", "Catering Service"]
cities = ["Delhi", "Mumbai", "Bangalore", "Ranchi", "Hyderabad", "Pune", "Chennai"]
cuisines = ["Continental", "Pan-Asian", "Indian", "Mediterranean", "Multi-Cuisine",
            "Italian", "Japanese", "Fusion"]
account_managers = ["Riya Sharma", "Arjun Mehta", "Priya Nair", "Vikram Singh", "Neha Joshi"]

outlet_type_weights = [0.30, 0.25, 0.10, 0.12, 0.12, 0.06, 0.05]
city_weights        = [0.22, 0.20, 0.18, 0.10, 0.12, 0.10, 0.08]

outlet_names_pool = [
    "The Blue Petal Café", "Skyline Bar", "Monarch Grand Hotel", "Spice Route",
    "The Artisan Brew", "Fusion 9", "Cloud Nine Lounge", "The Grand Palate",
    "Saffron Kitchen", "Urban Brew Co.", "The Cellar Bar", "Pearl Continental",
    "Breezy Bites", "Metro Mixology", "The Social Hub", "Terrace & Toast",
    "Altitude Lounge", "The Copper Pot", "Velvet Room", "Zest & Zeal",
    "Neon Nights Bar", "The Garden Table", "Harbour View", "Crimson Fork",
    "Olive & Oak", "The Rooftop Sip", "Downtown Diner", "Bombay Blend",
    "The Mango Tree", "Latitude 28", "The Ice House", "Pearl & Petal",
    "Mosaic Café", "The Black Label", "Sundowner Bar", "Spice Symphony",
    "Casa Milano", "The Lemon Tree", "Bourbon Street", "Café Nuvola"
]

# Generate 300 unique names by adding suffixes
all_names = []
used = set()
suffixes = ['', ' II', ' III', ' IV', ' V', ' VI', ' VII', ' VIII']
for i in range(N):
    base = outlet_names_pool[i % len(outlet_names_pool)]
    suffix_idx = i // len(outlet_names_pool)
    suffix = suffixes[suffix_idx] if suffix_idx < len(suffixes) else f" {suffix_idx+1}"
    name = base + suffix
    all_names.append(name)

outlet_type_arr  = np.random.choice(outlet_types, N, p=outlet_type_weights)
city_arr         = np.random.choice(cities, N, p=city_weights)
cuisine_arr      = np.random.choice(cuisines, N)
am_arr           = np.random.choice(account_managers, N)

# ─── Numerical Features (anchored to project economics) ───────────────────────
# Monthly cubes per outlet baseline = 13,500 (from project slide)
seating_cap      = np.random.randint(20, 300, N)
google_rating    = np.round(np.random.uniform(3.2, 5.0, N), 1)
instagram_flw    = np.random.randint(500, 85000, N)
months_onboarded = np.random.randint(1, 36, N)
flavours_ordered = np.random.randint(1, 8, N)
complaints_last3m= np.random.randint(0, 8, N)
payment_delay_days = np.random.randint(0, 45, N)
am_assigned      = np.random.choice([0, 1], N, p=[0.35, 0.65])
outreach_response= np.random.randint(0, 4, N)  # how many of last 3 outreach attempts responded

# Monthly order volumes — 6 months (M1=oldest, M6=most recent)
base_volume = np.random.randint(5000, 22000, N)

# Simulate declining trend for ~27% of outlets (at-risk / churned)
trend_factor = np.random.choice(
    [1.0, 0.95, 0.90, 0.80, 0.65, 0.50],
    N,
    p=[0.45, 0.15, 0.13, 0.10, 0.10, 0.07]
)

m1 = (base_volume * np.random.uniform(0.88, 1.12, N)).astype(int)
m2 = (m1 * np.random.uniform(0.92, 1.08, N)).astype(int)
m3 = (m2 * np.random.uniform(0.92, 1.08, N)).astype(int)
m4 = (m3 * trend_factor * np.random.uniform(0.90, 1.05, N)).astype(int)
m5 = (m4 * trend_factor * np.random.uniform(0.90, 1.05, N)).astype(int)
m6 = (m5 * trend_factor * np.random.uniform(0.88, 1.02, N)).astype(int)
m6 = np.clip(m6, 500, 25000)

avg_order_vol = ((m1+m2+m3+m4+m5+m6)/6).astype(int)
order_trend   = np.round((m6 - m1) / (m1 + 1) * 100, 2)  # % change M1→M6

days_since_last_order = np.random.randint(1, 90, N)
# At-risk outlets have higher days since last order
days_since_last_order = np.where(trend_factor < 0.75,
                                  days_since_last_order * np.random.uniform(1.5, 3.0, N),
                                  days_since_last_order).astype(int)
days_since_last_order = np.clip(days_since_last_order, 1, 120)

# ─── Churn Label (realistic logic) ───────────────────────────────────────────
churn_score_raw = (
    (trend_factor < 0.70).astype(int) * 3 +
    (complaints_last3m > 4).astype(int) * 2 +
    (payment_delay_days > 25).astype(int) * 2 +
    (days_since_last_order > 45).astype(int) * 2 +
    (outreach_response == 0).astype(int) * 1 +
    (google_rating < 3.8).astype(int) * 1 +
    (am_assigned == 0).astype(int) * 1
)
churn_prob = churn_score_raw / churn_score_raw.max()
churned = (churn_prob > 0.55).astype(int)
# Force ~17% churn rate (realistic for B2B SaaS/subscription)
churn_threshold = np.percentile(churn_prob, 83)
churned = (churn_prob >= churn_threshold).astype(int)

# ─── Revenue & LTV (anchored to ₹6/cube, 46% margin) ─────────────────────────
price_per_cube   = 6.0
cost_per_cube    = 3.24
margin_per_cube  = 2.76

monthly_revenue  = np.round(avg_order_vol * price_per_cube, 0).astype(int)
monthly_gp       = np.round(avg_order_vol * margin_per_cube, 0).astype(int)
annual_revenue   = monthly_revenue * 12

# LTV = monthly GP × predicted remaining months (based on churn risk)
predicted_remaining_months = np.where(
    churned == 1,
    np.random.randint(1, 8, N),
    np.random.randint(12, 36, N)
)
ltv_24m = np.round(monthly_gp * np.minimum(predicted_remaining_months, 24), 0).astype(int)

# ─── Churn Probability Score (0–100) ──────────────────────────────────────────
churn_risk_score = np.round(churn_prob * 100, 1)

# ─── Intervention Recommendation Logic ───────────────────────────────────────
def recommend_intervention(row):
    risk  = row["churn_risk_score"]
    ltv   = row["ltv_24m_inr"]
    delay = row["payment_delay_days"]
    comp  = row["complaints_last3m"]
    resp  = row["outreach_response_rate"]

    if risk < 35:
        return "No Intervention Needed"
    if delay > 25:
        return "Extended Credit Terms"
    if comp > 4:
        return "Dedicated Account Manager"
    if ltv > 500000:
        return "Dedicated Account Manager"
    if resp == 0:
        return "Premium Flavour Trial"
    if risk > 70:
        return "5% Discount Offer"
    return "Premium Flavour Trial"

intervention_costs = {
    "No Intervention Needed": 0,
    "5% Discount Offer": None,           # calculated below
    "Premium Flavour Trial": 500,
    "Extended Credit Terms": None,       # calculated below
    "Dedicated Account Manager": 8000
}

# ─── Assemble DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame({
    "outlet_id":             [f"FF-{1000+i}" for i in range(N)],
    "outlet_name":           all_names,
    "outlet_type":           outlet_type_arr,
    "city":                  city_arr,
    "cuisine_type":          cuisine_arr,
    "account_manager":       am_arr,
    "seating_capacity":      seating_cap,
    "google_rating":         google_rating,
    "instagram_followers":   instagram_flw,
    "months_onboarded":      months_onboarded,
    "flavours_ordered":      flavours_ordered,
    "complaints_last3m":     complaints_last3m,
    "payment_delay_days":    payment_delay_days,
    "am_assigned":           am_assigned,
    "outreach_response_rate": outreach_response,
    "order_m1":              m1,
    "order_m2":              m2,
    "order_m3":              m3,
    "order_m4":              m4,
    "order_m5":              m5,
    "order_m6":              m6,
    "avg_monthly_cubes":     avg_order_vol,
    "order_trend_pct":       order_trend,
    "days_since_last_order": days_since_last_order,
    "monthly_revenue_inr":   monthly_revenue,
    "monthly_gp_inr":        monthly_gp,
    "annual_revenue_inr":    annual_revenue,
    "ltv_24m_inr":           ltv_24m,
    "churn_risk_score":      churn_risk_score,
    "churned":               churned,
})

# Add intervention column
df["recommended_intervention"] = df.apply(recommend_intervention, axis=1)

# Intervention cost
def calc_intervention_cost(row):
    iv = row["recommended_intervention"]
    if iv == "No Intervention Needed":    return 0
    if iv == "5% Discount Offer":         return round(row["monthly_revenue_inr"] * 0.05)
    if iv == "Premium Flavour Trial":     return 500
    if iv == "Extended Credit Terms":     return round(row["monthly_revenue_inr"] * 0.02)
    if iv == "Dedicated Account Manager": return 8000
    return 0

df["intervention_cost_inr"] = df.apply(calc_intervention_cost, axis=1)

# Expected revenue saved = LTV × (1 - churn_risk/100) — simplified retention uplift
df["expected_revenue_saved_inr"] = np.where(
    df["recommended_intervention"] == "No Intervention Needed",
    0,
    (df["ltv_24m_inr"] * (df["churn_risk_score"] / 100) * 0.60).astype(int)
)

df["retention_roi"] = np.where(
    df["intervention_cost_inr"] > 0,
    np.round((df["expected_revenue_saved_inr"] - df["intervention_cost_inr"]) /
             (df["intervention_cost_inr"] + 1) * 100, 1),
    0.0
)

# Retention Priority Index = churn_risk × ltv × margin × intervention_efficiency
intervention_efficiency = {
    "No Intervention Needed": 0,
    "5% Discount Offer": 0.55,
    "Premium Flavour Trial": 0.65,
    "Extended Credit Terms": 0.70,
    "Dedicated Account Manager": 0.85,
}
df["intervention_efficiency"] = df["recommended_intervention"].map(intervention_efficiency)
df["retention_priority_index"] = np.round(
    (df["churn_risk_score"] / 100) *
    (df["ltv_24m_inr"] / df["ltv_24m_inr"].max()) *
    0.46 *
    df["intervention_efficiency"] * 100, 2
)

# Health tier
def health_tier(score):
    if score >= 70: return "🔴 Critical"
    if score >= 40: return "🟡 At Risk"
    return "🟢 Healthy"

df["health_tier"] = df["churn_risk_score"].apply(health_tier)

# Priority tier
rpi_max = df["retention_priority_index"].max()
df["priority_tier"] = pd.cut(
    df["retention_priority_index"],
    bins=[-0.01, rpi_max*0.33, rpi_max*0.66, rpi_max+0.01],
    labels=["Tier 3 – Monitor", "Tier 2 – Engage", "Tier 1 – Urgent"]
)

df.to_csv("fruitfrost_outlets.csv", index=False)
print(f"✅ Dataset generated: {len(df)} outlets")
print(f"   Churned: {df['churned'].sum()} ({df['churned'].mean()*100:.1f}%)")
print(f"   Critical risk: {(df['health_tier']=='🔴 Critical').sum()}")
print(f"   At risk: {(df['health_tier']=='🟡 At Risk').sum()}")
print(f"   Healthy: {(df['health_tier']=='🟢 Healthy').sum()}")
print(f"   Total ARR: ₹{df['annual_revenue_inr'].sum()/1e7:.2f} crore")
print(f"   Revenue at risk: ₹{df[df['health_tier']=='🔴 Critical']['annual_revenue_inr'].sum()/1e5:.1f} lakh")
print("\nSaved to: fruitfrost_outlets.csv")
