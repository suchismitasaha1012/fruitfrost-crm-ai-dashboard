"""
FruitFrost CRM — ML Model Training
Trains: (1) Churn Prediction, (2) LTV Regression
Outputs: model files + feature importance CSVs
Run: python train_models.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             mean_absolute_error, r2_score, confusion_matrix)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv("fruitfrost_outlets.csv")
print(f"Loaded {len(df)} records\n")

# ─── Feature Engineering ──────────────────────────────────────────────────────
# Encode categoricals
le_city    = LabelEncoder()
le_type    = LabelEncoder()
le_cuisine = LabelEncoder()

df["city_enc"]    = le_city.fit_transform(df["city"])
df["type_enc"]    = le_type.fit_transform(df["outlet_type"])
df["cuisine_enc"] = le_cuisine.fit_transform(df["cuisine_type"])

# Order trend features
df["order_slope"]      = (df["order_m6"] - df["order_m1"]) / 5
df["order_volatility"] = df[["order_m1","order_m2","order_m3",
                              "order_m4","order_m5","order_m6"]].std(axis=1)
df["recent_vs_avg"]    = df["order_m6"] / (df["avg_monthly_cubes"] + 1)

FEATURES = [
    "seating_capacity", "google_rating", "instagram_followers",
    "months_onboarded", "flavours_ordered", "complaints_last3m",
    "payment_delay_days", "am_assigned", "outreach_response_rate",
    "avg_monthly_cubes", "order_trend_pct", "days_since_last_order",
    "order_slope", "order_volatility", "recent_vs_avg",
    "city_enc", "type_enc", "cuisine_enc"
]

X = df[FEATURES]
y_churn = df["churned"]
y_ltv   = df["ltv_24m_inr"]

# ─── MODEL 1: Churn Prediction ────────────────────────────────────────────────
print("=" * 50)
print("MODEL 1 — CHURN PREDICTION (Random Forest)")
print("=" * 50)

X_tr, X_te, y_tr, y_te = train_test_split(X, y_churn, test_size=0.2,
                                           random_state=42, stratify=y_churn)

churn_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42
)
churn_model.fit(X_tr, y_tr)

y_pred  = churn_model.predict(X_te)
y_proba = churn_model.predict_proba(X_te)[:, 1]

print(classification_report(y_te, y_pred))
print(f"ROC-AUC Score : {roc_auc_score(y_te, y_proba):.4f}")
cv_scores = cross_val_score(churn_model, X, y_churn, cv=5, scoring="roc_auc")
print(f"5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
fi_churn = pd.DataFrame({
    "feature": FEATURES,
    "importance": churn_model.feature_importances_
}).sort_values("importance", ascending=False)
fi_churn.to_csv("feature_importance_churn.csv", index=False)
print(f"\nTop 5 Churn Drivers:")
print(fi_churn.head())

# ─── MODEL 2: LTV Prediction ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL 2 — LTV PREDICTION (Random Forest Regressor)")
print("=" * 50)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_ltv, test_size=0.2,
                                               random_state=42)

ltv_model = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42
)
ltv_model.fit(X_tr2, y_tr2)

y_pred_ltv = ltv_model.predict(X_te2)
mae = mean_absolute_error(y_te2, y_pred_ltv)
r2  = r2_score(y_te2, y_pred_ltv)
print(f"MAE  : ₹{mae:,.0f}")
print(f"R²   : {r2:.4f}")

fi_ltv = pd.DataFrame({
    "feature": FEATURES,
    "importance": ltv_model.feature_importances_
}).sort_values("importance", ascending=False)
fi_ltv.to_csv("feature_importance_ltv.csv", index=False)
print(f"\nTop 5 LTV Drivers:")
print(fi_ltv.head())

# ─── Generate Predictions on Full Dataset ────────────────────────────────────
df["churn_prob_model"]  = churn_model.predict_proba(X)[:, 1]
df["churn_risk_model"]  = np.round(df["churn_prob_model"] * 100, 1)
df["ltv_predicted_inr"] = ltv_model.predict(X).astype(int)

# Save enriched dataset
df.to_csv("fruitfrost_outlets_scored.csv", index=False)
print(f"\n✅ Enriched dataset saved: fruitfrost_outlets_scored.csv")

# ─── Save Models ──────────────────────────────────────────────────────────────
with open("churn_model.pkl", "wb") as f:
    pickle.dump(churn_model, f)
with open("ltv_model.pkl", "wb") as f:
    pickle.dump(ltv_model, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump({"city": le_city, "type": le_type, "cuisine": le_cuisine}, f)
with open("feature_list.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("✅ Models saved: churn_model.pkl, ltv_model.pkl")
print("\nAll done! Now run: streamlit run app.py")
