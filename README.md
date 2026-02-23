# 🧊 FruitFrost CRM Intelligence Dashboard
## AI-Driven Retention & Revenue Optimisation for B2B Clients

**IIM Ranchi · Entrepreneurial Marketing · Group 2**
**Role: CRM Manager · Student: Suchismita Saha (M105-23)**

---

## Problem Statement
> "How can AI be used to predict churn risk, estimate customer lifetime value, and optimise retention interventions to maximise long-term revenue for FruitFrost's B2B clients?"

---

## What This App Does

### 3 AI Engines:
1. **Churn Prediction** — Random Forest Classifier (AUC: 0.962) predicts which outlets will stop ordering within 60 days
2. **LTV Prediction** — Random Forest Regressor estimates 24-month revenue value per outlet
3. **Intervention Simulation** — ROI-optimised strategy per at-risk account (Discount / Trial / Credit / Account Manager)

### 5 Dashboard Pages:
| Page | What You See |
|------|-------------|
| 📊 Executive Summary | Portfolio KPIs, health distribution, city risk map |
| ⚠️ Churn Risk Analysis | Risk-LTV scatter, feature importance, watchlist |
| 💰 LTV & Revenue Intel | LTV by outlet type/city, top 20 accounts |
| 🎯 Intervention Engine | ROI table, waterfall chart, budget vs savings |
| 🏆 Retention Priority Index | Composite score leaderboard, account manager load |
| 🔮 Predict New Lead | Real-time prediction for any new outlet |

---

## Setup & Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate dataset
```bash
python generate_dataset.py
```

### Step 3: Train ML models
```bash
python train_models.py
```

### Step 4: Launch dashboard
```bash
streamlit run app.py
```

---

## Key Financial Assumptions (From Group Project)
- Price per cube: ₹6.00
- Cost per cube: ₹3.24
- Gross margin: 46% (₹2.76/cube)
- Baseline monthly cubes per outlet: 13,500
- Annual revenue per outlet: ₹9.72 lakh
- Annual gross profit per outlet: ₹4.47 lakh

---

## Model Performance
| Model | Metric | Score |
|-------|--------|-------|
| Churn Prediction | ROC-AUC | 0.962 |
| Churn Prediction | Accuracy | 90% |
| Churn Prediction | 5-Fold CV AUC | 0.962 ± 0.031 |
| LTV Prediction | R² | 0.674 |
| LTV Prediction | MAE | ₹1.57L |

---

## Files
```
├── app.py                          # Main Streamlit dashboard
├── generate_dataset.py             # Synthetic dataset generator
├── train_models.py                 # ML model training script
├── requirements.txt                # Python dependencies
├── fruitfrost_outlets.csv          # Raw synthetic dataset (300 outlets)
├── fruitfrost_outlets_scored.csv   # Dataset with model predictions
├── churn_model.pkl                 # Trained churn classifier
├── ltv_model.pkl                   # Trained LTV regressor
├── label_encoders.pkl              # Fitted label encoders
├── feature_list.pkl                # Feature names list
├── feature_importance_churn.csv    # Churn model feature importances
└── feature_importance_ltv.csv      # LTV model feature importances
```

---

## Deploying to Streamlit Cloud
1. Push all files to a GitHub repository
2. Go to share.streamlit.io
3. Connect your repo, set `app.py` as main file
4. Deploy!

> **Note**: All `.pkl`, `.csv` files must be included in the repo for deployment.
