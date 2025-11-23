# 🚀 DoorDash Delivery Time Prediction

> **Production-ready ML model achieving 11.04-minute MAE with 57.39% accuracy in predicting delivery times**

An end-to-end machine learning solution that predicts DoorDash delivery duration from order creation to customer delivery, optimized for real-time customer-facing applications.

---

## 📊 Key Results

| Metric | Value | Status |
|--------|-------|--------|
| **Mean Absolute Error** | 11.04 minutes (662.43 sec) | ✅ |
| **R² Score** | 0.307 | ✅ |
| **Accuracy (±10 min)** | 57.39% | ✅ |
| **Extreme Error Rate** | 4.71% | ✅ (<5% guardrail) |
| **Statistical Validation** | p-value < 0.001 (McNemar's test) | ✅ |

---

## 🎯 Problem Statement

DoorDash needs accurate real-time predictions of **total delivery duration** (seconds between order creation and delivery) to:
- Provide reliable ETAs to customers
- Optimize dasher allocation
- Enable proactive operations management
- Reduce customer complaints and churn

**Challenge:** Complex marketplace dynamics with non-linear relationships between features including order details, restaurant characteristics, and real-time marketplace load.

---

## 🛠️ Technical Approach

### Architecture Overview

```
Data Pipeline → Feature Engineering → Model Training → Validation → Deployment Ready
     ↓              ↓                      ↓              ↓            ↓
 197K records   Temporal features     XGBoost        SHAP +       API-ready
 99.3% clean    Cyclic encoding       Tuned          McNemar      <200ms latency
```

### Key Features

**📈 Simple & Effective Feature Engineering**
- Cyclic encoding for temporal features (hour, day of week)
- No complex transformations (no log, no scaling - tree models don't need them)
- Label encoding for categorical features
- Focus on interpretability and production simplicity

**🧠 Iterative Model Development**
- 5 model iterations: Drop Missing → Simple Imputation → XGBoost Native → Temporal Features → Hyperparameter Tuning
- XGBoost selected for native missing value handling and non-linear pattern capture
- RandomizedSearchCV with TimeSeriesSplit for robust hyperparameter tuning

**✅ Rigorous Validation**
- Temporal train-test split (80/20) to prevent data leakage
- Business-aligned metrics (MAE in minutes, accuracy rate, extreme error rate)
- SHAP analysis for model explainability
- McNemar's test for statistical significance validation

---

## 📁 Repository Structure

```
doordash-delivery-time-prediction/
│
├── notebooks/
│   └── doordash_delivery_prediction.ipynb   # Complete analysis & modeling
│
├── data/
│   └── historical_data.csv                   # Training dataset (197K records)
│
├── models/
│   └── doordash_eta_xgb_artifacts.joblib    # Saved model + encoders + feature schema
│
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MohammedMoseena/doordash-delivery-time-prediction.git
cd doordash-delivery-time-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook**
```bash
jupyter notebook notebooks/doordash_delivery_prediction.ipynb
```

### Using the Trained Model

```python
import joblib
import pandas as pd
import numpy as np

# Load saved artifacts
artifacts = joblib.load("models/doordash_eta_xgb_artifacts.joblib")
model = artifacts["model"]
le_store_cat = artifacts["store_category_encoder"]
feature_columns = artifacts["feature_columns"]

# Example order
new_order = {
    "created_at": "2025-11-23 12:34:56",
    "market_id": 3,
    "store_id": 12345,
    "store_primary_category": "Pizza",
    "order_protocol": 2,
    "total_items": 5,
    "num_distinct_items": 3,
    "subtotal": 2400,
    "min_item_price": 200,
    "max_item_price": 800,
    "total_onshift_dashers": 25,
    "total_busy_dashers": 15,
    "total_outstanding_orders": 30,
    "estimated_order_place_duration": 600,
    "estimated_store_to_consumer_driving_duration": 900,
}

# Preprocess and predict
df_new = pd.DataFrame([new_order])
X_new = preprocess_for_inference(df_new)  # See notebook for function
pred_seconds = model.predict(X_new)[0]

print(f"Predicted ETA: {pred_seconds/60:.2f} minutes")
```

---

## 📊 Data Overview

### Dataset Statistics
- **Records:** 197,428 → 196,076 after cleaning (99.3% retained)
- **Features:** 18 features (14 original + 4 engineered temporal features)
- **Target:** `delivery_duration_seconds = actual_delivery_time - created_at`
- **Time Period:** 2015 data (US/Pacific timezone)

### Feature Categories

| Category | Features | Examples |
|----------|----------|----------|
| **Time** | 1 | `market_id` (city/region) |
| **Store** | 3 | `store_id`, `store_primary_category`, `order_protocol` |
| **Order Details** | 5 | `total_items`, `subtotal`, `num_distinct_items`, `min/max_item_price` |
| **Marketplace** | 3 | `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders` |
| **Model Predictions** | 2 | `estimated_order_place_duration`, `estimated_store_to_consumer_driving_duration` |
| **Temporal (Engineered)** | 4 | `hour_sin/cos`, `dayofweek_sin/cos` |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Analyzed 197,428 historical delivery records
- Identified right-skewed distributions in all numeric features (except driving duration)
- Discovered low correlations (<0.30) confirming non-linear relationships
- Found marketplace "inconsistencies" (`busy_dashers > onshift_dashers`) are real operational signals, not errors

### 2. Data Cleaning
```python
# Key cleaning steps:
✓ No duplicates found
✓ Removed min_price > max_price (0.4% of data)
✓ Removed prices > subtotal (0.3% of data)
✓ Removed extreme outliers: total_items=411, prices=14700
✓ Removed impossible deliveries >12 hours (only 7 records)
✓ Kept marketplace inconsistencies (real operational signal)
✓ Final dataset: 196,076 records (99.3% retention)
```

### 3. Missing Value Strategies (3 Approaches)

| Model | Approach | Pros | Cons |
|-------|----------|------|------|
| **Model 1** | Drop missing rows | Clean data, no noise | Loses ~10% data |
| **Model 2** | Simple imputation (median/mode) | Fast, keeps all data | Adds mild noise |
| **Model 3** | XGBoost native handling | Best for marketplace data | Slightly complex |

**Winner:** Model 3 (XGBoost handles missing values optimally)

### 4. Feature Engineering

**Temporal Features (Model 4 improvement):**
```python
# Cyclic encoding preserves temporal continuity
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
dayofweek_sin = sin(2π × dayofweek / 7)
dayofweek_cos = cos(2π × dayofweek / 7)
```

**Why cyclic?** 11 PM is closer to 12 AM than to 3 PM

### 5. Model Development (5 Iterations)

| Model | Strategy | MAE (sec) | R² | Accuracy ≤10min | Key Insight |
|-------|----------|-----------|----|--------------------|-------------|
| Model 1 | Drop missing + RF | 705.60 | 0.223 | 54.02% | Baseline |
| Model 2 | Simple imputation + RF | 714.56 | 0.227 | 53.26% | Keeps data but adds noise |
| Model 3 | XGBoost native missing | **684.43** | **0.274** | **55.83%** | Best missing handling |
| Model 4 | + Temporal features | **668.21** | **0.296** | **57.24%** | Time patterns matter! |
| **Model 5** | + Hyperparameter tuning | **662.43** | **0.307** | **57.39%** | **Production model** |

### 6. Hyperparameter Tuning (Model 5)
```python
Best parameters:
├── n_estimators: 300
├── max_depth: 5
├── learning_rate: 0.1
├── subsample: 0.9
└── colsample_bytree: 0.9

Method: RandomizedSearchCV (20 iterations)
CV Strategy: TimeSeriesSplit (3 folds)
Result: +6 seconds MAE improvement, meets all guardrails
```

---

## 📈 Results & Insights

### Final Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 662.43 sec (11.04 min) | Average prediction error |
| **RMSE** | 967.79 sec (16.13 min) | Penalizes large errors |
| **R²** | 0.307 | Explains 30.7% of variance |
| **Accuracy ≤10 min** | 57.39% | Over half highly accurate |
| **Extreme errors >30 min** | 4.71% | Meets <5% safety threshold |

### SHAP Analysis: Top 5 Predictive Features

| Rank | Feature | Impact | Business Interpretation |
|------|---------|--------|------------------------|
| 1 🥇 | **total_outstanding_orders** | 🔴 Very Strong + | High demand → congestion → delays |
| 2 🥈 | **total_onshift_dashers** | 🟢 Very Strong - | More dashers → faster pickups |
| 3 🥉 | **estimated_store_to_consumer_driving_duration** | 🔴 Strong + | Distance = delivery time |
| 4 | **hour_sin** | 🔴 Moderate + | Peak hours cause delays |
| 5 | **subtotal** | 🔴 Moderate + | Large orders = longer prep |

**Key Insight:** Supply-demand imbalance (#1 and #2) is the primary driver of delivery delays.

### Statistical Validation (McNemar's Test)

```python
Question: Is Model 5 significantly better than Model 1?
Test: McNemar's test (paired predictions)
Result: p-value = 2×10⁻⁶ (<< 0.05)

✅ Conclusion: Improvement is statistically significant, not random chance
```

---

## 💼 Business Impact

### Actionable Recommendations

**1. 🚗 Optimize Dasher Deployment**
- Deploy more dashers when `total_outstanding_orders` is high
- Target markets with consistently high marketplace load
- **Expected Impact:** Reduce average delivery time

**2. ⏱️ Dynamic ETA Buffers**
- Add safety margins during peak hours (`hour_sin` patterns)
- Personalize buffers based on `subtotal` (order complexity)
- **Expected Impact:** Reduction in customer complaints

**3. 🚨 Proactive Monitoring**
- Auto-flag orders predicted >45 min for ops review
- Real-time alerts when marketplace saturation detected
- **Expected Impact:** Faster response to delays

**4. 🤝 Restaurant Partnerships**
- Share `estimated_order_place_duration` insights
- Collaborate on reducing order receiving time variability
- **Expected Impact:**  Improvement in prep time accuracy

### Expected Business Outcomes
- ✅ **Customer Satisfaction:** Fewer missed ETAs → higher trust
- ✅ **Operational Efficiency:** Better dasher allocation → lower costs
- ✅ **Revenue Growth:** Improved retention → reduced churn
- ✅ **Competitive Advantage:** Industry-leading ETA accuracy

---

## ⚠️ Limitations & Future Work

### Current Limitations
- 📅 **Static 2015 dataset** - Marketplace has evolved
- 🚦 **No real-time traffic data** - Weather, accidents, road closures
- 🍕 **Missing restaurant capacity** - Kitchen busyness, staff levels
- 📊 **Accuracy ceiling at ~57%** - Inherent data limitations

### Planned Improvements
```python
Future enhancements:
├── Integrate real-time traffic APIs (Google Maps, Waze)
├── Add weather conditions (rain, snow → slower deliveries)
├── Include restaurant historical prep time patterns
├── Build market-specific models (different cities, different patterns)
├── Weekly retraining pipeline (capture seasonality, drift)
├── A/B testing framework (5% traffic rollout before full deployment)
└── MLOps monitoring (Prometheus/Grafana for drift detection)
```

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **ML/DL** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP |
| **Validation** | TimeSeriesSplit, RandomizedSearchCV, McNemar's Test |
| **Deployment** | Joblib (model serialization) |

---

## 📚 Dependencies

```
# Core Data Science Libraries
numpy==2.0.2
pandas==2.2.2

# Visualization
matplotlib==3.10.0
seaborn==0.13.2

# Machine Learning
scikit-learn==1.6.1
xgboost==3.1.1

# Model Explainability
shap==0.50.0

# Model Serialization
joblib==1.5.2

# Statistical Testing
statsmodels==0.14.5


```

---

## 📖 Documentation & Resources

- **[📓 Jupyter Notebook]()** - Complete analysis with step-by-step explanations
- **[📝 Medium Article](https://medium.com/@mdmoseena22/doordash-delivery-time-prediction-fc633580e9f8)** - Detailed writeup with business insights

---

## 👤 Author

**Md Moseena**

- 🔗 LinkedIn: [linkedin.com/in/mdmoseena](https://www.linkedin.com/in/mdmoseena/)
- 🐙 GitHub: [github.com/MohammedMoseena](https://github.com/MohammedMoseena)
- 📝 Medium: [medium.com/@mdmoseena22](https://medium.com/@mdmoseena22)

---

## 📊 Project Status

✅ **Production-Ready**
- Model meets all business guardrails (<5% extreme errors)
- Statistically validated improvement (p < 0.001)
- Complete deployment artifacts saved (model + encoders + schema)
- Ready for A/B testing and gradual rollout
---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**[📖 Read the Full Article](https://medium.com/@mdmoseena22/doordash-delivery-time-prediction-fc633580e9f8)** | **[📧 Contact Me](https://www.linkedin.com/in/mdmoseena/)**

---

*Last Updated: November 2025*

</div>