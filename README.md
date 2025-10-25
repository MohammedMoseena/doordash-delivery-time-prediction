# 🚀 DoorDash Delivery Time Prediction


> **Production-ready ML model achieving 10.92-minute MAE with 57.95% accuracy in predicting delivery times**

An end-to-end machine learning solution that predicts DoorDash delivery duration from order creation to customer delivery, optimized for real-time customer-facing applications.

---

## 📊 Key Results

| Metric | Value | Status |
|--------|-------|--------|
| **Mean Absolute Error** | 10.92 minutes | ✅ |
| **Accuracy (±10 min)** | 57.95% | ✅ |
| **Extreme Error Rate** | 4.69% | ✅ (<5% guardrail) |
| **Model Improvement** | 9.1% vs Linear Regression | ✅ |

---

## 🎯 Problem Statement

DoorDash needs accurate real-time predictions of **total delivery duration** (seconds between order creation and delivery) to:
- Provide reliable ETAs to customers
- Optimize dasher allocation
- Enable proactive operations management
- Reduce customer complaints and churn

**Challenge:** Complex marketplace dynamics with non-linear relationships between 19 features including order details, restaurant characteristics, and real-time marketplace load.

---

## 🛠️ Technical Approach

### Architecture Overview

```
Data Pipeline → Feature Engineering → Model Training → Evaluation → Deployment Ready
     ↓              ↓                      ↓              ↓            ↓
 197K records   19 features          XGBoost        Business     API-ready
 Cleaned data   Log transform        Tuned          Metrics      <200ms latency
```

### Key Features

**📈 Advanced Feature Engineering**
- Cyclical encoding for temporal features (hour, day of week)
- Ratio features to reduce multicollinearity (`busy_dasher_ratio`, `orders_per_dasher`)
- Market-store combined identifiers
- Log transformation + RobustScaler for skewed distributions

**🧠 Model Selection**
- Evaluated 5 algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- XGBoost selected for best bias-variance tradeoff
- Hyperparameter tuning via RandomizedSearchCV with TimeSeriesSplit

**✅ Rigorous Validation**
- Temporal train-test split (80/20) to prevent data leakage
- Business-aligned metrics (MAE in minutes, accuracy rate, extreme error rate)
- Inverse transformation validation on original scale

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
│   └── (model artifacts)                     # xgboost model pickel file and metadata file
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
git clone https://github.com/yourusername/doordash-delivery-time-prediction.git
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

---

## 📊 Data Overview

### Dataset Statistics
- **Records:** 197,428 (99.86% retained after cleaning)
- **Features:** 19 engineered features
- **Target:** `total_delivery_duration_seconds`
- **Time Period:** 2015 data (US/Pacific timezone)

### Feature Categories

| Category | Features | Examples |
|----------|----------|----------|
| **Marketplace** | 2 | `orders_per_dasher`, `busy_dasher_ratio` |
| **Order Details** | 5 | `total_items`, `subtotal`, `num_distinct_items` |
| **Temporal** | 6 | `hour_sin/cos`, `dayofweek_sin/cos`, `month` |
| **Location** | 3 | `market_id`, `market_store_id`, `store_primary_category` |
| **Predictions** | 2 | `estimated_order_place_duration`, `estimated_store_to_consumer_driving_duration` |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Identified and removed 0.14% outliers (ultra-fast <5min, delayed >3hrs deliveries)
- Handled missing values via group-based imputation + Iterative Imputer
- Discovered high multicollinearity in marketplace features

### 2. Data Cleaning
```python
# Key cleaning steps:
✓ Removed 94 records with negative values (<0.05%)
✓ Removed 143 extreme outliers (0.07%)
✓ Imputed ~8% missing marketplace data using ML-based approach
✓ Final dataset: 197,151 records (99.86% retention)
```

### 3. Feature Engineering Highlights
- **Temporal:** Sine-cosine encoding for cyclical patterns
- **Marketplace Load:** `busy_dasher_ratio = total_busy_dashers / (total_onshift_dashers + 1)`
- **Scaling:** RobustScaler for outlier resilience
- **Target:** Log transform + separate scaler for inverse transformation

### 4. Model Training
```python
Models evaluated:
├── Linear Regression (baseline)
├── Decision Tree (severe overfitting)
├── Random Forest (moderate performance)
├── Gradient Boosting (close runner-up)
└── XGBoost ⭐ (winner)
```

### 5. Hyperparameter Tuning
- **Strategy:** RandomizedSearchCV (20 iterations)
- **Cross-Validation:** TimeSeriesSplit (3 folds)
- **Result:** 0.7% MAE improvement (marginal but validates strong baseline)

---

## 📈 Results & Insights

### Model Performance Comparison

| Model | MAE (min) | Accuracy (±10 min) | Extreme Errors | Status |
|-------|-----------|-------------------|----------------|--------|
| Linear Regression | 12.01 | 53.71% | 6.31% | ❌ |
| Decision Tree | 16.90 | 41.34% | 15.63% | ❌ |
| Random Forest | 11.44 | 56.65% | 5.84% | ⚠️ |
| Gradient Boosting | 11.32 | 57.24% | 5.75% | ⚠️ |
| **XGBoost (Tuned)** | **10.92** | **57.95%** | **4.69%** | ✅ |

### Top 5 Predictive Features

1. 🥇 **orders_per_dasher** - Marketplace load (strongest signal)
2. 🥈 **estimated_order_place_duration** - Restaurant receiving order time
3. 🥉 **estimated_store_to_consumer_driving_duration** - Travel distance
4. **hour_sin/hour_cos** - Peak hour effects
5. **subtotal_log** - Order complexity proxy

---

## 💼 Business Impact

### Actionable Recommendations

**1. 📍 Optimize Dasher Deployment**
- Deploy more dashers during peak hours when `orders_per_dasher` ratio is high
- Target markets with consistently high marketplace load

**2. ⏱️ Dynamic ETA Buffers**
- Add safety margins during high-load periods to reduce customer complaints
- Personalize buffers based on order complexity (`subtotal`, `total_items`)

**3. 🚨 Flag Extreme Predictions**
- Auto-flag deliveries predicted >45 min for operations review
- Proactive customer communication for delayed orders

**4. 🤝 Restaurant Partnerships**
- Share `estimated_order_place_duration` insights with restaurants
- Collaborate on reducing order receiving time variability

### Expected Business Outcomes
- ✅ Reduced customer complaints (fewer missed ETAs)
- ✅ Improved operational efficiency (better dasher allocation)
- ✅ Higher customer retention (increased trust)
- ✅ Competitive differentiation (accuracy in ETA predictions)

---

## ⚠️ Limitations & Future Work

### Current Limitations
- 📅 Static 2015 dataset (marketplace has evolved)
- 🚦 No real-time traffic/weather data
- 🏪 Missing restaurant capacity metrics
- 📍 Accuracy ceiling at ~58% with available features

### Planned Improvements
```python
Future enhancements:
├── Integrate real-time traffic APIs (Google Maps, Waze)
├── Add restaurant historical prep time patterns
├── Incorporate weather conditions
├── Weekly model retraining pipeline
├── A/B testing framework (5% traffic rollout)
└── MLOps monitoring (drift detection, performance tracking)
```

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **ML/DL** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Encoding** | Label Encoding, Cyclical Encoding |
| **Scaling** | RobustScaler |
| **Imputation** | IterativeImputer |
| **Validation** | TimeSeriesSplit, RandomizedSearchCV |

---

## 📚 Dependencies

```python
numpy==2.0.2
pandas==2.2.2
matplotlib==3.10.0
seaborn==0.13.2
scikit-learn==1.6.1
xgboost==3.1.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📖 Documentation
- **[Jupyter Notebook](notebooks/doordash_delivery_prediction.ipynb)** - Interactive analysis and modeling

---

## 👤 Author

**Md Moseena**

- LinkedIn: [linkedin.com/in/mdmoseena](https://www.linkedin.com/in/mdmoseena/)
- GitHub: [https://github.com/MohammedMoseena](https://github.com/MohammedMoseena)
- Medium: [https://medium.com/@mdmoseena22](https://medium.com/@mdmoseena22)

---
## 📊 Project Status

✅ **Production-Ready** - Model meets all business guardrails and is ready for deployment with monitoring

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**[📝 Read the Full Article](https://medium.com/@mdmoseena22/doordash-delivery-time-prediction-fc633580e9f8)** | **[📧 Contact Me](https://www.linkedin.com/in/mdmoseena/)**

</div>


