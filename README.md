# 📉 Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, with a Streamlit app for single and batch predictions.

**🔗 Live demo:** https://customer-churn-ml1.streamlit.app/

---

## 🎯 Problem

Keeping a customer costs far less than winning a new one. The goal is to flag customers who are likely to churn so the business can act early with offers or support.

Missing a churner is more costly than a false alarm, so the final model is tuned to catch more churners (higher **recall**), accepting some extra false alarms.

## 📊 Dataset

- **Telco Customer Churn** (IBM sample dataset): 7,043 customers, 21 columns
- Target: `Churn` (Yes / No). About 27% of customers churn, so the classes are imbalanced
- Features: demographics, subscribed services, contract type, billing method and charges

## 🔎 Key findings from EDA

| Factor | Finding |
|---|---|
| Contract | Month-to-month customers churn far more than 1- or 2-year customers |
| Tenure | New customers (low tenure) are the most likely to leave |
| Monthly charges | Churned customers pay more per month on average |
| Services | Fiber optic users, and customers without Tech Support or Online Security, churn more |
| Gender, phone service | Little or no effect |

## 🛠️ Methodology

1. **Cleaning:** 11 rows with blank `TotalCharges` were dropped. All have tenure = 0 (brand-new customers), so filling them with an average would not make business sense.
2. **Feature engineering:** `TenureGroup` (0-12, 13-24, 25-48, 49-72 months).
3. **Preprocessing:** one-hot encoding and scaling inside scikit-learn pipelines, so no information leaks from the test set.
4. **Class imbalance:** class weights and SMOTENC.
5. **Models compared:** Logistic Regression, Decision Tree, Random Forest and XGBoost, with GridSearchCV / RandomizedSearchCV (scored on F1).
6. **Threshold tuning:** the decision threshold (**0.32**) was chosen on a validation set using the precision-recall curve. The model was then refit on the full training data and evaluated on the test set.

## 📈 Results (test set, 1,407 customers)

| Model | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.72 | 0.49 | 0.79 | 0.60 |
| Logistic Regression + SMOTENC | 0.74 | 0.51 | 0.72 | 0.59 |
| Decision Tree (tuned) | 0.77 | 0.56 | 0.60 | 0.58 |
| Random Forest (tuned) | 0.77 | 0.55 | 0.73 | 0.63 |
| XGBoost (tuned, threshold 0.50) | 0.80 | 0.66 | 0.53 | 0.59 |
| **XGBoost (tuned, threshold 0.32)** | **0.77** | **0.56** | **0.72** | **0.63** |

- XGBoost ROC-AUC ≈ **0.84**, balanced accuracy **0.758** at threshold 0.32
- Lowering the threshold from 0.50 to 0.32 raises churn recall from 53% to 72%, at the cost of lower precision (66% → 56%).
- Accuracy alone is misleading here: about 73% of customers do not churn, so recall and F1 on the churn class matter more.

## 🚀 Run locally

```bash
git clone https://github.com/your-username/customer-churn.git
cd customer-churn

python -m venv venv
source venv\Scripts\activate

pip install -r requirements.txt
streamlit run app/app.py
```

> The saved model was trained with **scikit-learn 1.8.0**, so that version is pinned in `requirements.txt`. If you see a version warning from XGBoost, install the same XGBoost version you used in the notebook.

## 🖥️ App features

- **Single customer:** enter details, get churn probability, prediction and risk level
- **Batch (CSV):** upload a CSV, score every customer, sort by risk and download the results
- **Adjustable threshold:** sidebar slider to trade precision for recall (default 0.32)
- **About tab:** project summary inside the app

## 📁 Project structure

```text
customer-churn/
├── app/
│   └── app.py                              # Streamlit app
├── notebooks/
│   ├── customer_churn.ipynb                # EDA, modelling, tuning
├── models/
│   ├── customer_churn_xgboost.pkl          # Trained XGBoost pipeline
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── .gitignore
├── requirements.txt
└── README.md
```

## 🧰 Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · matplotlib · Streamlit

## 🎓 What I learned

Building the full workflow: cleaning with business reasoning, leak-free pipelines, handling imbalanced data, model tuning, threshold optimisation for recall, and deploying a model as an app.
