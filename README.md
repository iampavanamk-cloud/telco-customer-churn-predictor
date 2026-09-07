# 🛒 E-Commerce / Telco Customer Churn Predictor

## 📌 Business Overview
Customer churn is one of the most critical metrics for subscription and retail businesses. Retaining existing customers is significantly cheaper than acquiring new ones. This project analyzes customer behavior data to identify high-risk churners and pinpoints the primary factors driving customer attrition.

## 📊 Dataset & Features
* **Source:** Telco Customer Churn Dataset (Kaggle)
* **Size:** 7,043 customers across 21 features.
* **Target Variable:** `Churn` (Binary: Yes/No)

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy (Data Cleaning & Manipulation)
* **Visualization:** Matplotlib, Seaborn (EDA & Feature Importance)
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`)

## 🔍 Key Insights & EDA Findings
* **Contract Impact:** Customers on Month-to-Month contracts exhibit a disproportionately higher churn rate compared to those on 1-year or 2-year commitments.
* **Financial Drivers:** Feature importance modeling revealed that `TotalCharges`, `MonthlyCharges`, and overall `tenure` are the top predictors of whether a customer will leave.

## 🤖 Machine Learning Model Performance
* **Model Used:** Random Forest Classifier (80/20 Train-Test Split)
* **Accuracy:** ~78.50%
* **Evaluation:** Addressed categorical encoding via One-Hot Encoding and handled missing values in `TotalCharges`.

## 🚀 How to Run the Project
1. Clone this repository.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run the script: `python churn_analysis.py`
