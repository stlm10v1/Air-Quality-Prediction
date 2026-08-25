# ☁️ 🌍 Air Quality Prediction using Machine Learning

This project performs **multi-class classification** on a dataset containing sensor responses and weather conditions to predict the Air Quality Class (Good, Moderate, Poor). The primary goal is to build and compare classifiers using various machine learning models:

* **Ensemble & Advanced ML Models** (XGBoost, Random Forest)
* **Traditional ML models** (KNN, Logistic Regression, SVM, Decision Trees)

---

## 📍 Project Highlights

* Conducted extensive **EDA**, addressed missing data via **time-based interpolation**, and mitigated anomalies by capping outliers using the **1.5*IQR rule**.
* Engineered predictive **lag features** (1-hour), **rolling averages** (24-hour), and temporal variables (hour, day, month) to effectively capture seasonal and diurnal environmental trends.
* Formulated a custom **tie-breaking algorithm** for multi-pollutant classification, prioritizing conservative "Poor" air quality labels to resolve class ambiguities across 19.7% of the dataset.
* Evaluated baseline versus pruned **Decision Trees** using **cost-complexity pruning (ccp_alpha)**, successfully reducing tree nodes from 1,221 to 319.
* Built and tuned ensemble models utilizing **GridSearchCV** and validated model stability and generalization via **5-Fold Cross-Validation**.
* Achieved a peak classification accuracy of **92%** and a macro F1-score of **0.922** using an optimized **XGBoost** model.

---

## 🔧 Tech Stack

* **Python**
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn, Plotly

---

## 📁 Files Included

* `Group4_Air_Quality_Project.ipynb` – Complete ML pipeline including EDA, feature engineering, and model training
* `AirQualityUCI.csv` – Raw sensor and weather dataset
* `model_features_X.csv` – Processed and scaled feature dataset
* `model_targets_y.csv` – Unscaled Ground Truth target values
* `requirements.txt` – Python dependencies

---

## 🧠 Learning Outcomes

* Deep understanding of handling time-series data, including the creation of lag and rolling features.
* Practical experience with advanced data preprocessing, imputation, and outlier treatment.
* Comparison between various classification algorithms (Logistic Regression, Decision Trees, Random Forest, KNN, SVM, XGBoost).
* Hands-on experience with hyperparameter tuning, cost-complexity pruning, and cross-validation strategies.

---

## 👨‍💻 Author

**Vikram Anand**
M.Tech (2025), Department of Management Sciences (DoMS)
IIT Kanpur
