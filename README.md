# 🏡 Real Estate Valuation & Predictive Pricing Model

An end-to-end data science project predicting urban real estate prices using regularized linear regression techniques (LassoCV & RidgeCV). This project covers baseline benchmark comparison, hyperparameter tuning via Cross-Validation, feature selection, and residual diagnostic analysis.

## 📊 Key Results & Findings

* **Model Accuracy:** Reduced baseline Root Mean Squared Error (RMSE) from **$59,227.73** down to **$35,564.49** (~40% improvement in predictive precision).
* **Final Model:** Selected **LassoCV** for production due to its built-in $L_1$ feature selection, which successfully eliminated uninformative features while matching Ridge performance.
* **Top Value Drivers:** Property size (`surface_covered_in_m2`) is the dominant factor driving price (~$40k impact per std dev), alongside key location premiums (e.g., *Puerto Madero*, *Palermo*).
* **Diagnostic Insights:** Identified **heteroscedasticity** (funnel pattern in residuals), revealing that linear models perform exceptionally well for standard properties ($\le \$150\text{k}$) but exhibit higher variance on luxury homes.

## 🛠️ Tech Stack & Methods
* **Python Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Modeling:** RidgeCV, LassoCV, One-Hot Encoding, StandardScaler
* **Diagnostics:** Residual plots, Coefficient Feature Importance Analysis

## 📁 Repository Structure
* `notebook.ipynb` - Complete Jupyter Notebook containing EDA, data preprocessing, model training, and evaluation.
