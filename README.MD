# Credit Risk Modelling Using ML Techniques 📊

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

Developed a predictive model to categorize applicants into credit approval tiers based on their risk profile, enabling better lending decisions.

---

### 🎯 Key Features & Methodology

- **Feature Selection**: Reduced 80+ initial features down to the 42 most impactful ones using **Chi-Square, ANOVA, and VIF**.
- **Model Training**: Applied various classification models, including **Logistic Regression, SVM, Random Forest, and XGBoost**.
- **Hyperparameter Tuning**: Optimized model performance by fine-tuning hyperparameters with **Cross-Validation (CV)**.
- **MLOps**: Tracked all experiments with **MLflow** for reproducibility and served the final model for real-time inference using a **FastAPI** endpoint.

---

### 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, XGBoost
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI

---

### 📈 Key Results

- **Selected Model**: `XGBoost`
- **Precision**: `80%`
- **Recall**: `73%`
