# customer-churn-prediction

This repository contains an end-to-end machine learning pipeline for predicting customer churn in Telecom domain. The project includes data preprocessing, exploratory analysis, model training, evaluation, explainability using SHAP and a Streamlit web application for real-time churn scoring.

## Features
- Complete churn prediction workflow using Python and scikit-learn
- Models implemented: Logistic Regression, Random Forest and XGBoost
- Automated preprocessing using pipelines (handling missing values, encoding and scaling)
- Exploratory data analysis to understand churn behavior
- Model comparison using ROC-AUC, accuracy, precision, recall, F1-score and confusion matrix
- SHAP-based explainability to interpret feature impact on individual predictions
- Streamlit dashboard for interactive churn probability prediction
- Jupyter Notebook for experimentation and reproducibility
- Saved model artifacts for deployment

## Dataset
Telco Customer Churn (Kaggle)

## SHAP Explainability
SHAP is used to:
- Identify top contributing features
- Explain individual customer churn risk
- Improve model transparency for business stakeholders

Random Forest and XGBoost support direct SHAP explanations.

## Future Enhancements
- Add LightGBM and CatBoost models
- Implement hyperparameter tuning with GridSearchCV or Optuna
- Build API endpoint using FastAPI for model serving
- Create automated CI/CD pipeline for deployment
- Add customer segmentation and retention strategy recommendations
