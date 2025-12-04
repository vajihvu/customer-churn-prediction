import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def load_telco(path):
    df = pd.read_csv(path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    return df

def preprocess_and_split(df, target='Churn', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return preprocessor, X_train, X_test, y_train, y_test

def train_models(preprocessor, X_train, y_train, X_test, y_test, random_state=42):
    pipe_lr = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    pipe_lr.fit(X_train, y_train)
    preds_lr = pipe_lr.predict_proba(X_test)[:,1]
    auc_lr = roc_auc_score(y_test, preds_lr)

    pipe_rf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))
    ])
    pipe_rf.fit(X_train, y_train)
    preds_rf = pipe_rf.predict_proba(X_test)[:,1]
    auc_rf = roc_auc_score(y_test, preds_rf)

    pipe_xgb = Pipeline([
        ('pre', preprocessor),
        ('clf', xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='auc', random_state=random_state))
    ])
    pipe_xgb.fit(X_train, y_train)
    preds_xgb = pipe_xgb.predict_proba(X_test)[:,1]
    auc_xgb = roc_auc_score(y_test, preds_xgb)

    results = {
        'logistic_auc': auc_lr,
        'rf_auc': auc_rf,
        'xgb_auc': auc_xgb,
        'lr_model': pipe_lr,
        'rf_model': pipe_rf,
        'xgb_model': pipe_xgb
    }
    return results

def evaluate_and_save(results, X_test, y_test, out_dir='models'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(results['lr_model'], f'{out_dir}/logistic_pipeline.joblib')
    joblib.dump(results['rf_model'], f'{out_dir}/rf_pipeline.joblib')
    joblib.dump(results['xgb_model'], f'{out_dir}/xgb_pipeline.joblib')

    print('AUC scores:')
    print(f"  Logistic: {results['logistic_auc']:.4f}")
    print(f"  RandomForest: {results['rf_auc']:.4f}")
    print(f"  XGBoost: {results['xgb_auc']:.4f}")

    best = max([('lr',results['logistic_auc']), ('rf',results['rf_auc']), ('xgb',results['xgb_auc'])], key=lambda x: x[1])[0]
    best_pipe = {'lr':results['lr_model'], 'rf':results['rf_model'], 'xgb':results['xgb_model']}[best]
    preds = (best_pipe.predict_proba(X_test)[:,1] >= 0.5).astype(int)
    print(f"\nBest model: {best}")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)
    return best, best_pipe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--target', default='Churn')
    args = parser.parse_args()

    df = load_telco(args.data)
    preprocessor, X_train, X_test, y_train, y_test = preprocess_and_split(df, target=args.target)
    results = train_models(preprocessor, X_train, y_train, X_test, y_test)
    best_name, best_pipe = evaluate_and_save(results, X_test, y_test)

    print(f"Training complete. Best model: {best_name}. Pipelines saved to ./models")