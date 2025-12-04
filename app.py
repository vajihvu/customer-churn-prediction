import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

st.set_page_config(layout='wide', page_title='Churn Predicton Model')

st.title("Customer Churn Predicton")

import os
model_paths = {
    'xgb':'models/xgb_pipeline.joblib',
    'rf':'models/rf_pipeline.joblib',
    'lr':'models/logistic_pipeline.joblib'
}
model = None
for k,p in model_paths.items():
    if os.path.exists(p):
        model = joblib.load(p)
        model_name = k
        break
if model is None:
    st.error("No model found. Run `python churn_model.py` first to train and save models.")
    st.stop()

st.sidebar.write(f"Using model: {model_name.upper()} pipeline")

uploaded = st.file_uploader("Upload CSV with same features as training (optional).", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded).head(1000)
    st.write("Preview:")
    st.dataframe(df.head())
    preds = model.predict_proba(df)[:,1]
    df['churn_prob'] = preds
    st.write("Predictions (top 10):")
    st.dataframe(df.sort_values('churn_prob', ascending=False).head(10))
else:
    st.info("Or input a single customer record manually below.")
    text = st.text_area("Paste a single-row CSV or enter key=value pairs separated by commas.", height=120)
    if st.button("Predict single sample"):
        if not text.strip():
            st.warning("Paste a sample row or upload CSV.")
        else:
            try:
                if "," in text and "\n" in text:
                    df = pd.read_csv(pd.compat.StringIO(text))
                elif "," in text and "=" in text:
                    pairs = [p.strip() for p in text.split(',')]
                    data = {}
                    for p in pairs:
                        k,v = p.split('=',1)
                        data[k.strip()] = v.strip()
                    df = pd.DataFrame([data])
                else:
                    st.error("Could not parse input. Upload CSV for best results.")
                    st.stop()
                st.write("Input parsed:")
                st.dataframe(df)
                prob = model.predict_proba(df)[:,1][0]
                st.metric("Churn probability", f"{prob:.3f}")
                
                try:
                    preprocessor = model.named_steps['pre']
                    clf = model.named_steps['clf']
                    X_trans = preprocessor.transform(df)
                    explainer = None
                    if hasattr(clf, "predict_proba") and (model_name in ['rf','xgb']):
                        explainer = shap.TreeExplainer(clf)
                        shap_values = explainer.shap_values(X_trans)
                        st.write("SHAP values (first row):")
                        try:
                            def get_feature_names(column_transformer):
                                feature_names = []
                                for name, trans, cols in column_transformer.transformers_:
                                    if name == 'remainder':
                                        continue
                                    if hasattr(trans, 'named_steps') and 'ohe' in trans.named_steps:
                                        ohe = trans.named_steps['ohe']
                                        cats = ohe.categories_
                                        for col, cat in zip(cols, cats):
                                            for c in cat:
                                                feature_names.append(f"{col}__{c}")
                                    else:
                                        for col in cols:
                                            feature_names.append(col)
                                return feature_names
                            feat_names = get_feature_names(preprocessor)
                        except Exception:
                            feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
                        row_shap = np.array(shap_values[1] if isinstance(shap_values, list) else shap_values)[0]
                        contribs = sorted(zip(feat_names, row_shap), key=lambda x: -abs(x[1]))
                        st.table(pd.DataFrame(contribs[:15], columns=['feature','shap_value']))
                    else:
                        st.info("SHAP explanation not available for this model type in demo.")
                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")
            except Exception as e:
                st.error(f"Parsing/prediction error: {e}")