import streamlit as st, pandas as pd, joblib

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.title('Healthcare Provider Fraud Detection')

f = st.file_uploader('Upload CSV', type='csv')

if f is not None:
    data = pd.read_csv(f)

    X = data[features]
    X_scaled = scaler.transform(X)

    data['FraudProbability'] = model.predict_proba(X_scaled)[:, 1]
    data['PredictedFraud'] = model.predict(X_scaled)

    st.dataframe(data)

    st.download_button('Download predictions',data.to_csv(index=False),'predictions.csv')

    