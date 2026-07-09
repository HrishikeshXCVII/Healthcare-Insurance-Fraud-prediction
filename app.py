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
    
st.markdown("---")
st.subheader("Observations")

st.markdown("""
- Almost **62%** of cases are non-fraudulent compared to **38%** fraudulent.
- Most patients are between **70 and 80 years** of age, with a small proportion below 30.
- **Gender 2** has a higher number of beneficiaries than Gender 1.
- Both genders have more non-fraudulent than fraudulent cases, although Gender 2 has a higher number of fraud cases.
- **Race 1** accounts for the majority of beneficiaries.
- Fraudulent and non-fraudulent cases for **Races 3 and 5** are nearly equal.
- Most outpatient claims are reimbursed for **less than $20,000**, while some inpatient claims exceed **$120,000**.
- Inpatient reimbursements are distributed over a much wider range than outpatient reimbursements.
- **DiagnosisGroupCode 882** has the highest number of claims.
- Most claims are filed within **3 days**.
- Claims filed after **20 days** are generally associated with reimbursements exceeding **$15,000**.
- **Provider PRV51459** has the highest number of fraudulent cases.
- **Chronic ischemic heart disease** is the most common chronic condition, while **chronic stroke** is the least common.
""")

    
