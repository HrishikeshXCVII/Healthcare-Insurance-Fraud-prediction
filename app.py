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
- Almost **62%** of cases are non-fraudulent compared to 8% fraudulent.
- Most patients are between 70 and 80 years of age, with a small proportion below 30.
- Gender 2 has a higher number of beneficiaries than Gender 1.
- Both genders have more non-fraudulent than fraudulent cases, although Gender 2 has a higher number of fraud cases.
- **Race 1** accounts for the majority of beneficiaries.
- Fraudulent and non-fraudulent cases for Races 3 and 5 are nearly equal.
- Most outpatient claims are reimbursed for less than 20,000 USD , while some inpatient claims exceed 120,000 USD.
- Inpatient reimbursements are distributed over a much wider range than outpatient reimbursements.
- DiagnosisGroupCode 882 has the highest number of claims.
- Most claims are filed within 3 days.
- Claims filed after 20 days are generally associated with reimbursements exceeding 15000 USD
- Provider PRV51459 has the highest number of fraudulent cases.
- **Chronic ischemic heart disease** is the most common chronic condition, while **chronic stroke** is the least common.
""")

st.markdown("---")
st.subheader("Business Recommendations")

st.markdown("""
- Providers with a high number of fraudulent claims should be prioritized for detailed audits and investigations.
- Implement additional verification procedures for inpatient claims, as they involve a wider range of reimbursement amounts and higher financial risk.
- Monitor claims filed after 20 days, as they are generally associated with higher reimbursement amounts.
- Continuously monitor providers with frequent claims under DiagnosisGroupCode 882 to identify unusual billing patterns or potential overutilization.
- Analyze claims from providers serving minority patient groups (Races 3 and 5) to identify any unusual fraud trends.
- Prioritize monitoring of claims involving chronic ischemic heart disease, as it represents the largest patient population and accounts for a significant portion of healthcare expenditure.
- Establish real-time monitoring dashboards to track fraud patterns and reimbursement trends
""")

    
