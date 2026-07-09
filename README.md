# Healthcare-Insurance-Fraud-prediction

# Provider Fraud Predictor

## Overview

This project is a machine learning web application that predicts the likelihood of healthcare provider fraud using a Logistic Regression model. The application is built with Streamlit and allows users to upload a CSV file to generate fraud predictions and fraud probabilities.

## Features

* Upload provider data in CSV format.
* Predict fraudulent providers using a trained Logistic Regression model.
* Display fraud probability for each record.
* Download prediction results as a CSV file.

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* 
**TEST FILE:** `HrishikeshDeka_submission.csv` (included in this repository) can be used to test the Streamlit application.
  
## Files

* `app.py` – Streamlit application
* `fraud_model.pkl` – Trained Logistic Regression model
* `scaler.pkl` – StandardScaler used during training
* `features.pkl` – Feature names used by the model
* `requirements.txt` – Project dependencies

## Running the App

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```


## Author

Hrishikesh Deka
