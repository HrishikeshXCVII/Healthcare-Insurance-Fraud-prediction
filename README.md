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
Using the App:
Launch the Streamlit application.
Upload the sample test file HrishikeshDeka_submission.csv available in this repository.

The application will:
Predict whether each healthcare provider is potentially fraudulent.
Display the fraud probability for each provider.
Download the prediction results as a CSV file using the download option provided in the app.

## Author

Hrishikesh Deka
