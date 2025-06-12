# fraud_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure plots render in Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Credit Card Fraud Detection App")

# Load and prepare the original dataset
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('creditcard_2023.csv')  # Path to your original dataset

    x = df.drop(['id', 'Class'], axis=1, errors='ignore')
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    rf_model = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=2, random_state=42)
    rf_model.fit(x_train_scaled, y_train)

    return rf_model, scaler, x.columns

model, scaler, feature_columns = load_and_train_model()

st.write("Upload a CSV file to detect fraud:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(new_data.head())

    # Drop ID column if present
    if 'id' in new_data.columns:
        new_data = new_data.drop(['id'], axis=1)

    # Feature Scaling
    new_data_scaled = scaler.transform(new_data)

    # Make Predictions
    predictions = model.predict(new_data_scaled)

    new_data['Fraud_Prediction'] = predictions

    st.subheader("Prediction Results")
    st.write(new_data)

    # Show only predicted fraud cases
    fraud_cases = new_data[new_data['Fraud_Prediction'] == 1]
    st.subheader("Flagged Fraudulent Transactions")
    st.write(fraud_cases)

    # Optional: Plot feature importance
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp, x='Importance', y='Feature')
    plt.title('Feature Importance')
    st.pyplot()

    # Optional: If the uploaded data contains actual 'Class' labels
    if 'Class' in new_data.columns:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(new_data['Class'], predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot()
else:
    st.info("Please upload a CSV file to get started.")
