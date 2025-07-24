import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Store customer IDs
customer_ids = df['customerID'].copy()

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop(['Churn_Yes'], axis=1)
y = df_encoded['Churn_Yes']

# Train-test split
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Reset indices
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
test_ids_reset = test_ids.reset_index(drop=True)

# Create predictions_df
predictions_df = X_test_reset.copy()
predictions_df['customerID'] = test_ids_reset
predictions_df['Actual_Churn'] = y_test_reset
predictions_df['Predicted_Churn'] = y_pred
predictions_df['Churn_Probability'] = y_pred_proba

# Extract high-risk customers (probability > 70% and predicted to churn)
high_risk_customers = predictions_df[
    (predictions_df['Predicted_Churn'] == 1) & 
    (predictions_df['Churn_Probability'] > 0.7)
].copy()

# Sort by churn probability descending
high_risk_customers = high_risk_customers.sort_values('Churn_Probability', ascending=False)

# Save to CSV
high_risk_customers.to_csv('high_risk_customers.csv', index=False)
print(f"✓ High-risk customers saved: {len(high_risk_customers):,}")

print("\nTop 5 High-Risk Customers:")
print(high_risk_customers[['customerID', 'Churn_Probability']].head())
