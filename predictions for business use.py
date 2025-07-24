import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load and preprocess data
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Store customer IDs before preprocessing
customer_ids = df['customerID'].copy()

# Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Prepare features and target
X = df_encoded.drop(['Churn_Yes'], axis=1)
y = df_encoded['Churn_Yes']

# Split the data (stratify to maintain class balance)
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)

# Train the model
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of churn (class 1)

# Reset indices to ensure alignment
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
test_ids_reset = test_ids.reset_index(drop=True)

# Create dataframe with predictions and probabilities
predictions_df = X_test_reset.copy()
predictions_df['customerID'] = test_ids_reset
predictions_df['Actual_Churn'] = y_test_reset
predictions_df['Predicted_Churn'] = y_pred
predictions_df['Churn_Probability'] = y_pred_proba

# Add interpretation columns
predictions_df['Prediction_Correct'] = (predictions_df['Actual_Churn'] == predictions_df['Predicted_Churn'])
predictions_df['Risk_Level'] = pd.cut(
    predictions_df['Churn_Probability'], 
    bins=[0, 0.3, 0.7, 1.0], 
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Reorder columns for better readability
cols_order = ['customerID', 'Actual_Churn', 'Predicted_Churn', 'Churn_Probability', 
              'Risk_Level', 'Prediction_Correct'] + [col for col in predictions_df.columns 
              if col not in ['customerID', 'Actual_Churn', 'Predicted_Churn', 
                           'Churn_Probability', 'Risk_Level', 'Prediction_Correct']]

predictions_df = predictions_df[cols_order]

# Save to CSV
try:
    predictions_df.to_csv('customer_churn_predictions.csv', index=False)
    print("✓ Predictions saved to 'customer_churn_predictions.csv'")
    print(f"✓ Total predictions: {len(predictions_df):,}")
    
    # Display summary statistics
    print(f"\nPrediction Summary:")
    print(f"{'='*40}")
    print(f"Accuracy: {predictions_df['Prediction_Correct'].mean():.2%}")
    print(f"Actual Churn Rate: {predictions_df['Actual_Churn'].mean():.2%}")
    print(f"Predicted Churn Rate: {predictions_df['Predicted_Churn'].mean():.2%}")
    print(f"Average Churn Probability: {predictions_df['Churn_Probability'].mean():.2%}")
    
    print(f"\nRisk Level Distribution:")
    print(f"{'-'*25}")
    risk_dist = predictions_df['Risk_Level'].value_counts()
    for risk, count in risk_dist.items():
        pct = count / len(predictions_df) * 100
        print(f"{risk:<12}: {count:,} ({pct:.1f}%)")
    
    # Show a few sample predictions
    print(f"\nSample Predictions (first 5 rows):")
    print(f"{'-'*50}")
    sample_cols = ['customerID', 'Actual_Churn', 'Predicted_Churn', 'Churn_Probability', 'Risk_Level']
    print(predictions_df[sample_cols].head())
    
except Exception as e:
    print(f"Error saving predictions: {e}")

# Optional: Save only high-risk customers for targeted intervention
high_risk_customers = predictions_df[predictions_df['Risk_Level'] == 'High Risk'].copy()
if len(high_risk_customers) > 0:
    try:
        high_risk_customers.to_csv('high_risk_customers.csv', index=False)
        print(f"\n✓ High-risk customers saved to 'high_risk_customers.csv'")
        print(f"✓ High-risk customers identified: {len(high_risk_customers):,}")
    except Exception as e:
        print(f"Error saving high-risk customers: {e}")

# Performance metrics
print(f"\nModel Performance:")
print(f"{'='*30}")
print("Classification Report:")
print(classification_report(y_test, y_pred))