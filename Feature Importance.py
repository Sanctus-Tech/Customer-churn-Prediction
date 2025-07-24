import matplotlib.pyplot as plt
import seaborn as sns  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data 
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Prepare features and target
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

# Get feature importances
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': importances
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)


print("Top 10 Most Important Features:")
print("-" * 40)
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} : {row['Importance']:.4f}")