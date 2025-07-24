from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into the DataFrame
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Encode categorical variables
df_encoded = pd.get_dummies(df)


# Define the list of features to use (exclude all Churn columns)
selected_features = [col for col in df_encoded.columns if not col.startswith('Churn_')]

# Find the correct target column name after encoding
target_col = [col for col in df_encoded.columns if col.startswith('Churn_')][0]

X = df_encoded[selected_features]
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
