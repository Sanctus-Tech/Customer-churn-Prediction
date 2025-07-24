
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

from sklearn.feature_selection import SelectKBest, f_classif


# Find the correct target column name after encoding
target_col = [col for col in df_encoded.columns if col.startswith('Churn_')][0]

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]


# Select top 15 features
selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X, y)

# Get selected features
selected_features = X.columns[selector.get_support()]
print("Selected Features:")
print(selected_features)