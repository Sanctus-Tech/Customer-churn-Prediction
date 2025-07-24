
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert target variable to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Check correlation with target
plt.figure(figsize=(12,8))
df_encoded.corr()['Churn'].sort_values()[:-1].plot(kind='barh')
plt.title('Feature Correlation with Churn')
plt.show()