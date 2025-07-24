import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display basic information
print(f"Dataset shape: {df.shape}")
print(df.info())