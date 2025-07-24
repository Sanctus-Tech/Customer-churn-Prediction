import pandas as pd

# Load your data into a DataFrame (replace 'your_file.csv' with your actual file)
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Convert TotalCharges to numeric (it's stored as object/string)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop customer ID as it's not useful for prediction
df.drop('customerID', axis=1, inplace=True)