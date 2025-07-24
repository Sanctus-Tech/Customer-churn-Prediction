import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original dataset shape:", df.shape)
print("\nData types before conversion:")
print(df[['TotalCharges', 'tenure', 'MonthlyCharges']].dtypes)

# Ensure numeric conversion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')

print(f"\nMissing values after conversion:")
print(f"TotalCharges: {df['TotalCharges'].isna().sum()}")
print(f"tenure: {df['tenure'].isna().sum()}")
print(f"MonthlyCharges: {df['MonthlyCharges'].isna().sum()}")

# Fill missing values with safe defaults
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Handle tenure more carefully
print(f"\nTenure statistics:")
print(f"Zero tenure customers: {(df['tenure'] == 0).sum()}")
print(f"Tenure range: {df['tenure'].min()} to {df['tenure'].max()}")

# Replace 0 tenure with 1 (more logical than NaN for new customers)
df['tenure'] = df['tenure'].replace(0, 1)
df['tenure'].fillna(df['tenure'].median(), inplace=True)

# 1. Avg Monthly Charge (Ratio Feature) - with safe division
df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure']
df['AvgMonthlyCharge'] = df['AvgMonthlyCharge'].replace([np.inf, -np.inf], 0)

# 2. Interaction Feature: tenure × monthly charges
df['Tenure_MonthlyCharge'] = df['tenure'] * df['MonthlyCharges']

# 3. Bin tenure into groups (improved binning)
tenure_max = df['tenure'].max()
print(f"Maximum tenure: {tenure_max}")

# Create bins that cover the full range
if tenure_max <= 72:
    bins = [0, 12, 24, 36, 60, 72]
    labels = ['0-1yr', '1-2yr', '2-3yr', '3-5yr', '5-6yr']
else:
    bins = [0, 12, 24, 36, 60, 72, tenure_max + 1]
    labels = ['0-1yr', '1-2yr', '2-3yr', '3-5yr', '5-6yr', '6yr+']

df['TenureGroup'] = pd.cut(
    df['tenure'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# 4. Create a service count column (with error handling)
service_cols = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies'
]

# Check which service columns actually exist
existing_service_cols = [col for col in service_cols if col in df.columns]
missing_service_cols = [col for col in service_cols if col not in df.columns]

if missing_service_cols:
    print(f"\nWarning: Missing service columns: {missing_service_cols}")

print(f"Using service columns: {existing_service_cols}")

# Convert 'No internet service' and 'No phone service' to 'No'
for col in existing_service_cols:
    if col in df.columns:
        df[col] = df[col].replace(['No internet service', 'No phone service'], 'No')

# Count services more safely
df['NumServices'] = 0
for col in existing_service_cols:
    if col in df.columns:
        df['NumServices'] += (df[col] == 'Yes').astype(int)

# 5. Charges per service (with better handling)
df['ChargesPerService'] = np.where(
    df['NumServices'] > 0, 
    df['MonthlyCharges'] / df['NumServices'], 
    0
)

# 6. Additional useful features
df['TotalChargesPerTenure'] = df['TotalCharges'] / df['tenure']
df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)
df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)

# Create monthly charges categories
df['MonthlyChargesBin'] = pd.cut(
    df['MonthlyCharges'], 
    bins=[0, 30, 60, 90, df['MonthlyCharges'].max()],
    labels=['Low', 'Medium', 'High', 'Premium']
)

# 7. Payment and contract interaction
if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
    df['Contract_Payment'] = df['Contract'].astype(str) + '_' + df['PaymentMethod'].astype(str)

print(f"\nFeature Engineering Summary:")
print(f"{'='*40}")
print(f"Original features: {df.shape[1] - 8}")  # Subtract new features
print(f"New features created: 8+")
print(f"Total features: {df.shape[1]}")

# Display feature statistics
new_features = [
    'AvgMonthlyCharge', 'Tenure_MonthlyCharge', 'NumServices', 
    'ChargesPerService', 'TotalChargesPerTenure', 'IsNewCustomer', 'IsLongTermCustomer'
]

print(f"\nNew Feature Statistics:")
print(f"{'-'*50}")
for feature in new_features:
    if feature in df.columns:
        print(f"{feature:<20}: Mean={df[feature].mean():.2f}, Std={df[feature].std():.2f}")

# Check for any infinite or NaN values in new features
print(f"\nData Quality Check:")
print(f"{'-'*30}")
for feature in new_features:
    if feature in df.columns:
        inf_count = np.isinf(df[feature]).sum()
        nan_count = df[feature].isna().sum()
        if inf_count > 0 or nan_count > 0:
            print(f"{feature}: {inf_count} infinite, {nan_count} NaN values")

# Visualize some of the new features


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Average Monthly Charge distribution
axes[0,0].hist(df['AvgMonthlyCharge'], bins=30, alpha=0.7, color='skyblue')
axes[0,0].set_title('Distribution of Average Monthly Charge')
axes[0,0].set_xlabel('Average Monthly Charge')

# Plot 2: Number of Services
if 'Churn' in df.columns:
    df_plot = df.copy()
    df_plot['Churn_Binary'] = (df_plot['Churn'] == 'Yes').astype(int)
    service_churn = df_plot.groupby('NumServices')['Churn_Binary'].mean()
    axes[0,1].bar(service_churn.index, service_churn.values, alpha=0.7, color='lightcoral')
    axes[0,1].set_title('Churn Rate by Number of Services')
    axes[0,1].set_xlabel('Number of Services')
    axes[0,1].set_ylabel('Churn Rate')

# Plot 3: Tenure Groups
if 'TenureGroup' in df.columns:
    tenure_counts = df['TenureGroup'].value_counts()
    axes[1,0].bar(range(len(tenure_counts)), tenure_counts.values, alpha=0.7, color='lightgreen')
    axes[1,0].set_title('Customer Distribution by Tenure Group')
    axes[1,0].set_xlabel('Tenure Group')
    axes[1,0].set_ylabel('Number of Customers')
    axes[1,0].set_xticks(range(len(tenure_counts)))
    axes[1,0].set_xticklabels(tenure_counts.index, rotation=45)

# Plot 4: Charges per Service
axes[1,1].hist(df['ChargesPerService'], bins=30, alpha=0.7, color='gold')
axes[1,1].set_title('Distribution of Charges per Service')
axes[1,1].set_xlabel('Charges per Service')

plt.tight_layout(pad=2.0)
plt.show()

print(f"\n✓ Feature engineering completed successfully!")
print(f"Dataset ready for modeling with {df.shape[1]} features")