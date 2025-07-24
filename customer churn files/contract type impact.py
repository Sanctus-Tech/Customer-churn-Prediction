from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check the unique values in the Churn column
print("Unique values in Churn column:", df['Churn'].unique())
print("Data types:")
print(df.dtypes)

# Convert Churn column to binary (1 for 'Yes', 0 for 'No')
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Analyze churn by contract type
contract_churn = df.groupby('Contract')['Churn_Binary'].mean().sort_values(ascending=False)
print("\nChurn Rate by Contract Type:")
print("-" * 40)
for contract, rate in contract_churn.items():
    print(f"{contract:<20}: {rate:.2%}")

# Create a visualization
plt.figure(figsize=(10, 6))
contract_churn.plot(kind='bar', color=['#ff7f0e', '#2ca02c', '#1f77b4'])
plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold')
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Churn Rate', fontsize=12)
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Additional analysis: Count of customers by contract type
print("\nCustomer Count by Contract Type:")
print("-" * 40)
contract_counts = df['Contract'].value_counts()
for contract, count in contract_counts.items():
    churn_rate = contract_churn[contract]
    print(f"{contract:<20}: {count:,} customers (Churn Rate: {churn_rate:.2%})")

# Cross-tabulation for more detailed view
print("\nDetailed Churn Analysis:")
print("-" * 40)
crosstab = pd.crosstab(df['Contract'], df['Churn'], margins=True)
print(crosstab)
