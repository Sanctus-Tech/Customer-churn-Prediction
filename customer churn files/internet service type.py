from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert Churn column to binary for proper calculation
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Analyze churn by Internet Service type
internet_churn = df.groupby('InternetService')['Churn_Binary'].mean().sort_values(ascending=False)

print("\nChurn Rate by Internet Service:")
print("-" * 40)
for service, rate in internet_churn.items():
    print(f"{service:<15}: {rate:.2%}")

# Create visualization
plt.figure(figsize=(10, 6))
internet_churn.plot(kind='bar', color=['#d62728', '#ff7f0e', '#2ca02c'])
plt.title('Churn Rate by Internet Service Type', fontsize=16, fontweight='bold')
plt.xlabel('Internet Service Type', fontsize=12)
plt.ylabel('Churn Rate', fontsize=12)
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Additional analysis: Customer counts and churn breakdown
print("\nDetailed Analysis by Internet Service:")
print("-" * 50)
service_counts = df['InternetService'].value_counts()
for service in internet_churn.index:
    count = service_counts[service]
    rate = internet_churn[service]
    churned = int(count * rate)
    retained = count - churned
    print(f"{service:<15}: {count:,} customers | Churned: {churned:,} ({rate:.2%}) | Retained: {retained:,}")

# Cross-tabulation for detailed breakdown
print("\nCross-tabulation:")
print("-" * 30)
crosstab = pd.crosstab(df['InternetService'], df['Churn'], margins=True)
print(crosstab)

# Percentage breakdown
print("\nPercentage Breakdown:")
print("-" * 30)
crosstab_pct = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
print(crosstab_pct.round(2))