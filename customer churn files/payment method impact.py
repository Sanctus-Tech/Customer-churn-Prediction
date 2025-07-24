import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a DataFrame
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert Churn column to binary for proper calculation
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Analyze churn by Payment Method
payment_churn = df.groupby('PaymentMethod')['Churn_Binary'].mean().sort_values(ascending=False)

print("\nChurn Rate by Payment Method:")
print("-" * 45)
for method, rate in payment_churn.items():
    print(f"{method:<25}: {rate:.2%}")

# Create visualization
plt.figure(figsize=(12, 6))
payment_churn.plot(kind='bar', color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
plt.title('Churn Rate by Payment Method', fontsize=16, fontweight='bold')
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Churn Rate', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Additional detailed analysis
print("\nDetailed Analysis by Payment Method:")
print("-" * 60)
method_counts = df['PaymentMethod'].value_counts()
for method in payment_churn.index:
    count = method_counts[method]
    rate = payment_churn[method]
    churned = int(count * rate)
    retained = count - churned
    print(f"{method:<25}: {count:,} customers | Churned: {churned:,} ({rate:.2%}) | Retained: {retained:,}")

# Cross-tabulation analysis
print("\nCross-tabulation (Count):")
print("-" * 35)
crosstab = pd.crosstab(df['PaymentMethod'], df['Churn'], margins=True)
print(crosstab)

print("\nCross-tabulation (Percentage):")
print("-" * 35)
crosstab_pct = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
print(crosstab_pct.round(2))

# Summary insights
print("\nKey Insights:")
print("-" * 20)
highest_churn = payment_churn.index[0]
lowest_churn = payment_churn.index[-1]
print(f"• Highest churn rate: {highest_churn} ({payment_churn.iloc[0]:.2%})")
print(f"• Lowest churn rate: {lowest_churn} ({payment_churn.iloc[-1]:.2%})")
print(f"• Difference: {(payment_churn.iloc[0] - payment_churn.iloc[-1]):.2%}")