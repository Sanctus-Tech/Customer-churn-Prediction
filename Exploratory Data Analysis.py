import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into a DataFrame 
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Churn distribution
plt.figure(figsize=(6,6))
df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Churn Distribution')
plt.ylabel('')
plt.show()

# Numerical features analysis
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols].hist(bins=30, figsize=(15,5))
plt.tight_layout()
plt.show()

# Categorical features analysis
cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

plt.figure(figsize=(20,30))
for i, col in enumerate(cat_cols):
    plt.subplot(6, 3, i+1)
    sns.countplot(data=df, x=col, hue='Churn')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()