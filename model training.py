from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load and inspect data
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Handle missing values if any
df = df.dropna()  # or use appropriate imputation

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# More robust target column identification
try:
    target_col = [col for col in df_encoded.columns if col.startswith('Churn_')][0]
except IndexError:
    print("No target column found starting with 'Churn_'")
    print("Available columns:", df_encoded.columns.tolist())
    exit()

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': (RandomForestClassifier(random_state=42), X_train, X_test),
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
    'SVM': (SVC(probability=True, random_state=42), X_train_scaled, X_test_scaled),
    'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), X_train, X_test)
}

# Train and evaluate models
for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)
    scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    print(f"{name} - Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")