import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# load data set
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

target_col = [col for col in df_encoded.columns if col.startswith('Churn_')][0]
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Feature selection: select top 20 features
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Scale features for Logistic Regression and SVM
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} - Mean CV Accuracy: {scores.mean():.4f}")
