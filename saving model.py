import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Load and preprocess data
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Prepare features and target
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Feature selection (optional - select top k features)
selector = SelectKBest(score_func=f_classif, k=15)  # Select top 15 features
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

print("Selected features:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the best model
best_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
best_model.fit(X_train, y_train)

# Evaluate the model
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print(f"\nModel Performance:")
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Save the model
print("\nSaving model and features...")
try:
    joblib.dump(best_model, 'churn_prediction_model.pkl')
    print("✓ Model saved as 'churn_prediction_model.pkl'")
    
    # Save the feature list
    joblib.dump(list(selected_features), 'selected_features.pkl')
    print("✓ Selected features saved as 'selected_features.pkl'")
    
    # Save the feature selector for future use
    joblib.dump(selector, 'feature_selector.pkl')
    print("✓ Feature selector saved as 'feature_selector.pkl'")
    
    # Save preprocessing info (for future predictions)
    preprocessing_info = {
        'feature_columns': list(X.columns),
        'categorical_columns': [col for col in df.columns if df[col].dtype == 'object'],
        'numerical_columns': [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    }
    joblib.dump(preprocessing_info, 'preprocessing_info.pkl')
    print("✓ Preprocessing info saved as 'preprocessing_info.pkl'")
    
    print("\nAll files saved successfully!")
    
except Exception as e:
    print(f"Error saving files: {e}")

# Example of how to load the model later
print("\n" + "="*50)
print("Example: How to load and use the saved model:")
print("="*50)
print("""
# To load the model later:
import joblib

# Load the saved components
loaded_model = joblib.load('churn_prediction_model.pkl')
loaded_features = joblib.load('selected_features.pkl')
feature_selector = joblib.load('feature_selector.pkl')
preprocessing_info = joblib.load('preprocessing_info.pkl')

# Use the model for predictions
# new_predictions = loaded_model.predict(new_data_selected_features)
""")