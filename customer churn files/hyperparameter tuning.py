from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import time

# Load and preprocess data 
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Quick preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Class distribution: {np.bincount(y_train)}")



# Option 2: Faster, more focused grid search
print("\n" + "="*60)
print("Option 2: Focused Grid Search (Recommended)")
print("="*60)

param_grid_focused = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']  # Focus on balanced since we have imbalanced data
}

print(f"Total combinations: {2*3*2*1} = 12")
print("With 5-fold CV, this means 60 model fits")

start_time = time.time()
grid_search_focused = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_focused,
    cv=5,
    scoring='recall',  # Optimize for recall (finding churners)
    n_jobs=-1,
    verbose=1
)

grid_search_focused.fit(X_train, y_train)
elapsed_time = time.time() - start_time

print(f"\n✓ Grid search completed in {elapsed_time:.2f} seconds")
print(f"Best parameters: {grid_search_focused.best_params_}")
print(f"Best cross-validation recall: {grid_search_focused.best_score_:.4f}")



# Smaller grid for multi-metric evaluation
param_grid_multi = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Test multiple scoring metrics
scoring_metrics = ['recall', 'precision', 'f1', 'roc_auc']
results = {}

for metric in scoring_metrics:
    print(f"\nOptimizing for {metric}...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_multi,
        cv=5,
        scoring=metric,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    results[metric] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }

# Compare results
print(f"\n{'Metric':<12} {'Best Score':<12} {'n_estimators':<12} {'max_depth':<10} {'min_samples_split':<15}")
print("-" * 70)
for metric, result in results.items():
    params = result['best_params']
    print(f"{metric:<12} {result['best_score']:<12.4f} {params['n_estimators']:<12} "
          f"{str(params['max_depth']):<10} {params['min_samples_split']:<15}")

# Evaluate the best models on test set
print(f"\n{'='*60}")
print("Test Set Performance Comparison")
print("="*60)

for metric, result in results.items():
    model = result['best_model']
    y_pred = model.predict(X_test)
    
    # Calculate multiple metrics
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"\nModel optimized for {metric}:")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  ROC-AUC: {test_auc:.4f}")

# Recommendation: Choose the best model based on business needs
print(f"\n{'='*60}")
print("Recommendations:")
print("="*60)
print("• For churn prediction, RECALL is often most important (don't miss churners)")
print("• F1-score provides a good balance between precision and recall")
print("• ROC-AUC is good for ranking customers by churn probability")
print("• Consider business cost: False negatives (missed churners) vs False positives (unnecessary interventions)")

# Save the best model (example: recall-optimized)
best_model = results['recall']['best_model']
print(f"\nSaving recall-optimized model...")

import joblib
joblib.dump(best_model, 'best_churn_model_gridsearch.pkl')
print("✓ Model saved as 'best_churn_model_gridsearch.pkl'")

# Feature importance from best model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print("-" * 40)
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<25} : {row['importance']:.4f}")