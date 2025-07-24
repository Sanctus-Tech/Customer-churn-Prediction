from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Load data
df = pd.read_csv("C:/Users/GABANI/Desktop/data set to work on/Python Project/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

# Prepare features and target
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check original class distribution
print("Original training set distribution:")
print(f"Class distribution: {Counter(y_train)}")
print(f"Class 0 (No Churn): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train):.2%})")
print(f"Class 1 (Churn): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train):.2%})")

# Method 1: SMOTE only (recommended for most cases)
print("\n" + "="*50)
print("Method 1: SMOTE Only")
print("="*50)

smote = SMOTE(sampling_strategy='auto', random_state=42)  # Balance to 50-50
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:")
print(f"Class distribution: {Counter(y_smote)}")
print(f"Total samples: {len(X_smote):,}")

# Method 2: Combined SMOTE + Undersampling 
print("\n" + "="*50)
print("Method 2: SMOTE + Undersampling (Corrected)")
print("="*50)

# Step 1: SMOTE to increase minority class to 70% of majority class
over = SMOTE(sampling_strategy=0.7, random_state=42)

# Step 2: Undersample so majority class is only 1.25x the minority class (80% balance)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)

# Create pipeline: Over -> Under
steps = [('oversample', over), ('undersample', under)]
pipeline = Pipeline(steps=steps)

# Apply to training set only
X_res, y_res = pipeline.fit_resample(X_train, y_train)

print(f"After SMOTE + Undersampling:")
print(f"Class distribution: {Counter(y_res)}")
print(f"Total samples: {len(X_res):,}")

# Method 3: Alternative approach with specific counts
print("\n" + "="*50)
print("Method 3: Specific Target Counts")
print("="*50)

# Calculate current counts
neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)

# Define target counts (example: want 60-40 split with reasonable total size)
target_total = 4000  # Adjust based on your needs
target_pos = int(target_total * 0.4)  # 40% positive class
target_neg = int(target_total * 0.6)  # 60% negative class

# Create sampling strategy dictionaries
over_strategy = {1: target_pos} if target_pos > pos_count else 'auto'
under_strategy = {0: target_neg, 1: target_pos}

over_specific = SMOTE(sampling_strategy=over_strategy, random_state=42)
under_specific = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)

steps_specific = [('over', over_specific), ('under', under_specific)]
pipeline_specific = Pipeline(steps=steps_specific)

X_specific, y_specific = pipeline_specific.fit_resample(X_train, y_train)

print(f"After specific count balancing:")
print(f"Class distribution: {Counter(y_specific)}")
print(f"Total samples: {len(X_specific):,}")

# Compare model performance with different sampling strategies
print("\n" + "="*50)
print("Model Performance Comparison")
print("="*50)

models_to_test = [
    ("Original (Imbalanced)", X_train, y_train),
    ("SMOTE Only", X_smote, y_smote),
    ("SMOTE + Undersampling", X_res, y_res),
    ("Specific Counts", X_specific, y_specific)
]

for name, X_train_sample, y_train_sample in models_to_test:
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_sample, y_train_sample)
    
    # Test on original test set
    test_score = rf.score(X_test, y_test)
    train_score = rf.score(X_train_sample, y_train_sample)
    
    print(f"{name:<25}: Train={train_score:.3f}, Test={test_score:.3f}")

# Recommendation
print("\n" + "="*50)
print("Recommendations:")
print("="*50)
print("1. For most cases, use Method 1 (SMOTE only) - simpler and often effective")
print("2. Use Method 2 if you have computational constraints and need smaller dataset")
print("3. Use Method 3 when you have specific business requirements for class distribution")
print("4. Always evaluate on the original test set to get realistic performance metrics")