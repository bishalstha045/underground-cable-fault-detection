import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UNDERGROUND CABLE FAULT DETECTION - MODEL TRAINING & EVALUATION
# ============================================================================

print("=" * 70)
print("LOADING AND PREPROCESSING DATA")
print("=" * 70)

# Load dataset
df = pd.read_csv("underground_cable_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

print(f"\n✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  • Features: {', '.join(X.columns.tolist())}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✓ Data split: {len(X_train)} training samples, {len(X_test)} test samples")

# Feature scaling (Critical for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# ============================================================================
print("\n" + "=" * 70)
print("MODEL TRAINING & HYPERPARAMETER TUNING")
print("=" * 70)

# 1. SVM
print("\n1. SVM (Support Vector Machine) - Training...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
print(f"   ✓ SVM Model trained")

# 2. Random Forest
print("\n2. Random Forest Classifier - Training...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)
print("   ✓ Model trained")

# 3. Logistic Regression
print("\n3. Logistic Regression - Training...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("   ✓ Model trained")

# ============================================================================
print("\n" + "=" * 70)
print("CROSS-VALIDATION EVALUATION")
print("=" * 70)

models = {
    'SVM': svm_model,
    'Random Forest': rf_model,
    'Logistic Regression': lr_model
}

cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    print(f"\n{name}:")
    print(f"  • CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================================================
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

results = {}

for name, model in models.items():
    print(f"\n{'=' * 70}")
    print(f"{name.upper()}")
    print('=' * 70)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['No Fault', 'Fault']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")

# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERSISTENCE")
print("=" * 70)

# Save best model (SVM)
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✓ Best SVM model saved as 'svm_model.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")

# Performance Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
print(f"\nBest Model: {best_model_name}")
print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print("\nAll models trained and evaluated successfully!")
print("=" * 70)