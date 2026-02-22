import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
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

# 1. SVM with GridSearchCV for hyperparameter tuning
print("\n1. SVM (Support Vector Machine) - Tuning hyperparameters...")
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly']
}

grid_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, n_jobs=-1)
grid_svm.fit(X_train_scaled, y_train)
svm_model = grid_svm.best_estimator_
print(f"   ✓ Best SVM params: {grid_svm.best_params_}")

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
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
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
print("VISUALIZATIONS")
print("=" * 70)

# 1. Model Comparison - Performance Metrics
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
x = np.arange(len(metrics))
width = 0.25

for i, (name, scores) in enumerate(results.items()):
    values = [scores[metric] for metric in metrics]
    ax.bar(x + i*width, values, width, label=name)

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Model comparison saved as 'model_comparison.png'")

# 2. Confusion Matrices for all models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices - Test Set', fontsize=13, fontweight='bold')

for idx, (name, model) in enumerate(models.items()):
    y_pred = results[name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                cbar=False, annot_kws={'size': 12})
    axes[idx].set_title(f'{name}\nAccuracy: {results[name]["accuracy"]:.2%}', fontsize=11)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
print("✓ Confusion matrices saved as 'confusion_matrices.png'")

# 3. ROC Curves for all models
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred_proba = results[name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=100, bbox_inches='tight')
print("✓ ROC curves saved as 'roc_curves.png'")

# 4. Feature Importance (Random Forest)
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=11)
plt.title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
print("✓ Feature importance saved as 'feature_importance.png'")

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
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print("\n✅ All models trained and evaluated successfully!")
print("=" * 70)