import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# UNDERGROUND CABLE FAULT DETECTION - DATASET GENERATION
# ============================================================================

np.random.seed(42)

# Number of samples
n_samples = 300

# Generate Features: voltage, current, temperature, resistance, insulation_resistance, age
# These are realistic parameters for underground power cables

# Normal operating conditions with some variations
voltage = np.random.normal(11, 1.5, n_samples)  # kV
current = np.random.normal(45, 12, n_samples)  # Amperes
temperature = np.random.normal(35, 8, n_samples)  # Celsius
resistance = np.random.normal(0.45, 0.08, n_samples)  # Ohms
insulation_resistance = np.random.normal(1.5, 0.3, n_samples)  # Megaohms
age = np.random.uniform(0, 30, n_samples)  # Years

# Create labels (0: No Fault, 1: Fault Detected)
# Fault detection logic based on physical parameters
fault = np.zeros(n_samples, dtype=int)

# Faulty cables have: high temperature AND degraded insulation OR high resistance
for i in range(n_samples):
    fault_indicators = 0
    if temperature[i] > 50:
        fault_indicators += 1
    if insulation_resistance[i] < 0.8:
        fault_indicators += 1
    if resistance[i] > 0.65:
        fault_indicators += 1
    if current[i] > 70:  # Overload condition
        fault_indicators += 1
    if age[i] > 20:  # Old cables more prone to faults
        fault_indicators += 1
    
    if fault_indicators >= 2:
        fault[i] = 1

# Create DataFrame
df = pd.DataFrame({
    'voltage': voltage,
    'current': current,
    'temperature': temperature,
    'resistance': resistance,
    'insulation_resistance': insulation_resistance,
    'cable_age': age,
    'label': fault
})

# Save dataset
df.to_csv("underground_cable_dataset.csv", index=False)

# Print detailed statistics
print("=" * 70)
print("UNDERGROUND CABLE FAULT DETECTION - DATASET OVERVIEW")
print("=" * 70)
print(f"\n✓ Dataset generated and saved as 'underground_cable_dataset.csv'")
print(f"\nDataset Statistics:")
print(f"  • Total samples: {len(df)}")
print(f"  • Fault-free samples: {sum(fault == 0)} ({100*sum(fault == 0)/len(df):.1f}%)")
print(f"  • Faulty samples: {sum(fault == 1)} ({100*sum(fault == 1)/len(df):.1f}%)")
print(f"  • Class balance ratio: {sum(fault == 1) / sum(fault == 0):.2f}:1")

print(f"\nFeature Statistics:")
print(df.describe().round(3))

print(f"\nDataset Preview (first 10 rows):")
print(df.head(10).to_string())

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Underground Cable Features Distribution by Fault Status', fontsize=14, fontweight='bold')

features = ['voltage', 'current', 'temperature', 'resistance', 'insulation_resistance', 'cable_age']

for idx, feature in enumerate(features):
    ax = axes[idx // 3, idx % 3]
    
    # Histogram with separation by fault status
    fault_data = df[df['label'] == 1][feature]
    normal_data = df[df['label'] == 0][feature]
    
    ax.hist(normal_data, bins=20, alpha=0.6, label='No Fault', color='green')
    ax.hist(fault_data, bins=20, alpha=0.6, label='Fault', color='red')
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100, bbox_inches='tight')
print("\n✓ Feature distributions saved as 'feature_distributions.png'")

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Correlation matrix saved as 'correlation_matrix.png'")

print("\n" + "=" * 70)