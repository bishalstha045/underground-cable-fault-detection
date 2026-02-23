import pandas as pd
import numpy as np

# ===============================================================
# UNDERGROUND CABLE FAULT DETECTION - DATASET GENERATION
# ===============================================================

n_samples = 120

print("Generating underground cable dataset...")

# Generate parameters
voltage = np.random.normal(11, 1.5, n_samples)
current = np.random.normal(45, 12, n_samples)
temperature = np.random.normal(35, 8, n_samples)
resistance = np.random.normal(0.45, 0.08, n_samples)
insulation_resistance = np.random.normal(1.5, 0.3, n_samples)
age = np.random.uniform(0, 30, n_samples)

# Simulate cable lengths (1km to 5km)
cable_length = np.random.uniform(1, 5, n_samples)

fault = np.zeros(n_samples, dtype=int)
fault_distance = np.zeros(n_samples)

# Randomly select 3-10 cables to have faults
num_faults = np.random.randint(3, 11)
fault_indices = np.random.choice(n_samples, size=num_faults, replace=False)

# Assign faults to randomly selected cables
for i in fault_indices:
    fault[i] = 1
    # Also increase parameters for faulty cables to make them more realistic
    temperature[i] += np.random.uniform(10, 20)
    insulation_resistance[i] -= np.random.uniform(0.3, 0.7)
    resistance[i] += np.random.uniform(0.1, 0.25)
    fault_distance[i] = np.random.uniform(0.1, cable_length[i])

# Set fault distance to 0 for non-faulty cables
for i in range(n_samples):
    if fault[i] == 0:
        fault_distance[i] = 0

# Create dataframe
df = pd.DataFrame({
    'voltage': voltage,
    'current': current,
    'temperature': temperature,
    'resistance': resistance,
    'insulation_resistance': insulation_resistance,
    'cable_age': age,
    'cable_length_km': cable_length,
    'fault_distance_km': fault_distance,
    'label': fault
})

df.to_csv("underground_cable_dataset.csv", index=False)

print(f"✓ Dataset created with {len(df)} samples")
print(f"  • Normal cables: {sum(fault == 0)}")
print(f"  • Faulty cables: {sum(fault == 1)}")
print("✓ Saved as underground_cable_dataset.csv")