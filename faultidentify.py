import pandas as pd
import numpy as np
from datetime import datetime

# ===============================================================
# UNDERGROUND CABLE FAULT IDENTIFICATION SYSTEM
# ===============================================================

print("Loading dataset...")

df = pd.read_csv("underground_cable_dataset.csv")

faulty_cables = df[df["label"] == 1]

if len(faulty_cables) > 0:

    for idx, row in faulty_cables.iterrows():

        print("\n" + "="*80)
        print("🚨 UNDERGROUND CABLE FAULT ALERT")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Cable ID: CABLE-{idx:04d}")

        # Fault location
        print("\n📍 FAULT LOCATION:")
        print(f"   Total Cable Length: {row['cable_length_km']:.2f} km")
        print(f"   Fault Distance: {row['fault_distance_km']:.2f} km from source")
        print(f"   Fault Position: {(row['fault_distance_km']/row['cable_length_km'])*100:.1f}% of cable length")

        # Current Parameters
        print("\n📊 CURRENT PARAMETERS:")
        print(f"   Voltage: {row['voltage']:.2f} kV")
        print(f"   Current: {row['current']:.2f} A")
        print(f"   Temperature: {row['temperature']:.2f} °C")
        print(f"   Resistance: {row['resistance']:.3f} Ω")
        print(f"   Insulation Resistance: {row['insulation_resistance']:.2f} MΩ")
        print(f"   Cable Age: {row['cable_age']:.1f} years")

        # Parameter deviation from normal
        print("\n📈 PARAMETER DEVIATION:")
        print(f"   Temperature Deviation: {row['temperature'] - 35:.2f} °C")
        print(f"   Current Deviation: {row['current'] - 45:.2f} A")
        print(f"   Resistance Deviation: {row['resistance'] - 0.45:.3f} Ω")
        print(f"   Insulation Drop: {1.5 - row['insulation_resistance']:.2f} MΩ")

        # Severity estimation
        severity_score = 0
        
        if row['temperature'] > 50:
            severity_score += 1
        if row['current'] > 70:
            severity_score += 1
        if row['resistance'] > 0.65:
            severity_score += 1
        if row['insulation_resistance'] < 0.8:
            severity_score += 1
        if row['cable_age'] > 20:
            severity_score += 1

        if severity_score >= 4:
            severity = "CRITICAL"
        elif severity_score >= 2:
            severity = "HIGH"
        else:
            severity = "MODERATE"

        print(f"\n⚠️ SEVERITY LEVEL: {severity}")
        print("="*80)

    print(f"\nTotal Faulty Cables Detected: {len(faulty_cables)}")

else:
    print("\n" + "="*80)
    print("✅ ALL CABLES OPERATING NORMALLY")
    print("="*80)
    print(f"Inspection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Cables Scanned: {len(df)}")
    print("Status: HEALTHY")
    print("="*80)