import numpy as np
import pandas as pd
import os

def generate_normal_data(samples=5000, sequence_length=64):
    np.random.seed(42)
    data = []
    for _ in range(samples):
        t = np.linspace(0, 10, sequence_length)
        # Normal machine vibration: Base frequency + harmonic + slight noise
        vibration = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.1, sequence_length)
        data.append(vibration)
    return pd.DataFrame(data)

def generate_anomalous_data(samples=500, sequence_length=64):
    np.random.seed(43)
    data = []
    for _ in range(samples):
        t = np.linspace(0, 10, sequence_length)
        # Anomalous vibration: Different frequency, higher amplitude, more noise (e.g. bearing failure)
        vibration = 2.0 * np.sin(2 * np.pi * 1.5 * t) + 1.0 * np.sin(2 * np.pi * 4.0 * t) + np.random.normal(0, 0.5, sequence_length)
        data.append(vibration)
    return pd.DataFrame(data)

if __name__ == "__main__":
    # The script might be run from the root directory or the data directory
    output_dir = 'data' if os.path.basename(os.getcwd()) != 'data' else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    normal_df = generate_normal_data()
    normal_df.to_csv(os.path.join(output_dir, 'normal_data.csv'), index=False)
    
    anomalous_df = generate_anomalous_data()
    anomalous_df.to_csv(os.path.join(output_dir, 'anomalous_data.csv'), index=False)
    
    print(f"Simulated data generated in {output_dir}")
