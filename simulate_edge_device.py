import numpy as np
import pandas as pd
import onnxruntime as ort
import os

def calculate_mse(original, reconstructed):
    return np.mean(np.square(original - reconstructed))

def run_edge_inference():
    print("Initializing edge device simulation (e.g., Raspberry Pi)...")
    model_path = "edge_model.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run export_edge_model.py first.")
        return
        
    print("Loading quantized edge_model.onnx into memory...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Set a threshold for what constitutes an anomaly.
    # In a real system, you'd calculate this based on the max MSE of the validation set during training.
    THRESHOLD = 0.5 
    
    print("\n--- Starting Real-Time Sensor Stream ---")
    
    # 1. Test Normal Data
    print("\n[STREAM A] Testing Normal Operating Conditions...")
    normal_df = pd.read_csv('data/normal_data.csv')
    for i in range(5):
        sample = normal_df.iloc[i].values.astype(np.float32).reshape(1, -1)
        reconstructed = session.run(None, {input_name: sample})[0]
        mse = calculate_mse(sample, reconstructed)
        status = "OK" if mse < THRESHOLD else "ANOMALY DETECTED"
        print(f"Reading {i}: MSE = {mse:.4f} | Status: {status}")
        
    # 2. Test Anomalous Data
    print("\n[STREAM B] Injecting Anomalous Sequence (e.g. Bearing Fault)...")
    anomalous_df = pd.read_csv('data/anomalous_data.csv')
    for i in range(5):
        sample = anomalous_df.iloc[i].values.astype(np.float32).reshape(1, -1)
        reconstructed = session.run(None, {input_name: sample})[0]
        mse = calculate_mse(sample, reconstructed)
        status = "OK" if mse < THRESHOLD else "🚨 ANOMALY DETECTED 🚨"
        print(f"Reading {i}: MSE = {mse:.4f} | Status: {status}")

if __name__ == "__main__":
    run_edge_inference()
