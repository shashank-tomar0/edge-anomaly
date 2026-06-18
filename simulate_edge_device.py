import numpy as np
import pandas as pd
import onnxruntime as ort
import os
import time
import json
import config

def calculate_mse(original, reconstructed):
    return np.mean(np.square(original - reconstructed))

def evaluate_onnx_model(model_path, normal_df, anomalous_df, threshold, num_samples=100):
    print(f"\nEvaluating ONNX model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return None
        
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Measure latency and classification
    latencies = []
    normal_mses = []
    normal_predictions = []
    
    # 1. Test Normal Data
    samples_to_test = min(num_samples, len(normal_df))
    for i in range(samples_to_test):
        sample = normal_df.iloc[i].values.astype(np.float32).reshape(1, -1)
        
        start_time = time.perf_counter()
        reconstructed = session.run(None, {input_name: sample})[0]
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)
        mse = calculate_mse(sample, reconstructed)
        normal_mses.append(mse)
        normal_predictions.append(mse >= threshold)
        
    # 2. Test Anomalous Data
    anom_mses = []
    anom_predictions = []
    anom_samples_to_test = min(num_samples, len(anomalous_df))
    for i in range(anom_samples_to_test):
        sample = anomalous_df.iloc[i].values.astype(np.float32).reshape(1, -1)
        
        start_time = time.perf_counter()
        reconstructed = session.run(None, {input_name: sample})[0]
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)
        mse = calculate_mse(sample, reconstructed)
        anom_mses.append(mse)
        anom_predictions.append(mse >= threshold)
        
    avg_latency_ms = np.mean(latencies) * 1000
    p95_latency_ms = np.percentile(latencies, 95) * 1000
    
    # Calculate detection metrics
    tp = np.sum(anom_predictions)
    fn = len(anom_predictions) - tp
    fp = np.sum(normal_predictions)
    tn = len(normal_predictions) - fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Average Inference Latency: {avg_latency_ms:.3f} ms (p95: {p95_latency_ms:.3f} ms)")
    print(f"Accuracy on Anomalies (Recall): {recall*100:.2f}% | False Alarm Rate (FPR): {fp/len(normal_predictions)*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "f1_score": f1,
        "recall": recall,
        "fpr": fp / len(normal_predictions)
    }

def run_edge_inference():
    print("Initializing edge device simulation...")
    
    # Load threshold from metadata
    if not os.path.exists(config.METADATA_PATH):
        print(f"Error: Metadata file {config.METADATA_PATH} not found. Run training command first.")
        return
        
    with open(config.METADATA_PATH, 'r') as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    print(f"Loaded statistical anomaly threshold from metadata: {threshold:.5f}")
    
    # Load data for evaluation
    if not os.path.exists(config.NORMAL_DATA_PATH) or not os.path.exists(config.ANOMALOUS_DATA_PATH):
        print("Error: Normal or anomalous data files not found. Run data generation first.")
        return
        
    normal_df = pd.read_csv(config.NORMAL_DATA_PATH)
    anomalous_df = pd.read_csv(config.ANOMALOUS_DATA_PATH)
    
    # Run evaluation on both FP32 and Quantized model
    results_fp32 = evaluate_onnx_model(config.ONNX_PATH, normal_df, anomalous_df, threshold)
    results_quant = evaluate_onnx_model(config.ONNX_QUANT_PATH, normal_df, anomalous_df, threshold)
    
    # Compare
    if results_fp32 and results_quant:
        print("\n=== Edge Inference Optimization Comparison ===")
        print(f"Model       | Avg Latency | p95 Latency | F1-Score | False Alarm Rate")
        print("-" * 70)
        print(f"FP32 ONNX   | {results_fp32['avg_latency_ms']:9.3f} ms | {results_fp32['p95_latency_ms']:9.3f} ms | {results_fp32['f1_score']:.4f}   | {results_fp32['fpr']*100:15.2f}%")
        print(f"INT8 Quant  | {results_quant['avg_latency_ms']:9.3f} ms | {results_quant['p95_latency_ms']:9.3f} ms | {results_quant['f1_score']:.4f}   | {results_quant['fpr']*100:15.2f}%")
        
        speedup = (results_fp32['avg_latency_ms'] / results_quant['avg_latency_ms'] - 1) * 100
        print(f"\nInference Speedup from Quantization: {speedup:.1f}%")
        
    # Simulate a stream run
    print("\n--- Simulating Live Sensor Feed (showing first 5 steps) ---")
    session = ort.InferenceSession(config.ONNX_QUANT_PATH if os.path.exists(config.ONNX_QUANT_PATH) else config.ONNX_PATH)
    input_name = session.get_inputs()[0].name
    
    print("\n[Normal Stream]")
    for i in range(5):
        sample = normal_df.iloc[i].values.astype(np.float32).reshape(1, -1)
        reconstructed = session.run(None, {input_name: sample})[0]
        mse = calculate_mse(sample, reconstructed)
        status = "OK" if mse < threshold else "⚠️ ANOMALY DETECTED"
        print(f"Time {i:02d} | Sensor MSE: {mse:.4f} | Status: {status}")
        
    print("\n[Anomalous Stream (Fault Injected)]")
    for i in range(5):
        sample = anomalous_df.iloc[i].values.astype(np.float32).reshape(1, -1)
        reconstructed = session.run(None, {input_name: sample})[0]
        mse = calculate_mse(sample, reconstructed)
        status = "OK" if mse < threshold else "🚨 ANOMALY DETECTED 🚨"
        print(f"Time {i:02d} | Sensor MSE: {mse:.4f} | Status: {status}")

if __name__ == "__main__":
    run_edge_inference()
