# Edge AI Anomaly Detection 🔍

A PyTorch-based autoencoder designed to run on Edge Devices (like Raspberry Pi) for real-time sensor anomaly detection.

## Features
- **Autoencoder Architecture**: A neural network with a deep bottleneck capable of reconstructing normal sensor patterns and flagging high reconstruction errors as anomalies.
- **ONNX Export**: Compiles the trained PyTorch model into an `.onnx` binary for extreme performance optimization and portability.
- **Edge Simulation**: A robust inference pipeline using `onnxruntime` that simulates real-time ingestion of noisy sensor data, classifying anomalies in milliseconds.

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
