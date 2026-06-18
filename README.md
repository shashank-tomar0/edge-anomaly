# Edge AI Anomaly Detection 🔍

An optimized, deploy-ready PyTorch autoencoder pipeline designed to run on Edge Devices (such as Raspberry Pi, Jetson Nano, or industrial gateways) for real-time sensor anomaly detection.

The system features dynamic INT8 quantization for performance optimization and a REST API server built with FastAPI for production deployment.

---

## 🛠️ Architecture & Features

1. **Autoencoder Network**: Compresses 64-dimensional sensor sequence readings into an 8-dimensional latent space bottleneck to learn normal machine state. Reconstruction errors (MSE) exceeding a statistically calculated threshold are flagged as anomalies.
2. **Unified Command-Line Interface (CLI)**: Orchestrates the entire life-cycle from data simulation to production serving under a single entrypoint `main.py`.
3. **Statistical Thresholding**: Splits data into Train/Validation sets, computes reconstruction MSEs on validation samples, and automatically sets the anomaly threshold at a configurable percentile (default: 99th percentile).
4. **ONNX Optimization & INT8 Quantization**: Exports PyTorch models to ONNX and performs dynamic INT8 quantization (using `onnxruntime.quantization`) to minimize CPU usage and speed up execution on low-resource hardware.
5. **Production REST API Server**: FastAPI service with Swagger UI, endpoints for prediction (supports batch and single inputs), health monitoring, and system metrics.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Use the unified CLI to run the pipeline:

```bash
# Step 1: Generate simulated sensor data (normal and anomalous)
python main.py generate-data

# Step 2: Train model, calculate threshold, and save metadata
python main.py train

# Step 3: Export the trained model to ONNX & perform INT8 quantization
python -X utf8 main.py export

# Step 4: Simulate real-time inference and compare FP32 vs. INT8 performance
python -X utf8 main.py simulate

# Step 5: Start the API server locally
python main.py serve
```

---

## 📊 Quantization Analysis & Trade-offs

During testing, we observe the following performance trade-offs on edge CPUs:

| Model Version | Average Latency | p95 Latency | F1-Score | False Alarm Rate (FPR) | File Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 ONNX** | **~0.20 ms** | **~0.23 ms** | **1.0000** | **0.00%** | **16.7 KB** |
| **INT8 Quantized** | ~0.28 ms | ~0.49 ms | 0.6803 | 94.00% | 17.3 KB |

> [!WARNING]
> For tiny models (such as this 16.7 KB autoencoder), **standard FP32 ONNX is highly recommended**.
> 1. **Quantization Noise**: Rounding weights to INT8 introduces quantization noise, which raises the baseline reconstruction error and causes high False Alarm Rates (FPR) unless the threshold is re-calibrated specifically for the quantized model.
> 2. **Inference Overhead**: On small tensor sizes, CPU float/integer conversion overhead outpaces any computational savings of INT8, making the quantized model slightly slower than its FP32 counterpart.

---

## 🌐 Deploying the Inference Server

The server runs a lightweight FastAPI application that is ready to be deployed on an edge gateway or cloud instance.

### FastAPI Endpoints
- **`GET /health`**: Health status and current model metadata.
- **`GET /info`**: Details on model parameters, threshold, and training metrics.
- **`GET /metrics`**: Service telemetry including total requests, anomaly rate, and average latency.
- **`POST /predict`**: Score sensor sequences. Supports single sequences or batches.

**Request Payload Example (`POST /predict`):**
```json
{
  "data": [0.1, 0.2, -0.1, 0.05, ...] // List of 64 floats
}
```

**Response Example:**
```json
{
  "mse": 0.0084,
  "is_anomaly": false,
  "threshold": 0.01412
}
```

---

## 📦 Production Deployment Guide

### Option A: Deployment as a systemd Service (Raspberry Pi/Debian)
To run the server as a background service that automatically starts on boot:

1. Create a service file `/etc/systemd/system/edge-anomaly.service`:
   ```ini
   [Unit]
   Description=Edge AI Anomaly Detection API
   After=network.target

   [Service]
   User=pi
   WorkingDirectory=/home/pi/edge-ai-anomaly
   ExecStart=/home/pi/edge-ai-anomaly/venv/bin/uvicorn edge_server:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
2. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable edge-anomaly.service
   sudo systemctl start edge-anomaly.service
   ```

### Option B: Docker Container Deployment
Use Docker to containerize and isolate the application:

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Build and run the container:
   ```bash
   docker build -t edge-anomaly-detector .
   docker run -d -p 8000:8000 edge-anomaly-detector
   ```
