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

## 📦 Production Deployment Playbook

### Option A: Cloud Deployment (Render Free Tier - Recommended)
Render allows you to deploy Docker containers directly from your GitHub repository for free:

1. **Push your code**: Ensure all changes are committed and pushed to your GitHub repository.
2. **Sign up for Render**: Go to [Render](https://render.com) and log in.
3. **Create a Web Service**:
   - Click **New +** and select **Web Service**.
   - Connect your GitHub account and select the `edge-anomaly` repository.
4. **Configure Deployment Settings**:
   - **Name**: `edge-anomaly` (or custom name).
   - **Environment**: **Docker** (Render will automatically detect your `Dockerfile`).
   - **Instance Type**: **Free** (CPU and RAM limits are perfectly fine for this lightweight API).
5. **Deploy**: Click **Create Web Service**. 
   - Render will build your Docker image and deploy it.
   - Once the deployment is complete, Render will provide a public URL (e.g., `https://edge-anomaly-xxxx.onrender.com`).
   - Open `https://your-app.onrender.com/docs` to see the live interactive Swagger UI!

---

### Option B: Local Docker Container Deployment
Use the included `Dockerfile` to containerize and run the server locally:

1. **Build the Docker image**:
   ```bash
   docker build -t edge-anomaly-detector .
   ```
2. **Run the container**:
   ```bash
   docker run -d -p 8000:8000 edge-anomaly-detector
   ```
   Access the server at `http://localhost:8000/docs`.

---

### Option C: Deployment as a systemd Service (Raspberry Pi/Debian)
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

---

## 🧪 Testing the Deployed API

Once your API is deployed, you and your users can test it using the interactive test scripts.

### Option A: Install and Run the CLI Tool Globally (Recommended)
Anyone can install the interactive terminal client globally on their machine directly from GitHub:

1. **Install globally**:
   ```bash
   pip install git+https://github.com/shashank-tomar0/edge-anomaly.git
   ```
2. **Run the client from anywhere**:
   Simply run the command:
   ```bash
   edge-anomaly-cli
   ```
3. *(Optional)* **Set default API URL via Environment Variable**:
   To avoid typing the URL every time:
   - **Linux/macOS**: `export EDGE_ANOMALY_API_URL="https://your-app.onrender.com"`
   - **Windows PowerShell**: `$env:EDGE_ANOMALY_API_URL="https://your-app.onrender.com"`

---

### Option B: Run the Deployment Test Client
Alternatively, run the standalone test script:

1. Run the test script:
   ```bash
   python test_deployed_api.py
   ```
2. Enter your public URL when prompted to verify basic single and batch predictions.


