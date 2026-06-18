import os
import time
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

import config

# Initialize FastAPI app
app = FastAPI(
    title="Edge AI Anomaly Detection Service",
    description="A deploy-ready service running on-device for real-time sensor anomaly detection using autoencoders.",
    version="1.0.0"
)

# Global state for loaded model and metadata
model_session = None
model_meta = {}
model_file_used = ""
input_name = ""

# Telemetry metrics
telemetry = {
    "total_predictions": 0,
    "anomalies_detected": 0,
    "total_inference_time_ms": 0.0
}

class PredictionRequest(BaseModel):
    data: Union[List[float], List[List[float]]] = Field(
        ...,
        description="A list of 64 sensor readings, or a nested list (batch) of shape [BatchSize, 64].",
        example=[0.1] * 64
    )

class PredictionResult(BaseModel):
    mse: float = Field(..., description="Reconstruction mean squared error.")
    is_anomaly: bool = Field(..., description="Flag indicating if the reading is anomalous.")
    threshold: float = Field(..., description="The threshold used for classification.")

class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    total_samples: int
    anomalies_count: int

@app.on_event("startup")
def load_model():
    global model_session, model_meta, model_file_used, input_name
    
    # 1. Load metadata
    if not os.path.exists(config.METADATA_PATH):
        raise RuntimeError(f"Metadata file {config.METADATA_PATH} not found. Run training command first.")
        
    with open(config.METADATA_PATH, 'r') as f:
        model_meta = json.load(f)
        
    # 2. Select ONNX model file (prefer quantized)
    if os.path.exists(config.ONNX_QUANT_PATH):
        model_file_used = config.ONNX_QUANT_PATH
        print(f"Loading optimized quantized model: {model_file_used}")
    elif os.path.exists(config.ONNX_PATH):
        model_file_used = config.ONNX_PATH
        print(f"Loading standard FP32 model: {model_file_used}")
    else:
        raise RuntimeError("No ONNX model found. Run export command first.")
        
    # 3. Create Inference Session
    model_session = ort.InferenceSession(model_file_used)
    input_name = model_session.get_inputs()[0].name
    print(f"Successfully loaded ONNX model. Input node name: {input_name}")

@app.get("/")
def read_root():
    return {
        "service": "Edge AI Anomaly Detection API",
        "status": "online",
        "model_file": os.path.basename(model_file_used),
        "threshold": model_meta.get("threshold", 0.0)
    }

@app.get("/health")
def health_check():
    if model_session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
    return {
        "status": "healthy",
        "model_loaded": os.path.basename(model_file_used),
        "threshold": model_meta.get("threshold", 0.0)
    }

@app.get("/info", response_model=Dict[str, Any])
def get_info():
    if not model_meta:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadata not loaded."
        )
    return {
        "model_name": "AnomalyAutoencoder",
        "parameters": {
            "sequence_length": model_meta.get("sequence_length"),
            "latent_dim": model_meta.get("latent_dim"),
            "threshold": model_meta.get("threshold"),
            "threshold_percentile": model_meta.get("threshold_percentile")
        },
        "training_performance": {
            "final_train_loss": model_meta.get("final_train_loss"),
            "final_val_loss": model_meta.get("final_val_loss")
        },
        "evaluation_metrics": model_meta.get("metrics", {})
    }

@app.get("/metrics")
def get_metrics():
    avg_inference_time = (
        telemetry["total_inference_time_ms"] / telemetry["total_predictions"]
        if telemetry["total_predictions"] > 0 else 0.0
    )
    anomaly_rate = (
        telemetry["anomalies_detected"] / telemetry["total_predictions"]
        if telemetry["total_predictions"] > 0 else 0.0
    )
    return {
        "total_predictions": telemetry["total_predictions"],
        "anomalies_detected": telemetry["anomalies_detected"],
        "anomaly_rate": anomaly_rate,
        "avg_inference_time_ms": avg_inference_time
    }

@app.post("/predict", response_model=Union[PredictionResult, BatchPredictionResult])
def predict(payload: PredictionRequest):
    global model_session, input_name, model_meta
    
    if model_session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
        
    threshold = model_meta.get("threshold", 0.5)
    sequence_length = model_meta.get("sequence_length", 64)
    
    raw_data = payload.data
    
    # Check if input is a flat list (single sample)
    if isinstance(raw_data[0], (int, float)):
        # Single sample input
        if len(raw_data) != sequence_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input sequence length must be {sequence_length}. Got {len(raw_data)}."
            )
        input_array = np.array(raw_data, dtype=np.float32).reshape(1, -1)
        is_batch = False
    else:
        # Batch input
        for idx, seq in enumerate(raw_data):
            if len(seq) != sequence_length:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sample at index {idx} has length {len(seq)}, but must be {sequence_length}."
                )
        input_array = np.array(raw_data, dtype=np.float32)
        is_batch = True
        
    try:
        # Run inference
        start_time = time.perf_counter()
        reconstructed = model_session.run(None, {input_name: input_array})[0]
        end_time = time.perf_counter()
        
        # Performance logging
        inference_time_ms = (end_time - start_time) * 1000
        
        # Calculate MSE
        # Reconstruction error per sample
        mses = np.mean(np.square(input_array - reconstructed), axis=1)
        
        results = []
        anomalies_in_batch = 0
        for mse in mses:
            is_anomaly = bool(mse >= threshold)
            if is_anomaly:
                anomalies_in_batch += 1
            results.append(PredictionResult(
                mse=float(mse),
                is_anomaly=is_anomaly,
                threshold=threshold
            ))
            
        # Update metrics
        num_samples = len(mses)
        telemetry["total_predictions"] += num_samples
        telemetry["anomalies_detected"] += anomalies_in_batch
        telemetry["total_inference_time_ms"] += inference_time_ms
        
        if is_batch:
            return BatchPredictionResult(
                results=results,
                total_samples=num_samples,
                anomalies_count=anomalies_in_batch
            )
        else:
            return results[0]
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model execution failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
