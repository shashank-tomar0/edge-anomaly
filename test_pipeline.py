import unittest
import os
import json
import numpy as np
import torch
import pandas as pd
from fastapi.testclient import TestClient

# Add current path to import local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.autoencoder import AnomalyAutoencoder
from data.simulate_sensor_data import generate_normal_data, generate_anomalous_data
import config

class TestAnomalyPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Save original config paths if we want to isolate tests,
        # but here we can just use the config files or mock them.
        pass

    def test_01_data_generation(self):
        """Test that sensor simulation generates correct shapes."""
        normal_df = generate_normal_data(samples=10, sequence_length=64)
        anomalous_df = generate_anomalous_data(samples=5, sequence_length=64)
        
        self.assertEqual(normal_df.shape, (10, 64))
        self.assertEqual(anomalous_df.shape, (5, 64))
        self.assertTrue(isinstance(normal_df, pd.DataFrame))

    def test_02_model_forward_pass(self):
        """Test autoencoder architecture and forward pass."""
        model = AnomalyAutoencoder(sequence_length=64, latent_dim=8)
        dummy_input = torch.randn(4, 64)
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 64))

    def test_03_training_outputs_and_meta(self):
        """Check if metadata contains necessary fields after training runs."""
        # Check if metadata exists (if training has run)
        if os.path.exists(config.METADATA_PATH):
            with open(config.METADATA_PATH, 'r') as f:
                meta = json.load(f)
            self.assertIn("threshold", meta)
            self.assertIn("sequence_length", meta)
            self.assertIn("latent_dim", meta)
            self.assertEqual(meta["sequence_length"], config.SEQUENCE_LENGTH)
            self.assertEqual(meta["latent_dim"], config.LATENT_DIM)
            self.assertTrue(isinstance(meta["threshold"], float))

    def test_04_onnx_models(self):
        """Check if ONNX and quantized ONNX files exist and are valid."""
        # Standard ONNX model
        if os.path.exists(config.ONNX_PATH):
            self.assertTrue(os.path.getsize(config.ONNX_PATH) > 0)
        # Quantized ONNX model
        if os.path.exists(config.ONNX_QUANT_PATH):
            self.assertTrue(os.path.getsize(config.ONNX_QUANT_PATH) > 0)

    def test_05_api_endpoints(self):
        """Test FastAPI endpoints using TestClient."""
        # Only run if model files and metadata are present
        if not os.path.exists(config.METADATA_PATH) or not os.path.exists(config.ONNX_PATH):
            self.skipTest("ONNX model or metadata files not found. Run training/export first.")
            
        from edge_server import app, load_model
        
        # Manually invoke startup event to load model
        load_model()
        
        client = TestClient(app)
        
        # Test GET /health
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "healthy")
        
        # Test GET /info
        response = client.get("/info")
        self.assertEqual(response.status_code, 200)
        self.assertIn("parameters", response.json())
        self.assertIn("threshold", response.json()["parameters"])
        
        # Test POST /predict single sample
        single_reading = [0.1] * config.SEQUENCE_LENGTH
        response = client.post("/predict", json={"data": single_reading})
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertIn("mse", res_data)
        self.assertIn("is_anomaly", res_data)
        self.assertIn("threshold", res_data)
        self.assertTrue(isinstance(res_data["is_anomaly"], bool))
        
        # Test POST /predict batch samples
        batch_readings = [[0.1] * config.SEQUENCE_LENGTH, [0.5] * config.SEQUENCE_LENGTH]
        response = client.post("/predict", json={"data": batch_readings})
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertIn("results", res_data)
        self.assertEqual(res_data["total_samples"], 2)
        self.assertEqual(len(res_data["results"]), 2)

    def test_06_cli_client_generation(self):
        """Test cli_client helper signal generation."""
        from cli_client import generate_signal
        sig_normal = generate_signal(anomaly=False)
        sig_anomalous = generate_signal(anomaly=True)
        self.assertEqual(len(sig_normal), 64)
        self.assertEqual(len(sig_anomalous), 64)

if __name__ == "__main__":
    unittest.main()
