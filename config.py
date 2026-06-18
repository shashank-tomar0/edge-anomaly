import os

# Centralized Configuration Parameters
SEQUENCE_LENGTH = 64
LATENT_DIM = 8

# Training parameters
TRAIN_SPLIT = 0.8
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 99.0  # Percentile of normal validation reconstruction error to set as threshold

# File paths
MODEL_PATH = "autoencoder_model.pth"
METADATA_PATH = "model_meta.json"
ONNX_PATH = "edge_model.onnx"
ONNX_QUANT_PATH = "edge_model_quantized.onnx"
NORMAL_DATA_PATH = os.path.join("data", "normal_data.csv")
ANOMALOUS_DATA_PATH = os.path.join("data", "anomalous_data.csv")
