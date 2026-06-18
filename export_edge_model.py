import torch
from models.autoencoder import AnomalyAutoencoder
import os
import config

def export_to_onnx():
    print("Loading PyTorch model...")
    sequence_length = config.SEQUENCE_LENGTH
    model = AnomalyAutoencoder(sequence_length=sequence_length, latent_dim=config.LATENT_DIM)
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: {config.MODEL_PATH} not found. Run training command first.")
        return
        
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, sequence_length)

    print("Exporting PyTorch model to ONNX format (FP32)...")
    torch.onnx.export(
        model, 
        dummy_input, 
        config.ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {config.ONNX_PATH}")

    # Apply dynamic quantization to INT8
    print("Performing INT8 dynamic quantization for edge optimization...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
        
        # Pre-process the model for quantization (fixes shape inference warnings/errors)
        preprocessed_path = config.ONNX_PATH.replace(".onnx", "_preprocessed.onnx")
        print("Running ONNX shape inference pre-processing...")
        shape_inference.quant_pre_process(
            config.ONNX_PATH,
            preprocessed_path
        )
        
        quantize_dynamic(
            model_input=preprocessed_path,
            model_output=config.ONNX_QUANT_PATH,
            weight_type=QuantType.QInt8
        )
        print(f"Quantized model saved to {config.ONNX_QUANT_PATH}")
        
        # Clean up temporary preprocessed file
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        
    except ImportError as e:
        print(f"Warning: Could not import onnxruntime.quantization ({e}). Dynamic quantization skipped.")
        print("Please make sure you have onnxruntime installed (pip install onnxruntime).")
        return

    # Print model size comparison
    pytorch_size = os.path.getsize(config.MODEL_PATH) / 1024
    onnx_size = os.path.getsize(config.ONNX_PATH) / 1024
    quant_size = os.path.getsize(config.ONNX_QUANT_PATH) / 1024 if os.path.exists(config.ONNX_QUANT_PATH) else 0.0

    print("\n--- Model Size Comparison ---")
    print(f"PyTorch State Dict: {pytorch_size:.2f} KB")
    print(f"ONNX FP32 Model:    {onnx_size:.2f} KB")
    if quant_size > 0:
        print(f"ONNX INT8 Quantized: {quant_size:.2f} KB (Reduction: {(1 - quant_size/onnx_size)*100:.1f}%)")

if __name__ == "__main__":
    export_to_onnx()
