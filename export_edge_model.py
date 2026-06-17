import torch
from models.autoencoder import AnomalyAutoencoder
import os

def export_to_onnx():
    print("Loading PyTorch model...")
    sequence_length = 64
    model = AnomalyAutoencoder(sequence_length)
    
    if not os.path.exists('autoencoder_model.pth'):
        print("Error: autoencoder_model.pth not found. Run train.py first.")
        return
        
    model.load_state_dict(torch.load('autoencoder_model.pth'))
    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, sequence_length)

    print("Exporting to ONNX format (for edge deployment)...")
    torch.onnx.export(
        model, 
        dummy_input, 
        "edge_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model exported to edge_model.onnx")

if __name__ == "__main__":
    export_to_onnx()
