import argparse
import sys
import os

def run_generate_data():
    print("Generating simulated sensor data...")
    # Add project root to sys.path to allow imports if running as main
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from data.simulate_sensor_data import generate_normal_data, generate_anomalous_data
        
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        
        normal_df = generate_normal_data()
        normal_df.to_csv(os.path.join(output_dir, 'normal_data.csv'), index=False)
        
        anomalous_df = generate_anomalous_data()
        anomalous_df.to_csv(os.path.join(output_dir, 'anomalous_data.csv'), index=False)
        
        print(f"Success: Simulated sensor data generated in '{output_dir}/'")
    except Exception as e:
        print(f"Error generating data: {e}")
        sys.exit(1)

def run_train():
    print("Starting model training and evaluation...")
    from train import train_pipeline
    train_pipeline()

def run_export():
    print("Starting ONNX export and quantization...")
    from export_edge_model import export_to_onnx
    export_to_onnx()

def run_simulate():
    print("Running edge device simulation...")
    from simulate_edge_device import run_edge_inference
    run_edge_inference()

def run_serve(host, port, reload):
    print(f"Starting FastAPI server on {host}:{port}...")
    import uvicorn
    uvicorn.run("edge_server:app", host=host, port=port, reload=reload)

def main():
    parser = argparse.ArgumentParser(
        description="Edge AI Anomaly Detection Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # generate-data parser
    subparsers.add_parser("generate-data", help="Generate simulated normal and anomalous sensor data")
    
    # train parser
    subparsers.add_parser("train", help="Train the autoencoder, compute threshold, and evaluate metrics")
    
    # export parser
    subparsers.add_parser("export", help="Export PyTorch model to ONNX & perform INT8 quantization")
    
    # simulate parser
    subparsers.add_parser("simulate", help="Simulate real-time edge device inference with ONNX models")
    
    # serve parser
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI REST API server for inference")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        run_generate_data()
    elif args.command == "train":
        run_train()
    elif args.command == "export":
        run_export()
    elif args.command == "simulate":
        run_simulate()
    elif args.command == "serve":
        run_serve(args.host, args.port, args.reload)
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
