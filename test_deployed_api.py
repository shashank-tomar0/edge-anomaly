import requests
import numpy as np
import sys
import time

def generate_sample(anomaly=False):
    t = np.linspace(0, 10, 64)
    if not anomaly:
        # Normal machine vibration: Base frequency + harmonic + slight noise
        vibration = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.1, 64)
    else:
        # Anomalous vibration: Different frequency, higher amplitude, more noise
        vibration = 2.0 * np.sin(2 * np.pi * 1.5 * t) + 1.0 * np.sin(2 * np.pi * 4.0 * t) + np.random.normal(0, 0.5, 64)
    return vibration.tolist()

def main():
    print("==================================================")
    print("   Edge AI Anomaly Detection Cloud Test Client    ")
    print("==================================================")
    
    url = input("Enter your deployed public API URL (e.g., https://edge-anomaly.onrender.com): ").strip()
    if not url:
        print("Error: URL cannot be empty.")
        sys.exit(1)
        
    # Clean trailing slashes
    url = url.rstrip('/')
    
    # 1. Health Check
    print(f"\n[1/3] Connecting to health endpoint at {url}/health ...")
    try:
        response = requests.get(f"{url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Connection successful! Status: {data['status']}")
            print(f"   Model used: {data['model_loaded']}")
            print(f"   Model Threshold: {data['threshold']:.5f}")
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to the API: {e}")
        print("Please check that the URL is correct and the server is fully booted (Render free tier may take a minute to spin up).")
        sys.exit(1)

    # 2. Test Single Prediction
    print("\n[2/3] Sending single sensor reading (Normal Sequence) to /predict ...")
    normal_sample = generate_sample(anomaly=False)
    payload = {"data": normal_sample}
    
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{url}/predict", json=payload, timeout=10)
        latency = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            res = response.json()
            status_text = "🚨 ANOMALY" if res["is_anomaly"] else "✅ OK"
            print(f"   Result: {status_text} (MSE: {res['mse']:.5f} | Threshold: {res['threshold']:.5f})")
            print(f"   Network round-trip latency: {latency:.2f} ms")
        else:
            print(f"❌ Prediction failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 3. Test Batch Prediction
    print("\n[3/3] Sending batch of readings (1 Normal, 1 Anomalous) to /predict ...")
    batch_payload = {
        "data": [
            generate_sample(anomaly=False),
            generate_sample(anomaly=True)
        ]
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{url}/predict", json=batch_payload, timeout=10)
        latency = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            res = response.json()
            print(f"\n   --- Batch Results (Round-trip: {latency:.2f} ms) ---")
            print(f"   {'Index':<6} | {'Reconstruction MSE':<20} | {'Status':<12}")
            print(f"   {'-'*45}")
            for idx, item in enumerate(res["results"]):
                status_text = "🚨 ANOMALY" if item["is_anomaly"] else "✅ OK"
                print(f"   {idx:<6} | {item['mse']:<20.5f} | {status_text:<12}")
            print(f"\n   Total Samples: {res['total_samples']} | Anomalies Flagged: {res['anomalies_count']}")
        else:
            print(f"❌ Batch prediction failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n==================================================")
    print("Testing complete. Open your browser and visit:")
    print(f"{url}/docs  <- for interactive Swagger API docs")
    print("==================================================")

if __name__ == "__main__":
    main()
