import requests
import numpy as np
import time
import os
import sys

# ANSI Escape Codes for Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[H\033[2J"

DEFAULT_URL = "http://localhost:8000"

def init_ansi():
    # Enable ANSI colors on Windows CMD/PowerShell if supported
    if sys.platform == "win32":
        os.system("color")

def clear():
    print(CLEAR_SCREEN, end="")

def print_header(title):
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f" {BOLD}{MAGENTA}{title.center(58)}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

def generate_signal(anomaly=False):
    t = np.linspace(0, 10, 64)
    if not anomaly:
        # Normal vibration pattern
        vibration = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.1, 64)
    else:
        # Anomalous vibration pattern (faulty bearing)
        vibration = 2.0 * np.sin(2 * np.pi * 1.5 * t) + 1.0 * np.sin(2 * np.pi * 4.0 * t) + np.random.normal(0, 0.5, 64)
    return vibration.tolist()

def plot_ascii_waveform(signal, height=7):
    """Draws a beautiful ASCII waveform plot in the terminal."""
    sig_min, sig_max = min(signal), max(signal)
    sig_range = sig_max - sig_min if sig_max != sig_min else 1.0
    
    # Create empty character canvas
    canvas = [[" " for _ in range(64)] for _ in range(height)]
    
    # Map each of the 64 readings to [0, height-1] row index
    for col, val in enumerate(signal):
        normalized = (val - sig_min) / sig_range
        row = int(normalized * (height - 1))
        # Invert row index so higher values appear at the top
        row = (height - 1) - row
        canvas[row][col] = "•"
        
    # Print the canvas with border
    print(f"   ┌{'─'*64}┐")
    for r in range(height):
        row_str = "".join(canvas[r])
        print(f"   │{CYAN}{row_str}{RESET}│")
    print(f"   └{'─'*64}┘")
    print(f"   Min: {sig_min:.2f} | Max: {sig_max:.2f}\n")

def check_connection(url):
    try:
        response = requests.get(f"{url}/health", timeout=3)
        if response.status_code == 200:
            return True, response.json()
    except Exception:
        pass
    return False, None

def run_single_test(url, is_anomaly):
    clear()
    label = "Anomalous" if is_anomaly else "Normal"
    print_header(f"Testing Single {label} Reading")
    
    print("Generating signal waveform...")
    signal = generate_signal(is_anomaly)
    plot_ascii_waveform(signal)
    
    print(f"Sending payload to {BOLD}{url}/predict{RESET} ...")
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{url}/predict", json={"data": signal}, timeout=5)
        latency = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            res = response.json()
            mse = res["mse"]
            threshold = res["threshold"]
            is_detected = res["is_anomaly"]
            
            print(f"\n{BOLD}--- Prediction Response ---{RESET}")
            print(f"Reconstruction MSE : {BOLD}{mse:.6f}{RESET}")
            print(f"Anomaly Threshold  : {threshold:.6f}")
            
            if is_detected:
                print(f"Status             : {BOLD}{RED}🚨 ANOMALY DETECTED{RESET}")
            else:
                print(f"Status             : {BOLD}{GREEN}✅ NORMAL OPERATION (OK){RESET}")
            print(f"Latency (Rountrip) : {CYAN}{latency:.2f} ms{RESET}")
        else:
            print(f"{RED}Error: Server returned status code {response.status_code}{RESET}")
    except Exception as e:
        print(f"{RED}Connection Error: {e}{RESET}")
        
    input("\nPress Enter to return to main menu...")

def run_batch_test(url):
    clear()
    print_header("Testing Batch Predictions")
    
    print("Generating 5 normal sequences and 5 anomalous sequences...")
    batch_data = []
    labels = []
    
    for _ in range(5):
        batch_data.append(generate_signal(anomaly=False))
        labels.append("Normal")
    for _ in range(5):
        batch_data.append(generate_signal(anomaly=True))
        labels.append("Anomaly")
        
    print(f"Sending batch payload (10 samples) to {BOLD}{url}/predict{RESET} ...")
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{url}/predict", json={"data": batch_data}, timeout=5)
        latency = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            res = response.json()
            print(f"\n{BOLD}--- Batch Results (Roundtrip: {latency:.2f} ms) ---{RESET}\n")
            print(f"   {BOLD}{'Index':<5} | {'Expected':<10} | {'Calculated MSE':<16} | {'Status':<14}{RESET}")
            print("   " + "-" * 53)
            
            for idx, item in enumerate(res["results"]):
                expected = labels[idx]
                mse = item["mse"]
                is_anomaly = item["is_anomaly"]
                
                if is_anomaly:
                    status_str = f"{RED}🚨 ANOMALY{RESET}"
                else:
                    status_str = f"{GREEN}✅ OK{RESET}"
                    
                color_expected = GREEN if expected == "Normal" else RED
                print(f"   {idx:<5} | {color_expected}{expected:<10}{RESET} | {mse:<16.5f} | {status_str:<14}")
                
            print(f"\n   Total Scored: {BOLD}{res['total_samples']}{RESET} | Anomaly Detections: {BOLD}{RED}{res['anomalies_count']}{RESET}")
        else:
            print(f"{RED}Error: Server returned status code {response.status_code}{RESET}")
    except Exception as e:
        print(f"{RED}Connection Error: {e}{RESET}")
        
    input("\nPress Enter to return to main menu...")

def run_stream_simulation(url):
    clear()
    print_header("Real-Time Sensor Stream Simulation")
    print(f"Feeding live sensor telemetry to {BOLD}{url}{RESET} every 1s...")
    print(f"Press {BOLD}Ctrl+C{RESET} to stop the stream.\n")
    print(f"   {BOLD}{'Timestamp':<10} | {'Sensor Value Range':<22} | {'MSE':<10} | {'Status Alert':<15}{RESET}")
    print("   " + "-" * 67)
    
    try:
        step = 0
        while True:
            # Randomly inject anomalies 25% of the time
            inject_fault = (np.random.random() < 0.25)
            signal = generate_signal(inject_fault)
            
            sig_min = min(signal)
            sig_max = max(signal)
            
            timestamp = time.strftime("%H:%M:%S")
            
            try:
                response = requests.post(f"{url}/predict", json={"data": signal}, timeout=2)
                if response.status_code == 200:
                    res = response.json()
                    mse = res["mse"]
                    is_anomaly = res["is_anomaly"]
                    
                    if is_anomaly:
                        status_str = f"{BOLD}{RED}🚨 ANOMALY ALERT{RESET}"
                    else:
                        status_str = f"{GREEN}Normal (OK){RESET}"
                        
                    print(f"   {timestamp:<10} | {sig_min:>9.2f} to {sig_max:<8.2f} | {mse:<10.5f} | {status_str:<15}")
                else:
                    print(f"   {timestamp:<10} | {RED}Error {response.status_code}{RESET}")
            except Exception as e:
                print(f"   {timestamp:<10} | {RED}Connection lost ({e}){RESET}")
                
            time.sleep(1.0)
            step += 1
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stream stopped by user.{RESET}")
        input("\nPress Enter to return to main menu...")

def show_server_metrics(url):
    clear()
    print_header("Server Performance Telemetry")
    
    try:
        # Get Info
        info_resp = requests.get(f"{url}/info", timeout=3)
        metrics_resp = requests.get(f"{url}/metrics", timeout=3)
        
        if info_resp.status_code == 200 and metrics_resp.status_code == 200:
            info = info_resp.json()
            metrics = metrics_resp.json()
            
            print(f"{BOLD}--- Model Metadata ---{RESET}")
            print(f"Model Architecture  : {info['model_name']}")
            print(f"Sequence Length     : {info['parameters']['sequence_length']}")
            print(f"Latent Bottleneck   : {info['parameters']['latent_dim']}")
            print(f"Anomaly Threshold   : {CYAN}{info['parameters']['threshold']:.6f}{RESET} ({info['parameters']['threshold_percentile']}th percentile)")
            print(f"Training Val Loss   : {info['training_performance']['final_val_loss']:.6f}")
            
            print(f"\n{BOLD}--- API Server Metrics ---{RESET}")
            print(f"Total Predictions   : {metrics['total_predictions']}")
            print(f"Anomalies Logged    : {RED}{metrics['anomalies_detected']}{RESET}")
            print(f"Anomaly Trigger Rate: {metrics['anomaly_rate']*100:.2f}%")
            print(f"Avg Inference Time  : {GREEN}{metrics['avg_inference_time_ms']:.3f} ms{RESET} (Server-side)")
        else:
            print(f"{RED}Failed to retrieve metrics from server.{RESET}")
    except Exception as e:
        print(f"{RED}Connection Error: {e}{RESET}")
        
    input("\nPress Enter to return to main menu...")

def main():
    init_ansi()
    url = DEFAULT_URL
    
    while True:
        clear()
        print_header("Edge AI Anomaly Detection Client")
        
        # Connection status block
        connected, health = check_connection(url)
        if connected:
            status_line = f"{GREEN}● CONNECTED{RESET} to {url} ({health['model_loaded']})"
        else:
            status_line = f"{RED}○ DISCONNECTED{RESET} from {url}"
            
        print(f"Status: {status_line}\n")
        print("Menu options:")
        print(f"  [{CYAN}1{RESET}] Configure Target API URL")
        print(f"  [{CYAN}2{RESET}] Send Normal Sequence Reading")
        print(f"  [{CYAN}3{RESET}] Send Anomalous Sequence Reading")
        print(f"  [{CYAN}4{RESET}] Test Batch Load (10 samples)")
        print(f"  [{CYAN}5{RESET}] Live Sensor Stream Simulation")
        print(f"  [{CYAN}6{RESET}] View Server Telemetry & Metrics")
        print(f"  [{CYAN}Q{RESET}] Quit Client")
        print(f"\n{'='*60}")
        
        choice = input("\nEnter your selection: ").strip().lower()
        
        if choice == '1':
            new_url = input(f"Enter target URL (default: {DEFAULT_URL}): ").strip()
            if not new_url:
                url = DEFAULT_URL
            else:
                if not new_url.startswith("http://") and not new_url.startswith("https://"):
                    new_url = "http://" + new_url
                url = new_url
        elif choice == '2':
            run_single_test(url, is_anomaly=False)
        elif choice == '3':
            run_single_test(url, is_anomaly=True)
        elif choice == '4':
            run_batch_test(url)
        elif choice == '5':
            run_stream_simulation(url)
        elif choice == '6':
            show_server_metrics(url)
        elif choice == 'q':
            clear()
            print("\nExiting Edge AI Client. Keep monitoring! 🔍\n")
            sys.exit(0)
        else:
            print(f"{RED}Invalid selection. Press Enter to try again...{RESET}")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
