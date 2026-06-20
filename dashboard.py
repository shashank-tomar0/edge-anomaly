import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import time
import os

# Streamlit Page Config for Premium Look
st.set_page_config(
    page_title="Edge AI Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling and glassmorphism look
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .metric-card {
        background-color: #1a1e29;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #63b3ed;
    }
    .status-ok {
        background-color: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid #48bb78;
        border-radius: 8px;
        padding: 15px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
    .status-anomaly {
        background-color: rgba(245, 101, 101, 0.15);
        color: #f56565;
        border: 1px solid #f56565;
        border-radius: 8px;
        padding: 15px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate simulated vibration waveforms
def generate_sample(anomaly=False):
    t = np.linspace(0, 10, 64)
    if not anomaly:
        # Normal machine vibration: Base frequency + harmonic + slight noise
        vibration = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.1, 64)
    else:
        # Anomalous vibration: Different frequency, higher amplitude, more noise
        vibration = 2.0 * np.sin(2 * np.pi * 1.5 * t) + 1.0 * np.sin(2 * np.pi * 4.0 * t) + np.random.normal(0, 0.5, 64)
    return vibration.tolist()

# Sidebar Setup
st.sidebar.title("🔌 Connection settings")
default_url = os.environ.get("EDGE_ANOMALY_API_URL", "https://edge-anomaly.onrender.com")
api_url = st.sidebar.text_input("API URL Target", value=default_url).rstrip('/')

# Sidebar Health Check
connected = False
model_threshold = 0.5
model_seq_length = 64

try:
    health_resp = requests.get(f"{api_url}/health", timeout=3)
    if health_resp.status_code == 200:
        connected = True
        health_data = health_resp.json()
        st.sidebar.success(f"● Connected to Backend")
        st.sidebar.info(f"Active Model: {health_data['model_loaded']}")
        model_threshold = health_data["threshold"]
    else:
        st.sidebar.error(f"○ Offline (Status {health_resp.status_code})")
except Exception:
    st.sidebar.error("○ Server Offline / Unreachable")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About the Model")
st.sidebar.write(
    "This Autoencoder compresses a 64-sample time-series window into an 8-dimensional bottleneck. "
    "If the reconstruction MSE is above the statistical 99th percentile threshold, an anomaly is flagged."
)

# Dashboard Title
st.title("🔍 Edge AI Real-Time Anomaly Detection")
st.write("Visual dashboard showing machine vibration state, telemetry performance, and model reconstruction errors.")
st.markdown("---")

# Metrics Block
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

total_predictions = 0
anomalies_detected = 0
avg_inference_time = 0.0
anomaly_rate = 0.0

if connected:
    try:
        metrics_resp = requests.get(f"{api_url}/metrics", timeout=2)
        if metrics_resp.status_code == 200:
            m_data = metrics_resp.json()
            total_predictions = m_data["total_predictions"]
            anomalies_detected = m_data["anomalies_detected"]
            anomaly_rate = m_data["anomaly_rate"] * 100
            avg_inference_time = m_data["avg_inference_time_ms"]
    except Exception:
        pass

with metrics_col1:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin:0; font-size:14px; color:#a0aec0;">Total Predictions</p>
        <p class="metric-value">{total_predictions}</p>
    </div>
    """, unsafe_allow_html=True)

with metrics_col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin:0; font-size:14px; color:#a0aec0;">Anomaly Rate</p>
        <p class="metric-value" style="color: { '#f56565' if anomaly_rate > 10 else '#48bb78' };">{anomaly_rate:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

with metrics_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin:0; font-size:14px; color:#a0aec0;">Avg Inference Latency</p>
        <p class="metric-value" style="color:#ed8936;">{avg_inference_time:.2f} ms</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab_manual, tab_live = st.tabs(["🎮 Manual Simulation", "📡 Live Stream Feed"])

# Dashboard plotting helper
def render_plots(original, reconstructed, mse, threshold, status_container, plot_container):
    is_anomaly = mse >= threshold
    
    # Update Status Container
    if is_anomaly:
        status_container.markdown(f"""
        <div class="status-anomaly">
            🚨 CRITICAL FAULT: ANOMALY DETECTED (MSE: {mse:.5f} &ge; Threshold: {threshold:.5f})
        </div>
        """, unsafe_allow_html=True)
    else:
        status_container.markdown(f"""
        <div class="status-ok">
            ✅ SYSTEM OPERATING NORMALLY (MSE: {mse:.5f} &lt; Threshold: {threshold:.5f})
        </div>
        """, unsafe_allow_html=True)

    # Plot 1: Signal Waveform Comparison
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(
        y=original,
        mode='lines',
        name='Original Sensor Signal',
        line=dict(color='#63b3ed', width=2.5)
    ))
    fig_signal.add_trace(go.Scatter(
        y=reconstructed,
        mode='lines',
        name='Model Reconstructed Signal',
        line=dict(color='#ed8936', width=2, dash='dash')
    ))
    
    fig_signal.update_layout(
        title="Vibration Signal Reconstruction Overlay",
        xaxis_title="Time Steps",
        yaxis_title="Amplitude",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Plot 2: MSE Error Bar Chart
    fig_error = go.Figure()
    fig_error.add_trace(go.Bar(
        x=['Reconstruction Error (MSE)'],
        y=[mse],
        name='Current MSE',
        marker_color='#f56565' if is_anomaly else '#48bb78',
        width=0.4
    ))
    fig_error.add_trace(go.Scatter(
        x=['Reconstruction Error (MSE)', 'Reconstruction Error (MSE)'],
        y=[threshold, threshold],
        mode='lines',
        name='Anomaly Threshold',
        line=dict(color='#ed8936', width=3, dash='dash')
    ))
    
    fig_error.update_layout(
        title="MSE vs Threshold",
        yaxis_title="Error",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(range=[0, max(threshold, mse) * 1.2])
    )
    
    # Render both plots in columns
    with plot_container:
        plot_col1, plot_col2 = st.columns([2, 1])
        with plot_col1:
            st.plotly_chart(fig_signal, use_container_width=True)
        with plot_col2:
            st.plotly_chart(fig_error, use_container_width=True)

# Tab 1: Manual Simulation
with tab_manual:
    st.subheader("Manual Telemetry Injection")
    st.write("Trigger normal state or bearing fault anomalies manually to observe how the autoencoder behaves.")
    
    sim_col1, sim_col2 = st.columns(2)
    
    trigger_normal = sim_col1.button("🟢 Simulate Normal Operation", use_container_width=True)
    trigger_anomaly = sim_col2.button("🔴 Simulate Bearing Failure (Anomaly)", use_container_width=True)
    
    status_manual = st.empty()
    plots_manual = st.container()
    
    # Trigger Logic
    if trigger_normal or trigger_anomaly:
        if not connected:
            st.warning("Cannot run test: Backend is offline.")
        else:
            with st.spinner("Processing signal..."):
                anomaly_flag = trigger_anomaly
                signal = generate_sample(anomaly_flag)
                
                try:
                    resp = requests.post(f"{api_url}/predict", json={"data": signal}, timeout=5)
                    if resp.status_code == 200:
                        res_data = resp.json()
                        render_plots(
                            original=signal,
                            reconstructed=res_data["reconstructed"],
                            mse=res_data["mse"],
                            threshold=res_data["threshold"],
                            status_container=status_manual,
                            plot_container=plots_manual
                        )
                    else:
                        st.error(f"Prediction API failed with status: {resp.status_code}")
                except Exception as e:
                    st.error(f"Inference Connection Error: {e}")

# Tab 2: Live Stream Feed
with tab_live:
    st.subheader("Live Telemetry Stream Feed")
    st.write("Toggling the stream feeds simulated sensor streams to the backend in real-time, plotting results dynamically.")
    
    stream_active = st.checkbox("🛰️ Start Live Sensor Stream Simulation")
    
    status_live = st.empty()
    plots_live = st.container()
    
    if stream_active:
        if not connected:
            st.warning("Cannot start stream: Backend is offline.")
        else:
            try:
                while stream_active:
                    # Randomly inject anomaly 20% of the time
                    fault = np.random.random() < 0.2
                    signal = generate_sample(fault)
                    
                    resp = requests.post(f"{api_url}/predict", json={"data": signal}, timeout=3)
                    if resp.status_code == 200:
                        res_data = resp.json()
                        
                        # Render the plots inside containers to update in-place
                        with plots_live:
                            render_plots(
                                original=signal,
                                reconstructed=res_data["reconstructed"],
                                mse=res_data["mse"],
                                threshold=res_data["threshold"],
                                status_container=status_live,
                                plot_container=plots_live
                            )
                    else:
                        st.error(f"Inference failed with status: {resp.status_code}")
                        break
                        
                    # sleep for 1 second before feeding next sample
                    time.sleep(1.0)
            except Exception as e:
                st.error(f"Stream interrupted: {e}")
