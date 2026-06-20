import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import time
import os

# Streamlit Page Config for Premium Layout
st.set_page_config(
    page_title="Edge AI Anomaly Detection Portal",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Apple CSS Injection
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    /* 1. App Styling overrides */
    .stApp {
        background-color: #080B11;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 2. Apple Glassmorphic Card Styling */
    .glass-card {
        background: rgba(17, 22, 34, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #F3F4F6;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .card-desc {
        font-size: 13px;
        color: #9CA3AF;
        margin-bottom: 20px;
    }
    
    /* 3. Glowing Status Banner */
    .status-banner {
        border-left: 4px solid #3B82F6;
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 20px;
        background: rgba(17, 22, 34, 0.65);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left-width: 4px;
    }
    .status-banner.ok {
        border-left-color: #10B981;
        background: linear-gradient(to right, rgba(16, 185, 129, 0.05), transparent);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.05);
    }
    .status-banner.anomaly {
        border-left-color: #EF4444;
        background: linear-gradient(to right, rgba(239, 68, 68, 0.05), transparent);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.05);
        animation: flash-red 1.5s infinite alternate;
    }
    @keyframes flash-red {
        0% { border-left-color: #EF4444; }
        100% { border-left-color: rgba(239, 68, 68, 0.3); }
    }
    
    .status-icon-box {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    .status-banner.info .status-icon-box { background: rgba(59, 130, 246, 0.1); color: #3B82F6; }
    .status-banner.ok .status-icon-box { background: rgba(16, 185, 129, 0.15); color: #10B981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }
    .status-banner.anomaly .status-icon-box { background: rgba(239, 68, 68, 0.15); color: #EF4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }
    
    .status-label {
        font-size: 11px;
        color: #6B7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-value {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #F3F4F6;
        margin-top: 2px;
    }
    
    /* 4. Streamlit Button Customizations */
    div.stButton > button {
        border: none;
        outline: none;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 14px;
        padding: 12px 24px;
        border-radius: 12px;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Simulate Normal Button Styling */
    div.stButton > button[key="normal_btn"] {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
    }
    div.stButton > button[key="normal_btn"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3);
    }
    
    /* Inject Fault Button Styling */
    div.stButton > button[key="anomaly_btn"] {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
    }
    div.stButton > button[key="anomaly_btn"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.3);
    }
    
    /* 5. Metrics styling */
    .metric-grid-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding-bottom: 14px;
        margin-bottom: 14px;
    }
    .metric-grid-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0;
    }
    .metric-title {
        font-size: 13px;
        color: #9CA3AF;
    }
    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #F3F4F6;
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
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
    <i class="fa-solid fa-microchip" style="font-size: 24px; color: #3B82F6;"></i>
    <div>
        <h2 style="font-family: 'Outfit'; font-size: 18px; margin: 0;">Edge Anomaly</h2>
        <span style="font-size: 9px; color: #6B7280; font-weight: bold; letter-spacing: 1px;">MACHINE TELEMETRY</span>
    </div>
</div>
""", unsafe_allow_html=True)

default_url = os.environ.get("EDGE_ANOMALY_API_URL", "https://edge-anomaly.onrender.com")
api_url = st.sidebar.text_input("🔌 API Endpoint URL", value=default_url).rstrip('/')

# Sidebar Health Check
connected = False
model_threshold = 0.5
model_seq_length = 64
model_name = "AnomalyAutoencoder"

try:
    health_resp = requests.get(f"{api_url}/health", timeout=3)
    if health_resp.ok:
        connected = True
        health_data = health_resp.json()
        model_threshold = health_data["threshold"]
        model_name = health_data["model_loaded"]
        
        st.sidebar.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: #10B981; margin-bottom: 20px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #10B981; box-shadow: 0 0 8px #10B981;"></span>
            <span>Online ({model_name})</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: #EF4444; margin-bottom: 20px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #EF4444; box-shadow: 0 0 8px #EF4444;"></span>
            <span>Server Offline (Status {health_resp.status_code})</span>
        </div>
        """, unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: #EF4444; margin-bottom: 20px;">
        <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #EF4444; box-shadow: 0 0 8px #EF4444;"></span>
        <span>Server Unreachable</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Neural Network Info")
st.sidebar.write(
    "A deep bottle-necked Autoencoder captures machine vibrations. "
    "Reconstruction deviations exceeding the threshold flag anomaly alarms."
)

# Fetch Metrics
total_predictions = 0
anomalies_detected = 0
avg_inference_time = 0.0
anomaly_rate = 0.0

if connected:
    try:
        metrics_resp = requests.get(f"{api_url}/metrics", timeout=2)
        if metrics_resp.ok:
            m_data = metrics_resp.json()
            total_predictions = m_data["total_predictions"]
            anomalies_detected = m_data["anomalies_detected"]
            anomaly_rate = m_data["anomaly_rate"] * 100
            avg_inference_time = m_data["avg_inference_time_ms"]
    except Exception:
        pass

# Main Dashboard Layout
main_col1, main_col2 = st.columns([360, 800])

# Left Column: Status, Controls, Telemetry
with main_col1:
    # Machine Status Banner (Injected HTML)
    status_card_container = st.empty()
    
    # Initialize Status Card
    status_card_container.markdown("""
    <div class="status-banner info">
        <div class="status-icon-box">
            <i class="fa-solid fa-microchip"></i>
        </div>
        <div class="status-details">
            <div class="status-label">Machine State</div>
            <div class="status-value">INITIALIZING...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simulation Console Card
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 0;">
        <div class="card-title"><i class="fa-solid fa-sliders" style="color: #3B82F6;"></i> Simulation Console</div>
        <div class="card-desc">Inject vibration sequences into the neural network to verify alarms.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render Streamlit buttons (overridden by custom CSS above)
    trigger_normal = st.button("🟢 Simulate Normal Operation", key="normal_btn")
    trigger_anomaly = st.button("🔴 Simulate Bearing Failure", key="anomaly_btn")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Live Stream simulator switch
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 0; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="font-family: 'Outfit'; font-size: 14px; margin:0;">Live Telemetry Stream</h4>
                <p style="font-size: 11px; color:#6B7280; margin: 2px 0 0 0;">Feed live sensor signals continuously.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Checkbox rendered inside card block
    stream_active = st.checkbox("🛰️ Start Live Telemetry Stream", label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Telemetry Metrics Card
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title"><i class="fa-solid fa-chart-line" style="color: #3B82F6;"></i> Live Telemetry</div>
        <div class="card-desc">Telemetry data polled from the cloud API server.</div>
        <div class="metric-grid-item">
            <span class="metric-title">Total Predictions</span>
            <span class="metric-val">{total_predictions}</span>
        </div>
        <div class="metric-grid-item">
            <span class="metric-title">Anomaly Rate</span>
            <span class="metric-val" style="color: { '#EF4444' if anomaly_rate > 10 else '#10B981' };">{anomaly_rate:.2f}%</span>
        </div>
        <div class="metric-grid-item">
            <span class="metric-title">Avg Latency (Server)</span>
            <span class="metric-val" style="color: #F59E0B;">{avg_inference_time:.2f} ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Right Column: Visual Charts & Overlay Comparison
with main_col2:
    # Waveform Reconstruction Card
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 12px; padding: 20px 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="card-title" style="margin:0;"><i class="fa-solid fa-wave-square" style="color: #3B82F6;"></i> Reconstruction Overlay</div>
                <div class="card-desc" style="margin:2px 0 0 0;">Overlay of original vibration vs model reconstruction.</div>
            </div>
            <div style="font-size: 11px; color: #9CA3AF; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.07); padding: 6px 12px; border-radius: 8px; font-weight: 500; font-family: 'Inter';">
                Threshold Limit: {model_threshold:.5f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    waveform_chart_placeholder = st.empty()
    
    # Error bar card
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 12px; padding: 20px 24px;">
        <div class="card-title" style="margin:0;"><i class="fa-solid fa-scale-unbalanced-flip" style="color: #3B82F6;"></i> Reconstruction Error Analysis</div>
        <div class="card-desc" style="margin:2px 0 0 0;">Sample Mean Squared Error (MSE) compared against the threshold.</div>
    </div>
    """, unsafe_allow_html=True)
    
    error_chart_placeholder = st.empty()


# Plotting Engine (Plotly)
def render_live_visuals(original, reconstructed, mse, threshold):
    is_anomaly = mse >= threshold
    
    # 1. Update Status Banner
    if is_anomaly:
        status_card_container.markdown(f"""
        <div class="status-banner anomaly">
            <div class="status-icon-box">
                <i class="fa-solid fa-circle-exclamation"></i>
            </div>
            <div class="status-details">
                <div class="status-label">Machine State</div>
                <div class="status-value">🚨 CRITICAL FAULT DETECTED</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        status_card_container.markdown(f"""
        <div class="status-banner ok">
            <div class="status-icon-box">
                <i class="fa-solid fa-circle-check"></i>
            </div>
            <div class="status-details">
                <div class="status-label">Machine State</div>
                <div class="status-value">✅ OPERATING NORMALLY</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Plotly Line Chart Overlay (Apple Dark Theme)
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(
        y=original,
        mode='lines',
        name='Original Sensor Vibration',
        line=dict(color='#3B82F6', width=3) # Bright Blue
    ))
    fig_signal.add_trace(go.Scatter(
        y=reconstructed,
        mode='lines',
        name='Neural Net Reconstruction',
        line=dict(color='#F59E0B', width=2.5, dash='dash') # Bright Orange
    ))
    
    fig_signal.update_layout(
        template="plotly_dark",
        plot_bgcolor="#080B11",
        paper_bgcolor="#080B11",
        margin=dict(l=40, r=40, t=10, b=40),
        height=320,
        font=dict(family="Inter", color="#9CA3AF"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", showgrid=True, zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )
    waveform_chart_placeholder.plotly_chart(fig_signal, use_container_width=True)

    # 3. Custom HTML Anomaly Bar Chart (Glassmorphism + Threshold Line)
    max_scale = max(threshold * 2.2, mse * 1.2, 0.03)
    percentage = min((mse / max_scale) * 100, 100)
    marker_percent = min((threshold / max_scale) * 100, 98)
    
    # Select bar color
    if mse >= threshold:
        bar_color = "linear-gradient(90deg, #EF4444, #F87171)" # Danger Red
        safety_text = "<span style='color: #EF4444; font-weight: 800;'>CRITICAL</span>"
    elif mse >= threshold * 0.7:
        bar_color = "linear-gradient(90deg, #F59E0B, #FBBF24)" # Warning Orange
        safety_text = f"<span style='color: #F59E0B; font-weight: 700;'>{(threshold/mse):.1f}x (WARN)</span>"
    else:
        bar_color = "linear-gradient(90deg, #10B981, #34D399)" # Normal Green
        safety_text = f"<span style='color: #10B981; font-weight: 700;'>{(threshold/mse):.1f}x Safety</span>"
        
    error_chart_placeholder.markdown(f"""
    <div style="font-family: 'Inter'; font-size: 13px; color: #9CA3AF; display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span>Current Reconstruction Error (MSE)</span>
        <span style="font-family: 'Outfit'; font-weight: 700; color: #F3F4F6;">{mse:.6f}</span>
    </div>
    <div style="width: 100%; height: 16px; background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; position: relative; overflow: visible; border: 1px solid rgba(255,255,255,0.07); margin-bottom: 24px;">
        <div style="height: 100%; width: {percentage}%; background: {bar_color}; border-radius: 8px; transition: width 0.5s ease-in-out;"></div>
        <div style="position: absolute; top: 0; bottom: 0; left: {marker_percent}%; width: 2px; background-color: #F59E0B; box-shadow: 0 0 8px #F59E0B;">
            <span style="position: absolute; bottom: 20px; transform: translateX(-50%); font-size: 9px; font-weight: 700; text-transform: uppercase; color: #F59E0B; background: #080B11; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.07); white-space: nowrap; font-family: 'Inter'; letter-spacing: 0.5px;">Threshold Limit ({threshold:.5f})</span>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6B7280; margin-top: -16px; margin-bottom: 20px;">
        <span>0.0</span>
        <span>{max_scale:.3f}</span>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; font-family: 'Inter';">
        <div style="background: rgba(0, 0, 0, 0.15); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 4px;">
            <span style="font-size: 10px; color: #6B7280; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Current MSE</span>
            <span style="font-family: 'Outfit'; font-size: 16px; font-weight: 700; color: #F3F4F6;">{mse:.5f}</span>
        </div>
        <div style="background: rgba(0, 0, 0, 0.15); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 4px;">
            <span style="font-size: 10px; color: #6B7280; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Threshold Limit</span>
            <span style="font-family: 'Outfit'; font-size: 16px; font-weight: 700; color: #F59E0B;">{threshold:.5f}</span>
        </div>
        <div style="background: rgba(0, 0, 0, 0.15); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 4px;">
            <span style="font-size: 10px; color: #6B7280; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">State Metric</span>
            <span style="font-family: 'Outfit'; font-size: 16px; font-weight: 700; color: #F3F4F6;">{safety_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Initialize default flat signals on load
if not trigger_normal and not trigger_anomaly and not stream_active:
    dummy_x = Array = [0.0] * 64
    render_live_visuals(dummy_x, dummy_x, 0.0, model_threshold)

# Execution Logic
if trigger_normal or trigger_anomaly:
    if not connected:
        st.sidebar.error("Cannot run simulation: Inference server offline.")
    else:
        anomaly_flag = trigger_anomaly
        signal = generate_sample(anomaly_flag)
        
        try:
            resp = requests.post(f"{api_url}/predict", json={"data": signal}, timeout=5)
            if resp.ok:
                res_data = resp.json()
                render_live_visuals(
                    original=signal,
                    reconstructed=res_data["reconstructed"],
                    mse=res_data["mse"],
                    threshold=res_data["threshold"]
                )
            else:
                st.sidebar.error(f"Inference API failed: {resp.status_code}")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

# Live Stream Loop
if stream_active:
    if not connected:
        st.sidebar.error("Cannot start stream: Inference server offline.")
    else:
        try:
            while stream_active:
                # 20% anomaly chance
                fault = np.random.random() < 0.2
                signal = generate_sample(fault)
                
                resp = requests.post(f"{api_url}/predict", json={"data": signal}, timeout=3)
                if resp.ok:
                    res_data = resp.json()
                    render_live_visuals(
                        original=signal,
                        reconstructed=res_data["reconstructed"],
                        mse=res_data["mse"],
                        threshold=res_data["threshold"]
                    )
                else:
                    st.sidebar.error(f"Stream failed: {resp.status_code}")
                    break
                    
                time.sleep(1.0)
        except Exception as e:
            st.sidebar.error(f"Stream interrupted: {e}")
