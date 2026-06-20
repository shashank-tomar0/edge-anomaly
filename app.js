// Initialize ANSI colors and setup variables
let apiBaseUrl = "https://edge-anomaly.onrender.com";
let isConnected = false;
let modelThreshold = 0.01412;
let modelSeqLength = 64;
let streamInterval = null;
let metricsInterval = null;
let chartInstance = null;

// DOM Elements
const apiUrlInput = document.getElementById("apiUrlInput");
const connectBtn = document.getElementById("connectBtn");
const connectionStatus = document.getElementById("connectionStatus");
const statusDot = connectionStatus.querySelector(".status-dot");
const statusLabel = connectionStatus.querySelector(".status-label");

const machineStatusCard = document.getElementById("machineStatusCard");
const machineStatusIcon = document.getElementById("machineStatusIcon");
const machineStatusText = document.getElementById("machineStatusText");

const simNormalBtn = document.getElementById("simNormalBtn");
const simAnomalyBtn = document.getElementById("simAnomalyBtn");
const streamToggle = document.getElementById("streamToggle");

const metricTotalPreds = document.getElementById("metricTotalPreds");
const metricAnomalyRate = document.getElementById("metricAnomalyRate");
const metricLatency = document.getElementById("metricLatency");
const modelMetaInfo = document.getElementById("modelMetaInfo");

const currentMseVal = document.getElementById("currentMseVal");
const errorProgressBar = document.getElementById("errorProgressBar");
const thresholdLineMarker = document.getElementById("thresholdLineMarker");
const errorMaxScale = document.getElementById("errorMaxScale");

const statsCurrentMse = document.getElementById("statsCurrentMse");
const statsThreshold = document.getElementById("statsThreshold");
const statsSafetyFactor = document.getElementById("statsSafetyFactor");

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('waveformChart').getContext('2d');
    
    // Create zero-filled initial datasets
    const initialLabels = Array.from({length: 64}, (_, i) => i);
    const initialData = Array(64).fill(0);
    
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: initialLabels,
            datasets: [
                {
                    label: 'Original Sensor Signal',
                    data: initialData,
                    borderColor: '#3B82F6', // Blue
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.2
                },
                {
                    label: 'Model Reconstructed Signal',
                    data: initialData,
                    borderColor: '#F59E0B', // Orange
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#9CA3AF',
                        font: {
                            family: 'Inter',
                            size: 12
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#111622',
                    titleColor: '#F3F4F6',
                    bodyColor: '#9CA3AF',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)',
                        drawTicks: false
                    },
                    ticks: {
                        color: '#6B7280',
                        font: { size: 10 }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)',
                        drawTicks: false
                    },
                    ticks: {
                        color: '#6B7280',
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}

// Generate vibration waveform signal
function generateLocalSignal(anomaly = false) {
    const signal = [];
    const steps = 64;
    for (let i = 0; i < steps; i++) {
        const t = (i / (steps - 1)) * 10;
        
        // Random normal noise approximation (Box-Muller transform)
        const u1 = Math.random();
        const u2 = Math.random();
        const noise = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        
        if (!anomaly) {
            // Normal machine vibration: sin(2*pi*1*t) + 0.5*sin(2*pi*2.5*t) + noise * 0.1
            const val = Math.sin(2.0 * Math.PI * 1.0 * t) + 0.5 * Math.sin(2.0 * Math.PI * 2.5 * t) + noise * 0.1;
            signal.push(val);
        } else {
            // Anomalous vibration: 2.0*sin(2*pi*1.5*t) + 1.0*sin(2*pi*4.0*t) + noise * 0.5
            const val = 2.0 * Math.sin(2.0 * Math.PI * 1.5 * t) + 1.0 * Math.sin(2.0 * Math.PI * 4.0 * t) + noise * 0.5;
            signal.push(val);
        }
    }
    return signal;
}

// Update API Server connection status
async function checkBackendConnection() {
    apiBaseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
    
    // Set status UI to connecting
    statusDot.className = "status-dot connecting";
    statusLabel.textContent = "Connecting...";
    
    try {
        const healthResp = await fetch(`${apiBaseUrl}/health`, { method: 'GET' });
        
        if (healthResp.ok) {
            const health = await healthResp.json();
            
            // Success
            isConnected = true;
            statusDot.className = "status-dot online";
            statusLabel.textContent = `Online (${health.model_loaded})`;
            
            // Get threshold
            modelThreshold = health.threshold;
            statsThreshold.textContent = modelThreshold.toFixed(5);
            
            // Get Info
            const infoResp = await fetch(`${apiBaseUrl}/info`);
            if (infoResp.ok) {
                const info = await infoResp.json();
                modelSeqLength = info.parameters.sequence_length;
                modelMetaInfo.textContent = `Threshold: ${modelThreshold.toFixed(5)} | Latent bottleneck: ${info.parameters.latent_dim}`;
            }
            
            // Fetch initial metrics
            fetchMetrics();
            
            // Start periodic metrics polling
            startMetricsPolling();
            return true;
        }
    } catch (err) {
        console.error("Connection check failed:", err);
    }
    
    // Fail
    isConnected = false;
    statusDot.className = "status-dot offline";
    statusLabel.textContent = "Offline";
    modelMetaInfo.textContent = "Threshold: -- | Percentile: --";
    stopMetricsPolling();
    stopStream();
    return false;
}

// Fetch Metrics from Server
async function fetchMetrics() {
    if (!isConnected) return;
    try {
        const resp = await fetch(`${apiBaseUrl}/metrics`);
        if (resp.ok) {
            const data = await resp.json();
            metricTotalPreds.textContent = data.total_predictions;
            metricAnomalyRate.textContent = `${(data.anomaly_rate * 100).toFixed(1)}%`;
            metricLatency.textContent = `${data.avg_inference_time_ms.toFixed(2)} ms`;
        }
    } catch (err) {
        console.error("Failed to fetch server metrics:", err);
    }
}

function startMetricsPolling() {
    stopMetricsPolling();
    metricsInterval = setInterval(fetchMetrics, 3000);
}

function stopMetricsPolling() {
    if (metricsInterval) {
        clearInterval(metricsInterval);
        metricsInterval = null;
    }
}

// Prediction Logic
async function runPrediction(isAnomaly) {
    if (!isConnected) {
        alert("Inference server offline. Please connect to a running backend first.");
        return;
    }
    
    const signal = generateLocalSignal(isAnomaly);
    
    try {
        const start = performance.now();
        const resp = await fetch(`${apiBaseUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: signal })
        });
        const latency = performance.now() - start;
        
        if (resp.ok) {
            const result = await resp.json();
            updateUI(signal, result, latency);
        } else {
            console.error("Prediction API failed:", resp.status, await resp.text());
        }
    } catch (err) {
        console.error("Inference network error:", err);
    }
}

// Update Dashboard Visuals
function updateUI(original, result, networkLatency) {
    const mse = result.mse;
    const isAnomaly = result.is_anomaly;
    const threshold = result.threshold;
    const reconstructed = result.reconstructed;
    
    // 1. Update Waveform Line Chart
    chartInstance.data.datasets[0].data = original;
    chartInstance.data.datasets[1].data = reconstructed;
    chartInstance.update('active'); // Animate transition
    
    // 2. Update Machine State Card
    if (isAnomaly) {
        machineStatusCard.className = "card status-card anomaly";
        machineStatusIcon.className = "fa-solid fa-circle-exclamation status-icon";
        machineStatusText.textContent = "CRITICAL FAULT";
    } else {
        machineStatusCard.className = "card status-card ok";
        machineStatusIcon.className = "fa-solid fa-circle-check status-icon";
        machineStatusText.textContent = "NORMAL STATE";
    }
    
    // 3. Update Anomaly Error Progress Bar
    // Dynamically calculate max scale of progress bar
    const maxScale = Math.max(threshold * 2.2, mse * 1.2, 0.03);
    errorMaxScale.textContent = maxScale.toFixed(3);
    
    const percentage = Math.min((mse / maxScale) * 100, 100);
    errorProgressBar.style.width = `${percentage}%`;
    
    // Color thresholds
    if (mse >= threshold) {
        errorProgressBar.className = "progress-bar danger"; // Red
    } else if (mse >= threshold * 0.7) {
        errorProgressBar.className = "progress-bar warning"; // Orange
    } else {
        errorProgressBar.className = "progress-bar"; // Green
    }
    
    // Position Threshold marker
    const markerPercent = Math.min((threshold / maxScale) * 100, 98);
    thresholdLineMarker.style.left = `${markerPercent}%`;
    thresholdLineMarker.style.display = "block";
    
    // 4. Update Stats Box values
    statsCurrentMse.textContent = mse.toFixed(5);
    currentMseVal.textContent = mse.toFixed(5);
    
    if (mse > 0) {
        const factor = threshold / mse;
        if (factor >= 1) {
            statsSafetyFactor.textContent = `${factor.toFixed(1)}x`;
            statsSafetyFactor.className = "stat-val text-green";
        } else {
            statsSafetyFactor.textContent = "FAULT";
            statsSafetyFactor.className = "stat-val text-red";
        }
    } else {
        statsSafetyFactor.textContent = "--";
        statsSafetyFactor.className = "stat-val";
    }
}

// Live Stream Simulation
function startStream() {
    if (streamInterval) return;
    
    streamInterval = setInterval(() => {
        // 20% chance of anomaly
        const fault = Math.random() < 0.2;
        runPrediction(fault);
    }, 1000);
}

function stopStream() {
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
    }
    streamToggle.checked = false;
}

// Event Listeners
connectBtn.addEventListener("click", checkBackendConnection);
simNormalBtn.addEventListener("click", () => runPrediction(false));
simAnomalyBtn.addEventListener("click", () => runPrediction(true));

streamToggle.addEventListener("change", (e) => {
    if (e.target.checked) {
        if (!isConnected) {
            alert("Please connect to the backend first.");
            e.target.checked = false;
            return;
        }
        startStream();
    } else {
        stopStream();
    }
});

// Detect Enter key on URL input
apiUrlInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        checkBackendConnection();
    }
});

// Initialize on Load
window.addEventListener("load", () => {
    initChart();
    checkBackendConnection();
});
