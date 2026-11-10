import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.npz")

# Load pre-extracted weights (Fast, lightweight, zero TensorFlow dependency)
weights_data = np.load(WEIGHTS_PATH)
W_C1 = weights_data['w_c1'] # (8, 4, 15)
B_C1 = weights_data['b_c1'] # (8,)
W_DW = weights_data['w_dw'] # (8, 7)
B_DW = weights_data['b_dw'] # (8,)
W_PW = weights_data['w_pw'] # (16, 8)
B_PW = weights_data['b_pw'] # (16,)
W_F1 = weights_data['w_f1'] # (12, 16)
B_F1 = weights_data['b_f1'] # (12,)
W_F2 = weights_data['w_f2'] # (1, 12)
B_F2 = weights_data['b_f2'] # (1,)

CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]

def forward_pass_numpy(x_norm_4x128):
    # 1. Conv1D with SAME padding (pad=7 on both sides of timesteps)
    x_pad = np.pad(x_norm_4x128, ((0,0), (7, 7)), mode='constant')
    y1 = np.zeros((8, 64), dtype=np.float32)
    for t in range(64):
        win = x_pad[:, t*2 : t*2 + 15]
        y1[:, t] = np.sum(W_C1 * win[None, :, :], axis=(1, 2)) + B_C1
    y1 = np.maximum(0, y1)

    # 2. Depthwise Conv1D with SAME padding (pad=3)
    y1_pad = np.pad(y1, ((0,0), (3, 3)), mode='constant')
    y2 = np.zeros((8, 32), dtype=np.float32)
    for t in range(32):
        win = y1_pad[:, t*2 : t*2 + 7]
        y2[:, t] = np.sum(W_DW * win, axis=1) + B_DW

    # 3. Pointwise Conv1D (16, 8)
    y3 = W_PW @ y2 + B_PW[:, None]
    y3 = np.maximum(0, y3)

    # 4. MaxPool1D (size=2, stride=2)
    y4 = np.maximum(y3[:, 0::2], y3[:, 1::2])

    # 5. Global Average Pooling
    y5 = np.mean(y4, axis=1)

    # 6. Dense 1
    y6 = np.maximum(0, W_F1 @ y5 + B_F1)

    # 7. Dense 2 + Sigmoid
    z = (W_F2 @ y6 + B_F2)[0]
    prob = 1.0 / (1.0 + np.exp(-z))
    return float(prob)

def build_reference_examples():
    t = np.linspace(0, 2.0, 128)
    
    # 1. Seizure 1: High-amplitude 3 Hz Spike-and-Wave Paroxysm
    s1 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 2.5 * np.sin(2 * np.pi * 3.0 * t + ch*0.2) + 1.5 * np.cos(2 * np.pi * 6.0 * t) + 1.0 * np.sin(2 * np.pi * 9.0 * t)
        s1[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 2. Seizure 2: Rhythmic 4 Hz Epileptiform Discharge
    s2 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 3.0 * np.sign(np.sin(2 * np.pi * 4.0 * t + ch*0.3)) + 1.2 * np.sin(2 * np.pi * 8.0 * t)
        s2[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 3. Non-Seizure 1: Normal 10 Hz Alpha Rhythm Background
    ns1 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 0.8 * np.sin(2 * np.pi * 10.0 * t + ch*0.5) + 0.2 * np.sin(2 * np.pi * 20.0 * t) + 0.1 * np.random.randn(128)
        ns1[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 4. Non-Seizure 2: Normal Drowsy Baseline with Slow Delta
    ns2 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.4 * np.random.randn(128)
        ns2[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    examples = [
        {
            "id": "seizure_1",
            "name": "Seizure Sample 1 (Spike-and-Wave)",
            "type": "seizure",
            "description": "3.0 Hz high-amplitude sharp-and-slow wave paroxysm across frontal and temporal electrodes.",
            "expected": "Seizure",
            "channels": CHANNELS,
            "data": s1.tolist()
        },
        {
            "id": "seizure_2",
            "name": "Seizure Sample 2 (Rhythmic Discharge)",
            "type": "seizure",
            "description": "4.0 Hz rhythmic epileptiform discharge with phase reversal in temporal channels.",
            "expected": "Seizure",
            "channels": CHANNELS,
            "data": s2.tolist()
        },
        {
            "id": "non_seizure_1",
            "name": "Non-Seizure Sample 1 (Alpha Rhythm)",
            "type": "non_seizure",
            "description": "Normal 10.0 Hz resting alpha background with stationary variance.",
            "expected": "Non-Seizure",
            "channels": CHANNELS,
            "data": ns1.tolist()
        },
        {
            "id": "non_seizure_2",
            "name": "Non-Seizure Sample 2 (Baseline Drift)",
            "type": "non_seizure",
            "description": "Normal drowsy baseline rhythm with slow drift and no epileptiform transients.",
            "expected": "Non-Seizure",
            "channels": CHANNELS,
            "data": ns2.tolist()
        }
    ]
    return examples

REFERENCE_EXAMPLES = build_reference_examples()

def compute_spectral_bands(data_4x128):
    fs = 64.0
    freqs = np.fft.rfftfreq(128, 1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(data_4x128, axis=1)) ** 2
    avg_power = np.mean(fft_vals, axis=0)
    
    delta_mask = (freqs >= 0.5) & (freqs < 4.0)
    theta_mask = (freqs >= 4.0) & (freqs < 8.0)
    alpha_mask = (freqs >= 8.0) & (freqs < 13.0)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
    
    total_power = np.sum(avg_power) + 1e-9
    
    return {
        "delta": round(float(np.sum(avg_power[delta_mask]) / total_power * 100), 1),
        "theta": round(float(np.sum(avg_power[theta_mask]) / total_power * 100), 1),
        "alpha": round(float(np.sum(avg_power[alpha_mask]) / total_power * 100), 1),
        "beta": round(float(np.sum(avg_power[beta_mask]) / total_power * 100), 1)
    }

def run_inference(input_array):
    arr = np.array(input_array, dtype=np.float32)
    
    if arr.shape == (128, 4):
        arr = arr.T
    elif arr.shape == (1, 128, 4):
        arr = arr[0].T
    elif arr.shape == (1, 4, 128):
        arr = arr[0]
    elif arr.ndim == 1:
        if len(arr) == 512:
            arr = arr.reshape(4, 128)
        elif len(arr) == 128:
            arr = np.tile(arr, (4, 1))
        else:
            raise ValueError(f"Expected 128 or 512 samples, received {len(arr)}")
            
    if arr.shape != (4, 128):
        raise ValueError(f"Invalid sequence shape {arr.shape}. Must be 4 channels x 128 samples.")
        
    mean_val = np.mean(arr, axis=1, keepdims=True)
    std_val = np.std(arr, axis=1, keepdims=True) + 1e-6
    norm_arr = (arr - mean_val) / std_val
    
    prob = forward_pass_numpy(norm_arr)
    prob = float(np.clip(prob, 0.0, 1.0))
    is_seizure = prob >= 0.50
    confidence = (prob if is_seizure else (1.0 - prob)) * 100.0
    
    bands = compute_spectral_bands(norm_arr)
    
    channel_stats = []
    for i, ch_name in enumerate(CHANNELS):
        channel_stats.append({
            "channel": ch_name,
            "mean": round(float(mean_val[i, 0]), 3),
            "std": round(float(std_val[i, 0]), 3),
            "peak_to_peak": round(float(np.ptp(arr[i])), 3)
        })
        
    return {
        "is_seizure": is_seizure,
        "prediction": "Seizure" if is_seizure else "Non-Seizure",
        "probability": round(prob, 4),
        "confidence_percentage": round(confidence, 2),
        "risk_level": "High" if is_seizure else "Normal",
        "frequency_bands": bands,
        "channel_stats": channel_stats,
        "waveforms": arr.tolist(),
        "channels": CHANNELS
    }
