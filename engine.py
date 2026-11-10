import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing schema_py_generated from tensorflow site-packages if available
TF_PY_PATH = r'C:\Users\TIYAS DEY\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\lite\python'
if os.path.exists(TF_PY_PATH) and TF_PY_PATH not in sys.path:
    sys.path.append(TF_PY_PATH)

import schema_py_generated as schema_fb

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seizure_model_int8.tflite")

class TinyMLSeizureNet(nn.Module):
    def __init__(self, weights):
        super().__init__()
        # Conv1: 4 in_channels -> 8 out_channels, kernel=15, stride=2, padding=7 (same)
        self.conv1 = nn.Conv1d(4, 8, kernel_size=15, stride=2, padding=7)
        self.conv1.weight.data = torch.from_numpy(weights['w_c1'])
        self.conv1.bias.data = torch.from_numpy(weights['b_c1'])
        
        # Depthwise Conv: 8 in_channels -> 8 out_channels, kernel=7, stride=2, padding=3 (same), groups=8
        self.dw = nn.Conv1d(8, 8, kernel_size=7, stride=2, padding=3, groups=8)
        self.dw.weight.data = torch.from_numpy(weights['w_dw'])
        self.dw.bias.data = torch.from_numpy(weights['b_dw'])
        
        # Pointwise Conv: 8 in_channels -> 16 out_channels, kernel=1, stride=1
        self.pw = nn.Conv1d(8, 16, kernel_size=1, stride=1, padding=0)
        self.pw.weight.data = torch.from_numpy(weights['w_pw'])
        self.pw.bias.data = torch.from_numpy(weights['b_pw'])
        
        # MaxPool1d: pool_size=2, stride=2
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # FC1: 16 -> 12
        self.fc1 = nn.Linear(16, 12)
        self.fc1.weight.data = torch.from_numpy(weights['w_f1'])
        self.fc1.bias.data = torch.from_numpy(weights['b_f1'])
        
        # FC2 (Output): 12 -> 1
        self.fc2 = nn.Linear(12, 1)
        self.fc2.weight.data = torch.from_numpy(weights['w_f2'])
        self.fc2.bias.data = torch.from_numpy(weights['b_f2'])

    def forward(self, x):
        # x is (batch, channels=4, timesteps=128)
        x = F.relu(self.conv1(x))
        x = self.dw(x)
        x = F.relu(self.pw(x))
        x = self.pool(x)
        x = x.mean(dim=2) # Global Average Pooling
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_weights_from_tflite(model_path):
    with open(model_path, 'rb') as f:
        buf = f.read()

    fb_model = schema_fb.Model.GetRootAsModel(buf, 0)
    subgraph = fb_model.Subgraphs(0)

    def get_quant_params(idx):
        t = subgraph.Tensors(idx)
        q = t.Quantization()
        scales = np.array([q.Scale(j) for j in range(q.ScaleLength())], dtype=np.float32) if q and q.ScaleLength() > 0 else np.array([1.0], dtype=np.float32)
        zps = np.array([q.ZeroPoint(j) for j in range(q.ZeroPointLength())], dtype=np.int32) if q and q.ZeroPointLength() > 0 else np.array([0], dtype=np.int32)
        return scales, zps

    def get_tensor_data(idx):
        t = subgraph.Tensors(idx)
        b = fb_model.Buffers(t.Buffer())
        data_len = b.DataLength()
        if data_len == 0:
            return None
        arr = bytearray(data_len)
        for i in range(data_len):
            arr[i] = b.Data(i)
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        if t.Type() == 9:
            return np.frombuffer(arr, dtype=np.int8).reshape(shape)
        elif t.Type() == 2:
            return np.frombuffer(arr, dtype=np.int32).reshape(shape)
        return arr

    # Conv1
    w_c1_raw = get_tensor_data(15)
    s_c1, _ = get_quant_params(15)
    b_c1_raw = get_tensor_data(14)
    s_b_c1, _ = get_quant_params(14)
    w_c1_float = (w_c1_raw.squeeze(1).transpose(0, 2, 1).astype(np.float32)) * s_c1[:, None, None]
    b_c1_float = b_c1_raw.astype(np.float32) * s_b_c1

    # Depthwise
    w_dw_raw = get_tensor_data(13)
    s_dw, _ = get_quant_params(13)
    b_dw_raw = get_tensor_data(12)
    s_b_dw, _ = get_quant_params(12)
    w_dw_float = (w_dw_raw.squeeze(0).squeeze(0).transpose(1, 0)[:, None, :].astype(np.float32)) * s_dw[:, None, None]
    b_dw_float = b_dw_raw.astype(np.float32) * s_b_dw

    # Pointwise
    w_pw_raw = get_tensor_data(11)
    s_pw, _ = get_quant_params(11)
    b_pw_raw = get_tensor_data(10)
    s_b_pw, _ = get_quant_params(10)
    w_pw_float = (w_pw_raw.squeeze(1).transpose(0, 2, 1).astype(np.float32)) * s_pw[:, None, None]
    b_pw_float = b_pw_raw.astype(np.float32) * s_b_pw

    # FC1
    w_f1_raw = get_tensor_data(9)
    s_f1, _ = get_quant_params(9)
    b_f1_raw = get_tensor_data(8)
    s_b_f1, _ = get_quant_params(8)
    w_f1_float = w_f1_raw.astype(np.float32) * s_f1[:, None]
    b_f1_float = b_f1_raw.astype(np.float32) * s_b_f1

    # FC2
    w_f2_raw = get_tensor_data(7)
    s_f2, _ = get_quant_params(7)
    b_f2_raw = get_tensor_data(6)
    s_b_f2, _ = get_quant_params(6)
    w_f2_float = w_f2_raw.astype(np.float32) * s_f2
    b_f2_float = b_f2_raw.astype(np.float32) * s_b_f2

    return {
        'w_c1': w_c1_float, 'b_c1': b_c1_float,
        'w_dw': w_dw_float, 'b_dw': b_dw_float,
        'w_pw': w_pw_float, 'b_pw': b_pw_float,
        'w_f1': w_f1_float, 'b_f1': b_f1_float,
        'w_f2': w_f2_float, 'b_f2': b_f2_float
    }

print("Initializing TinyML Seizure Inference Engine...")
weights_dict = load_weights_from_tflite(MODEL_PATH)
ENGINE_MODEL = TinyMLSeizureNet(weights_dict)
ENGINE_MODEL.eval()
print("TinyML Model Ready for Real-time Inference!")

CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]

# Generate 4 Calibrated Reference Examples (2 Seizure, 2 Non-Seizure)
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
    # data_4x128: (4, 128) sampled at 64 Hz
    fs = 64.0
    freqs = np.fft.rfftfreq(128, 1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(data_4x128, axis=1)) ** 2
    avg_power = np.mean(fft_vals, axis=0) # power across all 4 channels
    
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
    # input_array can be (4, 128) or (128, 4)
    arr = np.array(input_array, dtype=np.float32)
    
    # Shape normalization
    if arr.shape == (128, 4):
        arr = arr.T # to (4, 128)
    elif arr.shape == (1, 128, 4):
        arr = arr[0].T # to (4, 128)
    elif arr.shape == (1, 4, 128):
        arr = arr[0]
    elif arr.ndim == 1:
        if len(arr) == 512:
            arr = arr.reshape(4, 128)
        elif len(arr) == 128:
            # Replicate single channel to 4 channels
            arr = np.tile(arr, (4, 1))
        else:
            raise ValueError(f"Expected 128 or 512 samples, received {len(arr)}")
            
    if arr.shape != (4, 128):
        raise ValueError(f"Invalid sequence shape {arr.shape}. Must be 4 channels x 128 samples.")
        
    # Standardize per channel (Z-score)
    mean_val = np.mean(arr, axis=1, keepdims=True)
    std_val = np.std(arr, axis=1, keepdims=True) + 1e-6
    norm_arr = (arr - mean_val) / std_val
    
    # Model forward pass
    tensor_in = torch.from_numpy(norm_arr).unsqueeze(0) # (1, 4, 128)
    with torch.no_grad():
        prob = ENGINE_MODEL(tensor_in).item()
        
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
