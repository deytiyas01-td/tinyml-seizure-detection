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
    """
    Exact forward pass of the TinyML 1D-CNN using NumPy:
    1. Conv1D: 4 channels -> 8 filters, kernel=15, stride=2, SAME pad (pad 7 on both sides)
    2. Depthwise Conv1D: 8 filters, kernel=7, stride=2, SAME pad (pad 3)
    3. Pointwise Conv1D: 8 -> 16 filters, kernel=1, stride=1
    4. MaxPool1D: pool_size=2, stride=2
    5. Global Average Pooling: 16 values
    6. Dense 1: 16 -> 12, ReLU
    7. Dense 2: 12 -> 1, Sigmoid
    """
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
    """
    Generates 4 synthetic demonstration signals clearly labeled for UI/API verification.
    These are mathematical demonstrations, NOT presented as real CHB-MIT patient recordings.
    """
    t = np.linspace(0, 2.0, 128)
    
    # 1. Synthetic Seizure Pattern 1: Simulated 3.0 Hz spike-and-wave rhythm
    s1 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 2.5 * np.sin(2 * np.pi * 3.0 * t + ch*0.2) + 1.5 * np.cos(2 * np.pi * 6.0 * t) + 1.0 * np.sin(2 * np.pi * 9.0 * t)
        s1[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 2. Synthetic Seizure Pattern 2: Simulated 4.0 Hz rhythmic epileptiform paroxysm
    s2 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 3.0 * np.sign(np.sin(2 * np.pi * 4.0 * t + ch*0.3)) + 1.2 * np.sin(2 * np.pi * 8.0 * t)
        s2[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 3. Synthetic Non-Seizure Pattern 1: Simulated 10.0 Hz resting alpha background
    ns1 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 0.8 * np.sin(2 * np.pi * 10.0 * t + ch*0.5) + 0.2 * np.sin(2 * np.pi * 20.0 * t) + 0.1 * np.random.randn(128)
        ns1[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 4. Synthetic Non-Seizure Pattern 2: Simulated slow drift baseline without paroxysms
    ns2 = np.zeros((4, 128), dtype=np.float32)
    for ch in range(4):
        w = 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.4 * np.random.randn(128)
        ns2[ch] = (w - np.mean(w)) / (np.std(w) + 1e-6)

    examples = [
        {
            "id": "synthetic_seizure_1",
            "name": "Synthetic Pattern 1: 3 Hz Spike-and-Wave (Simulated)",
            "type": "synthetic_seizure",
            "is_synthetic": True,
            "description": "Synthetic demonstration waveform (3.0 Hz rhythmic spike-and-wave) for API/UI layout verification. Not a patient recording.",
            "expected": "Seizure",
            "channels": CHANNELS,
            "data": s1.tolist()
        },
        {
            "id": "synthetic_seizure_2",
            "name": "Synthetic Pattern 2: 4 Hz Rhythmic Discharge (Simulated)",
            "type": "synthetic_seizure",
            "is_synthetic": True,
            "description": "Synthetic demonstration waveform (4.0 Hz rhythmic square-wave burst) for API/UI layout verification. Not a patient recording.",
            "expected": "Seizure",
            "channels": CHANNELS,
            "data": s2.tolist()
        },
        {
            "id": "synthetic_non_seizure_1",
            "name": "Synthetic Pattern 3: 10 Hz Alpha Rhythm (Simulated)",
            "type": "synthetic_non_seizure",
            "is_synthetic": True,
            "description": "Synthetic demonstration waveform (10.0 Hz physiological alpha background) for API/UI layout verification. Not a patient recording.",
            "expected": "Non-Seizure",
            "channels": CHANNELS,
            "data": ns1.tolist()
        },
        {
            "id": "synthetic_non_seizure_2",
            "name": "Synthetic Pattern 4: Slow Baseline Drift (Simulated)",
            "type": "synthetic_non_seizure",
            "is_synthetic": True,
            "description": "Synthetic demonstration waveform (slow 1.5 Hz baseline drift) for API/UI layout verification. Not a patient recording.",
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

def run_inference(input_array, already_normalized=False):
    """
    Enforces strict 4-channel input requirement (shape (4, 128) or (128, 4), total 512 numbers).
    The 128-number single-channel duplicate shortcut has been removed per review recommendation.
    Channel ordering: Channel 0: FP1-F7, Channel 1: F7-T7, Channel 2: T7-P7, Channel 3: P7-O1.
    """
    arr = np.array(input_array, dtype=np.float32)
    
    # Reject 128-sample single channel shortcut explicitly
    if arr.ndim == 1 and len(arr) == 128:
        raise ValueError(
            "Single-channel input (128 values) is not permitted. "
            "The model requires four independent channels (FP1-F7, F7-T7, T7-P7, P7-O1) "
            "with 128 samples each (total 512 values in a 4x128 matrix or 512-element array)."
        )

    # Handle shape conversions
    if arr.shape == (128, 4):
        arr = arr.T  # to (4, 128)
    elif arr.shape == (1, 128, 4):
        arr = arr[0].T
    elif arr.shape == (1, 4, 128):
        arr = arr[0]
    elif arr.ndim == 1 and len(arr) == 512:
        arr = arr.reshape(4, 128)

    if arr.shape != (4, 128):
        raise ValueError(
            f"Invalid sequence shape {arr.shape}. "
            f"Expected 4 channels x 128 samples (total 512 values) in fixed channel order: "
            f"{', '.join(CHANNELS)}."
        )

    # Apply standardization per channel (matching notebook preprocessing)
    mean_val = np.mean(arr, axis=1, keepdims=True)
    std_val = np.std(arr, axis=1, keepdims=True)
    
    # Check if the signal is already standardized at recording level (|mean| < 0.6 and std in [0.4, 4.0])
    # Re-normalizing an isolated seizure window squashes the epileptiform amplitude burst.
    is_approx_normalized = (np.all(np.abs(mean_val) < 0.6) and np.all((std_val > 0.4) & (std_val < 4.0)))

    if already_normalized or is_approx_normalized:
        norm_arr = arr
    else:
        norm_arr = (arr - mean_val) / (std_val + 1e-6)

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
        "risk_level": "Elevated" if is_seizure else "Normal",
        "frequency_bands": bands,
        "channel_stats": channel_stats,
        "waveforms": arr.tolist(),
        "channels": CHANNELS
    }

CURRENT_EDF_CACHE = {
    "windows": None,
    "num_windows": 0
}

def get_cached_window(idx):
    """Retrieve raw 4x128 array for window index on demand."""
    if CURRENT_EDF_CACHE["windows"] is None:
        return None
    if 0 <= idx < CURRENT_EDF_CACHE["num_windows"]:
        win = CURRENT_EDF_CACHE["windows"][idx]
        return np.round(win, 3).tolist()
    return None

def evaluate_edf_stream(file_obj_or_path):
    """
    High-level pipeline for real EDF files:
    1. Reads EDF, verifies all 4 channels (FP1-F7, F7-T7, T7-P7, P7-O1).
    2. Applies notebook preprocessing (0.5-30 Hz bandpass, resample to 64 Hz, recording standardization).
    3. Windows into 2-second segments.
    4. Evaluates each window with the model.
    """
    import edf_parser
    edf_res = edf_parser.read_edf(file_obj_or_path, target_channels=CHANNELS)
    prep_res = edf_parser.preprocess_and_window_eeg(edf_res["signals"], edf_res["source_fs"])

    windows = prep_res["windows"] # (num_windows, 4, 128)
    CURRENT_EDF_CACHE["windows"] = windows
    CURRENT_EDF_CACHE["num_windows"] = len(windows)

    window_results = []
    seizure_count = 0

    for w_idx in range(len(windows)):
        win = windows[w_idx]
        res = run_inference(win, already_normalized=True)
        t_start = round(w_idx * prep_res["window_duration"], 1)
        t_end = round((w_idx + 1) * prep_res["window_duration"], 1)
        if res["is_seizure"]:
            seizure_count += 1
        window_results.append({
            "window_index": w_idx,
            "time_range": f"{t_start}s - {t_end}s",
            "prediction": res["prediction"],
            "confidence_percentage": res["confidence_percentage"],
            "probability": res["probability"],
            "is_seizure": res["is_seizure"]
        })

    # Clinical Temporal Cluster Filter (ACNS Standard):
    # An electrographic seizure must demonstrate sustained rhythmic discharge lasting >= 8 seconds
    # (i.e. at least 4 consecutive 2-second windows with elevated probability >= 0.65).
    # Brief 2-4 second transients (e.g. sleep spindles, vertex sharp waves, movement artifacts) are rejected.
    seizure_clusters = []
    current_cluster = []
    MIN_SEIZURE_WINDOWS = 4  # 4 * 2.0s = 8.0s minimum sustained epileptiform discharge
    
    for w in window_results:
        if w["probability"] >= 0.65:
            current_cluster.append(w)
        else:
            if len(current_cluster) >= MIN_SEIZURE_WINDOWS:
                seizure_clusters.append(current_cluster)
            current_cluster = []
    if len(current_cluster) >= MIN_SEIZURE_WINDOWS:
        seizure_clusters.append(current_cluster)

    confirmed_indices = set()
    cluster_summaries = []
    for cl in seizure_clusters:
        for cw in cl:
            confirmed_indices.add(cw["window_index"])
        cluster_summaries.append({
            "start_time": cl[0]["time_range"].split(" - ")[0],
            "end_time": cl[-1]["time_range"].split(" - ")[1],
            "duration_sec": len(cl) * 2.0,
            "peak_probability": round(max(cw["probability"] for cw in cl), 4),
            "num_windows": len(cl)
        })

    for w in window_results:
        w["is_confirmed_seizure"] = w["window_index"] in confirmed_indices

    seizure_detected_overall = len(seizure_clusters) > 0
    confirmed_seizure_windows_count = len(confirmed_indices)

    # Optimized window waveform payload:
    # If small file (<= 100 windows), pre-package all.
    # If large file (> 100 windows), pre-package all confirmed seizures + first 40 normal windows
    # Remaining windows are retrieved via /api/edf_window on demand in ~50ms.
    windows_data_dict = {}
    if len(windows) <= 100:
        for idx in range(len(windows)):
            windows_data_dict[str(idx)] = np.round(windows[idx], 3).tolist()
    else:
        for w in window_results:
            if w["is_confirmed_seizure"] or w["is_seizure"]:
                idx = w["window_index"]
                windows_data_dict[str(idx)] = np.round(windows[idx], 3).tolist()
        for idx in range(min(40, len(windows))):
            s_idx = str(idx)
            if s_idx not in windows_data_dict:
                windows_data_dict[s_idx] = np.round(windows[idx], 3).tolist()

    return {
        "duration_sec": round(edf_res["duration_sec"], 2),
        "source_fs": edf_res["source_fs"],
        "target_fs": 64,
        "channel_mapping": edf_res["channel_mapping"],
        "num_windows": prep_res["num_windows"],
        "seizure_windows_count": confirmed_seizure_windows_count,
        "raw_candidate_windows_count": seizure_count,
        "non_seizure_windows_count": prep_res["num_windows"] - confirmed_seizure_windows_count,
        "seizure_detected_overall": seizure_detected_overall,
        "seizure_clusters": cluster_summaries,
        "windows_summary": window_results,
        "all_windows_data": windows_data_dict
    }
