import os
import struct
import numpy as np
from scipy.signal import butter, filtfilt, resample, resample_poly

TARGET_CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1']
TARGET_FS = 64
WINDOW_SEC = 2.0
WINDOW_SIZE = int(TARGET_FS * WINDOW_SEC)  # 128 samples

def parse_edf_header(f):
    """
    Parses the standard 256-byte EDF main header and signal headers from a binary file object.
    """
    f.seek(0)
    main_hdr = f.read(256)
    if len(main_hdr) < 256:
        raise ValueError("Invalid file: File is smaller than standard 256-byte EDF header.")

    version = main_hdr[0:8].decode('latin1').strip()
    patient_id = main_hdr[8:88].decode('latin1').strip()
    recording_id = main_hdr[88:168].decode('latin1').strip()
    start_date = main_hdr[168:176].decode('latin1').strip()
    start_time = main_hdr[176:184].decode('latin1').strip()
    header_bytes = int(main_hdr[184:192].decode('latin1').strip())
    num_records = int(main_hdr[236:244].decode('latin1').strip())
    record_duration = float(main_hdr[244:252].decode('latin1').strip())
    num_signals = int(main_hdr[252:256].decode('latin1').strip())

    # Read signal headers (num_signals * 256 bytes)
    sig_hdr_bytes = f.read(num_signals * 256)
    if len(sig_hdr_bytes) < num_signals * 256:
        raise ValueError("Invalid file: Incomplete signal headers.")

    def read_sig_field(offset, length, count):
        items = []
        for i in range(count):
            start = offset + i * length
            items.append(sig_hdr_bytes[start:start + length].decode('latin1').strip())
        return items

    def to_float(val, default=0.0):
        try:
            return float(val.strip())
        except (ValueError, AttributeError):
            return default

    def to_int(val, default=0):
        try:
            return int(val.strip())
        except (ValueError, AttributeError):
            return default

    curr_pos = 0
    labels = read_sig_field(curr_pos, 16, num_signals); curr_pos += 16 * num_signals
    transducers = read_sig_field(curr_pos, 80, num_signals); curr_pos += 80 * num_signals
    physical_dims = read_sig_field(curr_pos, 8, num_signals); curr_pos += 8 * num_signals
    phys_mins = [to_float(x, -500.0) for x in read_sig_field(curr_pos, 8, num_signals)]; curr_pos += 8 * num_signals
    phys_maxs = [to_float(x, 500.0) for x in read_sig_field(curr_pos, 8, num_signals)]; curr_pos += 8 * num_signals
    dig_mins = [to_float(x, -32768.0) for x in read_sig_field(curr_pos, 8, num_signals)]; curr_pos += 8 * num_signals
    dig_maxs = [to_float(x, 32767.0) for x in read_sig_field(curr_pos, 8, num_signals)]; curr_pos += 8 * num_signals
    prefilterings = read_sig_field(curr_pos, 32, num_signals); curr_pos += 32 * num_signals
    samples_per_record = [to_int(x, 256) for x in read_sig_field(curr_pos, 8, num_signals)]; curr_pos += 8 * num_signals

    return {
        "version": version,
        "patient_id": patient_id,
        "recording_id": recording_id,
        "start_date": start_date,
        "start_time": start_time,
        "header_bytes": header_bytes,
        "num_records": num_records,
        "record_duration": record_duration,
        "num_signals": num_signals,
        "labels": labels,
        "physical_dims": physical_dims,
        "phys_mins": phys_mins,
        "phys_maxs": phys_maxs,
        "dig_mins": dig_mins,
        "dig_maxs": dig_maxs,
        "samples_per_record": samples_per_record
    }

def clean_channel_name(ch_name):
    """
    Standardizes channel names: strips 'EEG ', '.', spaces, '-REF', and uppercases.
    e.g. 'EEG FP1-F7.' -> 'FP1-F7'
    """
    cleaned = ch_name.strip()
    if cleaned.upper().startswith("EEG "):
        cleaned = cleaned[4:].strip()
    cleaned = cleaned.replace(".", "").replace("-REF", "").replace("-ref", "").strip().upper()
    return cleaned

def read_edf(file_obj_or_path, target_channels=TARGET_CHANNELS):
    """
    Reads an EDF file, validates required bipolar channels, and extracts raw signals.
    """
    if isinstance(file_obj_or_path, str):
        f = open(file_obj_or_path, 'rb')
        should_close = True
    else:
        f = file_obj_or_path
        should_close = False

    try:
        hdr = parse_edf_header(f)
        cleaned_labels = [clean_channel_name(lbl) for lbl in hdr["labels"]]

        # Find target channels
        channel_indices = {}
        missing_channels = []
        for target in target_channels:
            matched = False
            for idx, lbl in enumerate(cleaned_labels):
                if lbl == target.upper():
                    channel_indices[target] = idx
                    matched = True
                    break
            if not matched:
                missing_channels.append(target)

        # STRICT VALIDATION: Do not substitute channels silently
        if missing_channels:
            raise ValueError(
                f"Missing required channel(s): {', '.join(missing_channels)}. "
                f"Required channels are: {', '.join(target_channels)}. "
                f"Available channels in this file ({hdr['num_signals']}): {', '.join(cleaned_labels[:12])}..."
            )

        # Sampling rate of target channels (assuming equal)
        first_target_idx = channel_indices[target_channels[0]]
        samples_per_rec = hdr["samples_per_record"][first_target_idx]
        record_dur = hdr["record_duration"]
        source_fs = samples_per_rec / record_dur

        # Calculate scales for converting digital to physical (uV)
        gains = []
        offsets = []
        for ch in target_channels:
            idx = channel_indices[ch]
            d_min = hdr["dig_mins"][idx]
            d_max = hdr["dig_maxs"][idx]
            p_min = hdr["phys_mins"][idx]
            p_max = hdr["phys_maxs"][idx]
            gain = (p_max - p_min) / (d_max - d_min) if d_max != d_min else 1.0
            offset = p_min - gain * d_min
            gains.append(gain)
            offsets.append(offset)

        # Read all data records as int16
        f.seek(hdr["header_bytes"])
        num_records = hdr["num_records"]
        total_samples_per_record = sum(hdr["samples_per_record"])

        # Record offsets within each record for all signals
        sig_offsets = [0]
        for cnt in hdr["samples_per_record"][:-1]:
            sig_offsets.append(sig_offsets[-1] + cnt)

        # Allocate arrays for target channels
        total_target_samples = num_records * samples_per_rec
        signals = np.zeros((len(target_channels), total_target_samples), dtype=np.float32)

        # Read record by record
        record_bytes_len = total_samples_per_record * 2
        for r in range(num_records):
            rec_buf = f.read(record_bytes_len)
            if len(rec_buf) < record_bytes_len:
                break
            rec_int16 = np.frombuffer(rec_buf, dtype='<i2')
            
            for i, ch in enumerate(target_channels):
                idx = channel_indices[ch]
                start = sig_offsets[idx]
                count = hdr["samples_per_record"][idx]
                raw_chunk = rec_int16[start:start + count].astype(np.float32)
                # Convert digital to physical (uV)
                phys_chunk = raw_chunk * gains[i] + offsets[i]
                signals[i, r * samples_per_rec : r * samples_per_rec + count] = phys_chunk

        return {
            "signals": signals,
            "source_fs": source_fs,
            "duration_sec": signals.shape[1] / source_fs,
            "channel_mapping": {ch: cleaned_labels[channel_indices[ch]] for ch in target_channels},
            "header": hdr
        }
    finally:
        if should_close:
            f.close()

def preprocess_and_window_eeg(raw_signals_4xN, source_fs, target_fs=TARGET_FS, window_sec=WINDOW_SEC):
    """
    Applies the EXACT training notebook preprocessing:
    1. Bandpass filter: 0.5 to 30.0 Hz (Butterworth/FIR)
    2. Resample from source_fs to 64 Hz
    3. Recording-level channel standardization: (data - mean) / (std + 1e-6)
    4. Slice into consecutive 2-second windows (128 samples per channel).
    """
    # 1. Bandpass filter 0.5 - 30.0 Hz
    nyq = 0.5 * source_fs
    low = max(0.5 / nyq, 0.001)
    high = min(30.0 / nyq, 0.999)
    b, a = butter(4, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, raw_signals_4xN, axis=1)

    # 2. Resample to 64 Hz
    # If source_fs is 256 Hz (standard CHB-MIT), downsample by 4 (256 / 4 = 64) with resample_poly for 100x speedup
    if int(round(source_fs)) == 256 and target_fs == 64:
        resampled = resample_poly(filtered, 1, 4, axis=1).astype(np.float32)
    elif int(round(source_fs)) == 128 and target_fs == 64:
        resampled = resample_poly(filtered, 1, 2, axis=1).astype(np.float32)
    else:
        num_target_samples = int(filtered.shape[1] * (target_fs / source_fs))
        resampled = resample(filtered, num_target_samples, axis=1).astype(np.float32)

    # 3. Recording-level standardization (matching the notebook)
    mean_ch = np.mean(resampled, axis=1, keepdims=True)
    std_ch = np.std(resampled, axis=1, keepdims=True) + 1e-6
    normalized = (resampled - mean_ch) / std_ch

    # 4. Consecutive 2.0s non-overlapping windows (128 samples per channel)
    window_samples = int(target_fs * window_sec) # 128
    num_windows = normalized.shape[1] // window_samples

    windows = []
    for w in range(num_windows):
        start = w * window_samples
        win = normalized[:, start:start + window_samples] # (4, 128)
        windows.append(win)

    return {
        "windows": np.array(windows, dtype=np.float32), # (num_windows, 4, 128)
        "num_windows": num_windows,
        "normalized_signal": normalized,
        "sample_rate": target_fs,
        "window_duration": window_sec
    }
