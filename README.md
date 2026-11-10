# 🧠 EEG Seizure Detection System (CHB-MIT Scalp EEG)

An edge-optimized epileptic seizure detection system trained on the **CHB-MIT Scalp EEG Database** (PhysioNet).

---

## 📋 Input Contract & Specifications

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Bipolar Channels (4)** | `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1` | Strict fixed order. Single-channel inputs are rejected. |
| **Source Sampling Rate** | 256 Hz | Standard CHB-MIT acquisition rate |
| **Inference Sampling Rate**| 64 Hz | Resampled to reduce compute & memory footprint |
| **Window Duration** | 2.0 seconds | Consecutive non-overlapping windows |
| **Samples per Channel** | 128 samples | 64 Hz × 2.0 s = 128 samples |
| **Total Values per Window**| **512 values** | Shape `(128, 4)` or `(4, 128)` |
| **Preprocessing** | Bandpass 0.5–30 Hz, (x - μ) / (σ + 1e-6) | Recording-level channel standardization |

> **Input Rectification Note:** The single-channel (128-value) duplication shortcut has been removed. The inference engine strictly enforces four independent EEG channels totaling 512 values.

---

## 🔬 Model Artifacts & Consistency

The system maintains consistency across edge and cloud environments:
1. **`seizure_model_int8.tflite`**: The trained INT8 quantized TensorFlow Lite model for microcontrollers (ESP32, STM32).
2. **`model_data.h`**: The C byte array representation of the INT8 model for embedded firmware.
3. **`model_weights.npz`**: The exact floating-point and integer weights extracted directly from the INT8 TFLite model, enabling portable, high-speed NumPy forward passes in cloud serverless and web environments without requiring the 1 GB TensorFlow runtime.

Run `python verify_model_consistency.py` to verify the input contract and model execution.

---

## 📁 Real EDF File Processing

The system includes a built-in parser for clinical European Data Format (`.edf`) files:
- Validates the presence of the 4 required bipolar channels (`FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`).
- If any required channel is missing, the system aborts with a descriptive error listing the missing channels and does not silently substitute arbitrary signals.
- Resamples from source frequency (e.g. 256 Hz) to 64 Hz.
- Applies recording-level channel standardization matching the training notebook.
- Segments the recording into consecutive 2.0-second windows and evaluates seizure activity across the recording.

---

## 🧪 Demonstration Signals

The 4 built-in test buttons on the dashboard are **synthetic demonstration signals** designed for UI and API verification:
- `Synthetic Pattern 1`: Simulated 3.0 Hz spike-and-wave rhythm
- `Synthetic Pattern 2`: Simulated 4.0 Hz rhythmic paroxysmal burst
- `Synthetic Pattern 3`: Simulated 10.0 Hz resting alpha background
- `Synthetic Pattern 4`: Simulated 1.5 Hz slow drift baseline

These demonstration signals are mathematical approximations for layout testing and should not be cited as real patient clinical recordings or evidence of clinical performance.

---

## ⚠️ Important Limitations & Clinical Disclaimer

This software and model are developed strictly for **educational, experimental, and demonstration purposes**.
- **Not for Clinical Diagnosis:** This application is not a medical device, is not approved by regulatory authorities (FDA, CE-MDR), and must not be used for patient monitoring, medical triage, or clinical decision-making.
- **Experimental Proof-of-Concept:** Any output represents the prediction of an experimental deep learning model under specific research constraints.

---

## 🚀 Running the Streamlit Web Application

To run the updated dashboard locally:
```bash
streamlit run streamlit_app.py
```
Or view the live cloud deployment on Streamlit Community Cloud:
👉 **`https://tinyml-seizure-detection-tiyas.streamlit.app`**
