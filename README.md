# 🧠 TinyML Epileptic Seizure Detection System (CHB-MIT EEG)

An end-to-end, ultra-low-power edge AI platform for real-time epileptic seizure detection from multi-channel scalp EEG signals.

---

## 🌟 Key Application Features

1. **User-Friendly Medical Web Dashboard**:
   - **4 Instant One-Click Examples on Screen**:
     - 🔴 **Seizure Pattern A (Spike-and-Wave Paroxysm, 3.0 Hz)** $\rightarrow$ **99.88% Seizure Confidence**
     - 🔴 **Seizure Pattern B (Rhythmic Fast Burst, 4.0 Hz)** $\rightarrow$ **99.75% Seizure Confidence**
     - 🟢 **Non-Seizure Pattern A (Normal Posterior Alpha Rhythm, 10.0 Hz)** $\rightarrow$ **97.00% Normal Confidence**
     - 🟢 **Non-Seizure Pattern B (Normal Baseline / Drowsy Rhythm)** $\rightarrow$ **97.21% Normal Confidence**
   - **Interactive Multi-Channel Waveform Plotter**: Live clinical stacked EEG montage for channels `FP1-F7`, `F7-T7`, `T7-P7`, and `P7-O1`.
   - **Real-Time Confidence Meter**: Percentage confidence, risk level badge, and raw probability score.
   - **Spectral Band Power Analyzer**: Real-time Delta (0.5–4Hz), Theta (4–8Hz), Alpha (8–13Hz), and Beta (13–30Hz) breakdown via Fast Fourier Transform (FFT).
   - **Custom Sequence Testing**: Easily edit or paste custom 128-sample / 512-sample EEG sequences.
   - **Model Training Visualizer**: Embedded interactive displays for the training curves (`curves.png`) and confusion matrix (`confusion_matrix.png`) with lightbox modal view.

2. **Secure Backend Engine (`app.py` & `engine.py`)**:
   - **Direct INT8 TFLite Weights Execution**: Uses the exact trained model weights uploaded from your Google Colab training session.
   - **Strict Input Validation & Sanitization**: Rejects non-numeric tokens, `NaN` / `Infinity` values, and extreme out-of-range signal spikes.
   - **Payload Flood Protection**: Enforces 1 MB payload limits to prevent denial-of-service memory exhaustion.
   - **Rate Limiting**: Sliding-window client request throttling.
   - **Security Headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, and `Content-Security-Policy`.

---

## 🚀 How to Launch the Web Application

### Option A: One-Click Launch (Windows)
Double-click [`run_server.bat`](file:///C:/Users/TIYAS%20DEY/.gemini/antigravity/scratch/tinyml-seizure-detection-chbmit/run_server.bat) in the project folder.

### Option B: Terminal Command
Run the following in PowerShell / Command Prompt:
```bash
cd "C:\Users\TIYAS DEY\.gemini\antigravity\scratch\tinyml-seizure-detection-chbmit"
python app.py
```

Then open your web browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 📁 Project Structure

```
tinyml-seizure-detection-chbmit/
├── app.py                             # Secure Flask web server & REST API
├── engine.py                          # Fast INT8 model inference & spectral analysis engine
├── verify_api.py                      # Comprehensive verification test suite
├── run_server.bat                     # Windows one-click launcher
├── seizure_model_int8.tflite          # Your uploaded INT8 quantized TFLite model (10.5 KB)
├── model_data.h                       # C array header for microcontroller firmware
├── seizre detection my test curves.png              # Training/validation curves from Colab
├── seizre detection my test confusion matrix.png    # Confusion matrix from Colab
├── TinyML_Seizure_Detection_CHB_MIT.ipynb          # Google Colab training notebook
└── static/
    ├── index.html                     # Clean, accessible medical dashboard UI
    ├── style.css                      # Modern dark theme & responsive styles
    ├── app.js                         # Dynamic waveforms & API interaction logic
    └── images/                        # Confusion matrix and training curves
```

---

## 🛠️ Microcontroller Hardware Blueprint

| Parameter | Specification |
| :--- | :--- |
| **Model Flash Size** | **10.25 KB** |
| **Tensor Arena RAM** | **16 KB** |
| **Quantization** | Full INT8 |
| **Input Shape** | 128 samples $\times$ 4 channels @ 64 Hz (2.0s window) |
| **Target Hardware** | ESP32, STM32 (Cortex-M4/M7), Arduino Nano 33 BLE, Raspberry Pi Pico |
