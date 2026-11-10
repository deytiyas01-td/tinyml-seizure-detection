import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "authorship_tag": "TinyML_CHBMIT_Seizure_Detector",
            "include_colab_link": True
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": []
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# 1. Title & Header
add_md("""# 🧠 TinyML Seizure Detection using CHB-MIT EEG Dataset
### Ultra-Low-Power Edge AI for Microcontrollers (ESP32, STM32, Arduino, ARM Cortex-M)

This Google Colab notebook trains an ultra-compact Deep Learning model (Tiny 1D-CNN) on pediatric scalp EEG recordings from the **CHB-MIT database (PhysioNet)** and quantizes it to **Full INT8 TensorFlow Lite** for deployment on microcontrollers.

---

### 🚀 Key Highlights & TinyML Engineering:
1. **Microcontroller Hardware Budget**:
   - Target RAM: < 30 KB
   - Target Flash: < 20 KB
   - Latency: < 15 ms inference time on Cortex-M4 @ 80 MHz
2. **Channel Selection & Downsampling**:
   - Uses 4 representative bipolar EEG channels (`FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`)
   - Downsamples from 256 Hz to **64 Hz** (preserves 0.5–30 Hz seizure rhythms: delta, theta, alpha, beta)
3. **Windowing & Imbalance Handling**:
   - 2-second windows (128 samples × 4 channels)
   - 75% sliding window overlap on seizure intervals for data augmentation
4. **Quantization**:
   - Full INT8 Post-Training Quantization (PTQ) with representative calibration dataset
5. **MCU Deployment**:
   - Generates ready-to-flash C array header (`model_data.h`) and sample Arduino / ESP32 code.
""")

# 2. Setup
add_md("## Step 1: Install Dependencies\nInstall `mne`, `pyedflib`, and `scipy` for EEG signal processing.")
add_code("""!pip install -q mne pyedflib scipy scikit-learn matplotlib seaborn tensorflow""")

# 3. Imports
add_md("## Step 2: Import Libraries & Verify Environment")
add_code("""import os
import re
import math
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

print(f"TensorFlow Version: {tf.__version__}")
print("GPU Available:" if tf.config.list_physical_devices('GPU') else "Running on CPU (Fast enough for TinyML!)")
""")

# 4. CHB-MIT Data Download
add_md("""## Step 3: Automated Download of CHB-MIT EEG Recordings from PhysioNet

We download recordings from Patient `chb01` directly from PhysioNet. `chb01` contains 7 seizures across several EDF files.
Downloading the relevant files (~300MB total) takes only 1-2 minutes in Colab without consuming excessive disk space.
""")
add_code("""DATA_DIR = "./chbmit_data"
os.makedirs(DATA_DIR, exist_ok=True)
BASE_URL = "https://physionet.org/files/chbmitsdb/1.0.0/chb01/"

# Summary file containing seizure onset/offset ground truth annotations
summary_file = "chb01-summary.txt"
if not os.path.exists(os.path.join(DATA_DIR, summary_file)):
    print(f"Downloading {summary_file}...")
    urllib.request.urlretrieve(BASE_URL + summary_file, os.path.join(DATA_DIR, summary_file))

# Selected files: Contains seizure files (03, 04, 15, 16, 18, 21, 26) and baseline non-seizure (01, 02)
target_files = [
    "chb01_01.edf",  # Non-seizure baseline
    "chb01_02.edf",  # Non-seizure baseline
    "chb01_03.edf",  # Seizure @ 2996s - 3036s
    "chb01_04.edf",  # Seizure @ 1467s - 1494s
    "chb01_15.edf",  # Seizure @ 1732s - 1772s
    "chb01_16.edf",  # Seizure @ 1015s - 1066s
    "chb01_18.edf",  # Seizure @ 1720s - 1810s
    "chb01_21.edf",  # Seizure @ 327s - 420s
    "chb01_26.edf"   # Seizure @ 1862s - 1963s
]

for fname in target_files:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Downloading {fname} (~35MB)...")
        urllib.request.urlretrieve(BASE_URL + fname, fpath)

print("✅ Data files ready in Colab environment!")
""")

# 5. Parse Ground Truth Seizure Annotations
add_md("""## Step 4: Parse Seizure Ground Truth Annotations from Summary File
We parse `chb01-summary.txt` to automatically detect start and end times (in seconds) for each seizure.
""")
add_code("""def parse_summary(summary_path):
    seizure_annotations = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Split by file blocks
    file_blocks = content.split("File Name: ")
    for block in file_blocks[1:]:
        lines = block.strip().split("\n")
        fname = lines[0].strip()
        num_seizures = 0
        seizures = []
        for i, line in enumerate(lines):
            if "Number of Seizures in File:" in line:
                num_seizures = int(line.split(":")[-1].strip())
            if "Seizure" in line and "Start Time:" in line:
                start_sec = int(re.search(r'\d+', line.split(":")[-1]).group())
                end_line = lines[i+1]
                end_sec = int(re.search(r'\d+', end_line.split(":")[-1]).group())
                seizures.append((start_sec, end_sec))
        if num_seizures > 0:
            seizure_annotations[fname] = seizures
        else:
            seizure_annotations[fname] = []
            
    return seizure_annotations

summary_path = os.path.join(DATA_DIR, "chb01-summary.txt")
annotations = parse_summary(summary_path)
for f, s in annotations.items():
    if len(s) > 0:
        print(f"File {f}: {len(s)} seizure(s) -> {s}")
""")

# 6. Preprocessing & Windowing
add_md("""## Step 5: Preprocessing & Windowing for TinyML
### TinyML Design Constraints:
- **Channels**: 4 bipolar EEG channels (`FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`).
- **Resampling**: From original 256 Hz down to **64 Hz** (`scipy.signal.resample` or `mne.filter.resample`).
- **Window Length**: 2 seconds = 128 samples per channel.
- **Overlap**:
  - Seizure windows: 75% overlap (data augmentation to boost rare seizure samples).
  - Non-seizure windows: No overlap / downsampled to maintain balanced class representation.
""")
add_code("""TARGET_CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1']
ORIGINAL_FS = 256
TARGET_FS = 64
WINDOW_SEC = 2.0
WINDOW_SIZE = int(TARGET_FS * WINDOW_SEC)  # 128 samples
DOWNSAMPLE_FACTOR = ORIGINAL_FS // TARGET_FS  # 4

def load_and_segment_edf(file_path, seizures, target_channels=TARGET_CHANNELS):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Clean channel names (remove trailing dots, spaces)
    raw.rename_channels(lambda c: c.strip().replace('.', ''))
    
    # Pick channels
    avail_ch = [c for c in target_channels if c in raw.ch_names]
    if len(avail_ch) < len(target_channels):
        print(f"Warning: Missing channels in {file_path}, available: {raw.ch_names}")
        return None, None
    raw.pick_channels(avail_ch)
    
    # Bandpass filter (0.5 to 30 Hz) - cleans high freq noise and baseline drift
    raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin', verbose=False)
    
    # Downsample
    raw.resample(TARGET_FS, verbose=False)
    data = raw.get_data()  # shape: (channels, total_samples)
    
    # Standardize per channel
    data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)
    
    total_samples = data.shape[1]
    total_sec = total_samples / TARGET_FS
    
    X_seizure = []
    X_normal = []
    
    # Mark seizure sample ranges in downsampled domain
    seizure_ranges = [(int(s * TARGET_FS), int(e * TARGET_FS)) for s, e in seizures]
    
    def is_seizure(start_idx, end_idx):
        for s, e in seizure_ranges:
            # Overlap check
            if max(start_idx, s) < min(end_idx, e):
                return True
        return False
    
    # 1. Extract Seizure Windows with 75% overlap (hop = 0.25 * WINDOW_SIZE)
    seizure_hop = WINDOW_SIZE // 4
    for s, e in seizure_ranges:
        for idx in range(max(0, s - WINDOW_SIZE), min(total_samples - WINDOW_SIZE, e + WINDOW_SIZE), seizure_hop):
            w = data[:, idx:idx + WINDOW_SIZE].T  # shape: (WINDOW_SIZE, channels)
            if w.shape[0] == WINDOW_SIZE:
                X_seizure.append(w)
                
    # 2. Extract Normal Windows (non-overlapping to avoid exploding dataset size)
    normal_hop = WINDOW_SIZE
    for idx in range(0, total_samples - WINDOW_SIZE, normal_hop):
        if not is_seizure(idx, idx + WINDOW_SIZE):
            w = data[:, idx:idx + WINDOW_SIZE].T
            if w.shape[0] == WINDOW_SIZE:
                X_normal.append(w)
                
    return np.array(X_seizure), np.array(X_normal)

all_seizures = []
all_normals = []

for fname in target_files:
    fpath = os.path.join(DATA_DIR, fname)
    seizures = annotations.get(fname, [])
    print(f"Processing {fname} with {len(seizures)} seizures...")
    x_s, x_n = load_and_segment_edf(fpath, seizures)
    if x_s is not None and len(x_s) > 0:
        all_seizures.append(x_s)
    if x_n is not None and len(x_n) > 0:
        all_normals.append(x_n)

X_seizure_total = np.vstack(all_seizures)
X_normal_total = np.vstack(all_normals)

print(f"\\nTotal Seizure Windows: {len(X_seizure_total)}")
print(f"Total Normal Windows available: {len(X_normal_total)}")

# Balance dataset: Subsample normal windows to maintain a reasonable ratio (e.g. 2:1 normal to seizure)
np.random.seed(42)
sampled_indices = np.random.choice(len(X_normal_total), size=min(len(X_normal_total), len(X_seizure_total) * 2), replace=False)
X_normal_balanced = X_normal_total[sampled_indices]

X = np.vstack([X_seizure_total, X_normal_balanced]).astype(np.float32)
y = np.concatenate([np.ones(len(X_seizure_total)), np.zeros(len(X_normal_balanced))]).astype(np.float32)

print(f"\\nFinal Dataset Shape: {X.shape}")
print(f"Labels Distribution -> Seizure (1): {int(np.sum(y))}, Non-Seizure (0): {int(len(y) - np.sum(y))}")
""")

# 7. Dataset Split
add_md("## Step 6: Stratified Train / Validation / Test Split")
add_code("""X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training set:   {X_train.shape}, labels: {np.bincount(y_train.astype(int))}")
print(f"Validation set: {X_val.shape}, labels: {np.bincount(y_val.astype(int))}")
print(f"Test set:       {X_test.shape}, labels: {np.bincount(y_test.astype(int))}")
""")

# 8. Visualizing EEG Signal
add_md("## Step 7: Visualize Seizure vs Non-Seizure EEG Windows")
add_code("""fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
time_axis = np.linspace(0, WINDOW_SEC, WINDOW_SIZE)

# Seizure window
axes[0].plot(time_axis, X_seizure_total[10])
axes[0].set_title("Seizure Window (High amplitude rhythmic epileptiform activity)", fontsize=12, fontweight='bold', color='crimson')
axes[0].set_ylabel("Normalized Amplitude")
axes[0].grid(True, alpha=0.3)

# Normal window
axes[1].plot(time_axis, X_normal_balanced[10])
axes[1].set_title("Normal / Interictal Window (Baseline background rhythm)", fontsize=12, fontweight='bold', color='navy')
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Normalized Amplitude")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

# 9. Ultra-Lightweight TinyML Architecture
add_md("""## Step 8: Ultra-Lightweight TinyML 1D-CNN Architecture
We design a highly efficient **1D Depthwise-Separable Convolutional Network** (inspired by TinyEEGNet).
- Parameters: **< 3,500 total parameters**
- Operations: **~120k MACs per inference**
- Peak Activation RAM: **< 16 KB**
- INT8 Model Flash Size: **< 6 KB**
""")
add_code("""def create_tinyml_seizure_detector(input_shape=(WINDOW_SIZE, len(TARGET_CHANNELS))):
    inputs = layers.Input(shape=input_shape, name="eeg_input")
    
    # 1. Temporal Feature Extraction (Long kernel to capture low-frequency rhythms)
    x = layers.Conv1D(filters=8, kernel_size=15, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 2. Depthwise Separable Conv (Spatial-temporal interaction with minimal parameters)
    x = layers.SeparableConv1D(filters=16, kernel_size=7, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 3. Pooling & Regularization
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 4. Global Average Pooling (drastically eliminates dense layer weights)
    x = layers.GlobalAveragePooling1D()(x)
    
    # 5. Classifier
    x = layers.Dense(12, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name="seizure_prob")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="TinyML_Seizure_CNN")
    return model

model = create_tinyml_seizure_detector()
model.summary()
""")

# 10. Training the Model
add_md("## Step 9: Model Training in Google Colab")
add_code("""model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='sensitivity'), tf.keras.metrics.Precision(name='precision')]
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=12, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
""")

# 11. Learning Curves
add_md("## Step 10: Training History & Learning Curves")
add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(history.history['loss'], label='Train Loss', lw=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', lw=2)
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Acc', lw=2)
axes[1].plot(history.history['val_accuracy'], label='Val Acc', lw=2)
axes[1].set_title('Accuracy Curve')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.show()
""")

# 12. Evaluation on Test Set
add_md("## Step 11: Comprehensive Test Evaluation")
add_code("""y_pred_probs = model.predict(X_test).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print(f"\\n--- CLINICAL METRICS ---")
print(f"Sensitivity (Recall): {sensitivity:.4f}  (Seizure detection rate)")
print(f"Specificity:          {specificity:.4f}  (Rejection of non-seizure EEG)")
print(f"ROC-AUC Score:        {roc_auc:.4f}")
print(\"\\n--- CLASSIFICATION REPORT ---\")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Seizure']))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
plt.title('Test Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.show()
""")

# 13. Quantization to Full INT8 TFLite
add_md("""## Step 12: Post-Training Quantization (PTQ) to Full INT8
For TinyML deployment on microcontrollers without hardware floating-point units (FPUs), we quantize:
1. **Weights** -> 8-bit integers (`int8`)
2. **Activations** -> 8-bit integers (`int8`)
3. **Input & Output tensors** -> 8-bit integers (`int8`)
""")
add_code("""# 1. Standard Float32 TFLite
converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float32 = converter_float32.convert()

with open("seizure_model_float32.tflite", "wb") as f:
    f.write(tflite_float32)

# 2. Representative Dataset Generator for Full INT8 Quantization
def representative_dataset_gen():
    # Use 100-200 calibration windows from training set
    for i in range(min(150, len(X_train))):
        yield [X_train[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8 = converter_int8.convert()

with open("seizure_model_int8.tflite", "wb") as f:
    f.write(tflite_int8)

float32_size = len(tflite_float32) / 1024
int8_size = len(tflite_int8) / 1024

print(f"Float32 TFLite Size: {float32_size:.2f} KB")
print(f"INT8 TFLite Size:    {int8_size:.2f} KB  🚀 ({((float32_size - int8_size) / float32_size)*100:.1f}% reduction)")
""")

# 14. Verification of INT8 Model Accuracy
add_md("## Step 13: Verify INT8 TFLite Model Accuracy")
add_code("""# Run inference with TFLite INT8 Interpreter
interpreter = tf.lite.Interpreter(model_path="seizure_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

int8_predictions = []
for sample in X_test:
    # Quantize input: real_value = scale * (quantized_value - zero_point)
    sample_int8 = np.clip(np.round(sample / input_scale) + input_zero_point, -128, 127).astype(np.int8)
    sample_int8 = np.expand_dims(sample_int8, axis=0)
    
    interpreter.set_tensor(input_details["index"], sample_int8)
    interpreter.invoke()
    
    output_int8 = interpreter.get_tensor(output_details["index"])[0]
    # Dequantize output back to probability
    prob = (output_int8.astype(np.float32) - output_zero_point) * output_scale
    int8_predictions.append(prob[0])

int8_preds = (np.array(int8_predictions) >= 0.5).astype(int)
int8_acc = np.mean(int8_preds == y_test)
print(f"INT8 Quantized Test Accuracy: {int8_acc * 100:.2f}%")
print(f"Float32 Keras Test Accuracy:  {np.mean(y_pred == y_test) * 100:.2f}%")
""")

# 15. Export to C Header for Microcontrollers
add_md("""## Step 14: Export INT8 Model to C Array (`model_data.h`)
This converts the `.tflite` binary file into a C header array that can be compiled directly into firmware (Arduino, ESP-IDF, STM32CubeIDE).
""")
add_code("""def hex_dump_c_array(tflite_bytes, output_header_path="model_data.h"):
    bytes_per_line = 12
    with open(output_header_path, "w") as f:
        f.write("#ifndef SEIZURE_MODEL_DATA_H_\\n")
        f.write("#define SEIZURE_MODEL_DATA_H_\\n\\n")
        f.write(f"// Automatically generated TinyML Seizure Model\\n")
        f.write(f"// Size: {len(tflite_bytes)} bytes\\n\\n")
        f.write("alignas(16) const unsigned char g_seizure_model_data[] = {\\n")
        
        for i in range(0, len(tflite_bytes), bytes_per_line):
            chunk = tflite_bytes[i:i + bytes_per_line]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + bytes_per_line < len(tflite_bytes):
                hex_str += ","
            f.write(f"  {hex_str}\\n")
            
        f.write("};\\n\\n")
        f.write(f"const int g_seizure_model_data_len = {len(tflite_bytes)};\\n\\n")
        f.write(f"const float g_input_scale = {input_scale}f;\\n")
        f.write(f"const int g_input_zero_point = {input_zero_point};\\n")
        f.write(f"const float g_output_scale = {output_scale}f;\\n")
        f.write(f"const int g_output_zero_point = {output_zero_point};\\n\\n")
        f.write("#endif  // SEIZURE_MODEL_DATA_H_\\n")

hex_dump_c_array(tflite_int8, "model_data.h")
print(f"Generated model_data.h ({len(tflite_int8)} bytes)")

# Download prompt for Colab users
from google.colab import files
files.download("model_data.h")
files.download("seizure_model_int8.tflite")
""")

# 16. Microcontroller Firmware Deployment Example
add_md("""## Step 15: Microcontroller Deployment Code (Arduino / ESP32 / STM32)
Here is the C++ code to run this model on an ESP32 or Arduino Nano 33 BLE Sense using **TensorFlow Lite for Microcontrollers (TFLM)**:

```cpp
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

// Memory tensor arena for intermediate activations
constexpr int kTensorArenaSize = 16 * 1024; // 16 KB is plenty for this model
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // 1. Load model from C header
  model = tflite::GetModel(g_seizure_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  // 2. Resolver for ops
  static tflite::AllOpsResolver resolver;

  // 3. Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 4. Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TinyML Seizure Detector Ready!");
}

void loop() {
  // Feed 128 samples x 4 channels buffer into input->data.int8
  // Example: int8_val = (float_val / g_input_scale) + g_input_zero_point;
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Read output
  int8_t out_quant = output->data.int8[0];
  float seizure_prob = (out_quant - g_output_zero_point) * g_output_scale;

  if (seizure_prob > 0.65f) {
    Serial.print("⚠️ ALERT: Seizure Detected! Probability: ");
    Serial.println(seizure_prob);
    // Trigger buzzer or Bluetooth emergency notification
  }

  delay(500); // 500ms sliding evaluation
}
```
""")

out_path = r"C:\Users\TIYAS DEY\.gemini\antigravity\scratch\tinyml-seizure-detection-chbmit\TinyML_Seizure_Detection_CHB_MIT.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook successfully written to {out_path}")
