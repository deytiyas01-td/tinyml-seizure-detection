import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project path is loaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import engine
import edf_parser

st.set_page_config(
    page_title="EEG Seizure Detection Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clinical Clean Theme Stylesheet
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .header-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 3px;
    }
    .clinical-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .alert-seizure {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .alert-normal {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .alert-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .alert-title-seizure { color: #991b1b; }
    .alert-title-normal { color: #166534; }
    .alert-desc {
        font-size: 0.82rem;
        color: #475569;
        margin: 0;
    }
    .metric-pill {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: Menlo, Consolas, monospace;
    }
    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.78rem;
        color: #92400e;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Top Clinical Header & Disclaimer
st.markdown("""
<div class="header-box">
    <h1 class="header-title">EEG Seizure Detection Monitor</h1>
    <p class="header-subtitle">CHB-MIT Scalp EEG Analysis & Automated Detection (4 Channels: FP1-F7, F7-T7, T7-P7, P7-O1 @ 64 Hz)</p>
</div>
<div class="disclaimer-box">
    <strong>Notice:</strong> This software is an experimental TinyML demonstration based on the CHB-MIT Scalp EEG dataset. It is not intended for clinical diagnosis, patient treatment, or medical decision-making.
</div>
""", unsafe_allow_html=True)

# Session State Initialization
if "current_sequence" not in st.session_state:
    st.session_state.current_sequence = engine.REFERENCE_EXAMPLES[0]["data"]
if "sample_label" not in st.session_state:
    st.session_state.sample_label = engine.REFERENCE_EXAMPLES[0]["name"]
if "is_synthetic" not in st.session_state:
    st.session_state.is_synthetic = True
if "edf_analysis" not in st.session_state:
    st.session_state.edf_analysis = None

# Input Selection Tabs (Real EDF Upload vs. Synthetic Demonstration Presets)
tab_edf, tab_presets, tab_manual = st.tabs([
    "📁 Upload Real EDF File", 
    "🧪 Synthetic Test Signals (UI Verification)",
    "⌨️ Manual Sequence Input (512 Values)"
])

# TAB 1: Real EDF File Input
with tab_edf:
    st.markdown("#### Upload Clinical Scalp EEG Recording (.edf)")
    st.write("Upload a genuine recording in European Data Format (EDF). The system will validate the 4 required bipolar channels (`FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`), resample from source frequency to 64 Hz, and apply training-compatible standardization.")
    
    uploaded_file = st.file_uploader("Select .edf file", type=["edf"], key="edf_uploader")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Parsing and validating EDF file channels..."):
                edf_res = engine.evaluate_edf_stream(uploaded_file)
                st.session_state.edf_analysis = edf_res
            
            st.success(f"✅ Successfully validated and processed **{uploaded_file.name}**")
            
            # Metadata metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duration", f"{edf_res['duration_sec']} s")
            m2.metric("Source Rate", f"{edf_res['source_fs']:.0f} Hz")
            m3.metric("2-Sec Windows", f"{edf_res['num_windows']}")
            m4.metric("Seizure Windows", f"{edf_res['seizure_windows_count']} / {edf_res['num_windows']}")

            # Channel Mapping Confirmation
            st.caption(f"**Verified Bipolar Channels:** {', '.join([f'{k} -> {v}' for k, v in edf_res['channel_mapping'].items()])}")

            # Window Navigator Slider
            if edf_res['num_windows'] > 0:
                selected_w = st.slider(
                    "Navigate 2-Second Consecutive Windows:",
                    min_value=0,
                    max_value=edf_res['num_windows'] - 1,
                    value=0,
                    format="Window %d"
                )
                w_info = edf_res["windows_summary"][selected_w]
                st.session_state.current_sequence = edf_res["all_windows_data"][selected_w]
                st.session_state.sample_label = f"{uploaded_file.name} — Window {selected_w} ({w_info['time_range']})"
                st.session_state.is_synthetic = False
                
        except ValueError as ve:
            st.error(f"❌ Channel Validation Failed:\n{ve}")
        except Exception as e:
            st.error(f"❌ Error reading EDF: {e}")

# TAB 2: Synthetic Demonstration Signals
with tab_presets:
    st.markdown("#### Synthetic Demonstration Signals")
    st.caption("⚠️ Note: These 4 signals are synthetic mathematical waveforms used to verify the interface layout and API contract. They are not real patient recordings.")
    
    cols = st.columns(4)
    for i, ex in enumerate(engine.REFERENCE_EXAMPLES):
        with cols[i]:
            tag = "SYNTHETIC SEIZURE" if "seizure" in ex["type"] else "SYNTHETIC NORMAL"
            if st.button(f"{ex['name']}\n[{tag}]", key=f"btn_ex_{i}", use_container_width=True):
                st.session_state.current_sequence = ex["data"]
                st.session_state.sample_label = ex["name"]
                st.session_state.is_synthetic = True

# TAB 3: Manual Sequence Input
with tab_manual:
    st.markdown("#### Manual 4-Channel Sequence Input")
    st.caption("Input Contract: Exactly 4 channels × 128 samples (total 512 values) in fixed channel order: `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`.")
    user_input = st.text_area(
        "Paste JSON 4x128 array or 512 comma-separated numbers:",
        value=str(st.session_state.current_sequence),
        height=100
    )
    if st.button("Analyze Custom Sequence", type="primary"):
        import json
        try:
            raw_text = user_input.strip()
            if raw_text.startswith("["):
                parsed = json.loads(raw_text)
            else:
                nums = [float(x.strip()) for x in raw_text.replace("\n", ",").split(",") if x.strip()]
                if len(nums) == 512:
                    parsed = [nums[0:128], nums[128:256], nums[256:384], nums[384:512]]
                elif len(nums) == 128:
                    st.error("Single-channel input (128 values) is not permitted. Please provide all 4 channels (512 values total).")
                    parsed = None
                else:
                    st.error(f"Expected 512 values (4 channels x 128 samples), got {len(nums)}.")
                    parsed = None
            if parsed is not None:
                st.session_state.current_sequence = parsed
                st.session_state.sample_label = "Custom User Sequence (4 Channels x 128 Samples)"
                st.session_state.is_synthetic = False
                st.rerun()
        except Exception as e:
            st.error(f"Failed to parse custom sequence: {e}")

st.markdown("---")

# Main Analysis Display (Waveform left, Diagnostic Report right)
left_col, right_col = st.columns([1.35, 1.0], gap="large")

with left_col:
    st.markdown("### EEG Waveform Montage (2.0-Second Window @ 64 Hz)")
    source_tag = "🧪 Synthetic Signal" if st.session_state.is_synthetic else "📁 Real Recording Data"
    st.caption(f"Active Sample: **{st.session_state.sample_label}** &bull; [{source_tag}]")
    
    # Plot Stacked EEG Traces
    data_4x128 = np.array(st.session_state.current_sequence, dtype=np.float32)
    if data_4x128.shape == (128, 4):
        data_4x128 = data_4x128.T
    elif data_4x128.ndim == 1 and len(data_4x128) == 512:
        data_4x128 = data_4x128.reshape(4, 128)
        
    time_sec = np.linspace(0, 2.0, 128)
    channels = engine.CHANNELS
    offsets = [6.0, 2.0, -2.0, -6.0]
    colors = ["#0284c7", "#2563eb", "#4f46e5", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    for ch_idx in range(4):
        offset = offsets[ch_idx]
        ch_name = channels[ch_idx]
        shifted = data_4x128[ch_idx] + offset
        ax.plot(time_sec, shifted, color=colors[ch_idx], lw=1.3, label=f"{ch_name} ({offset:+0.1f} μV)")

    ax.set_xlabel("Time (seconds) @ 64 Hz (128 samples per channel)", fontsize=8.5, color="#475569")
    ax.set_ylabel("Standardized Amplitude (μV) [Stacked]", fontsize=8.5, color="#475569")
    ax.set_xlim([0, 2.0])
    ax.set_ylim([-9.5, 9.5])
    ax.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, facecolor="#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.tick_params(colors="#64748b", labelsize=8)

    st.pyplot(fig)
    plt.close(fig)

with right_col:
    st.markdown("### Detection Result")
    
    try:
        # Run inference using the pre-extracted numpy model
        result = engine.run_inference(st.session_state.current_sequence)
        is_seizure = result["is_seizure"]
        conf = result["confidence_percentage"]
        prob = result["probability"]

        if is_seizure:
            st.markdown(f"""
            <div class="alert-seizure">
                <div class="alert-title alert-title-seizure">Seizure Activity Detected</div>
                <p class="alert-desc">Rhythmic epileptiform activity identified in the evaluated 2-second window.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-normal">
                <div class="alert-title alert-title-normal">No Seizure Detected (Normal EEG)</div>
                <p class="alert-desc">Baseline physiological rhythm within normal limits.</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Card
        st.markdown(f"""
        <div class="clinical-card">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-weight:600; color:#475569; font-size:0.9rem;">Confidence</span>
                <span class="metric-pill" style="color:{'#dc2626' if is_seizure else '#16a34a'};">{conf:.2f}%</span>
            </div>
            <div style="font-size:0.75rem; color:#64748b; font-family:monospace; margin-top:8px;">
                Sigmoid Output Probability: <strong>{prob:.4f}</strong> &bull; Threshold: <strong>0.5000</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Frequency Bands Breakdown
        bands = result.get("frequency_bands", {})
        st.markdown("#### Frequency Band Distribution")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Delta (0.5–4Hz)", f"{bands.get('delta', 0)}%")
        b2.metric("Theta (4–8Hz)", f"{bands.get('theta', 0)}%")
        b3.metric("Alpha (8–13Hz)", f"{bands.get('alpha', 0)}%")
        b4.metric("Beta (13–30Hz)", f"{bands.get('beta', 0)}%")

        # Channel Statistics Table
        st.markdown("#### Channel Signal Metrics")
        stats = result.get("channel_stats", [])
        if stats:
            st.dataframe(
                stats,
                column_config={
                    "channel": "Electrode",
                    "mean": "Mean",
                    "std": "Std Dev",
                    "peak_to_peak": "Peak-to-Peak (μV)"
                },
                hide_index=True,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Inference error: {e}")

st.markdown("---")
st.caption("CHB-MIT Scalp EEG Seizure Detection Model • Model Artifact: model_weights.npz (NumPy inference matching INT8 TFLite) • Fixed Channel Order: FP1-F7, F7-T7, T7-P7, P7-O1")
