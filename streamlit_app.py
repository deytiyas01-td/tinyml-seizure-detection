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

st.set_page_config(
    page_title="EEG Seizure Detection Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clinical Clean White Theme CSS
st.markdown("""
<style>
    /* Clean white background and clinical typography */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .header-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Section Cards */
    .clinical-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    /* Diagnostic Alert Boxes */
    .alert-seizure {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .alert-normal {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .alert-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .alert-title-seizure { color: #991b1b; }
    .alert-title-normal { color: #166534; }
    
    .alert-desc {
        font-size: 0.85rem;
        color: #475569;
        margin: 0;
    }
    
    /* Metric pill */
    .metric-pill {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: Menlo, Consolas, monospace;
    }
</style>
""", unsafe_allow_html=True)

# Top Clinical Header
st.markdown("""
<div class="header-box">
    <h1 class="header-title">EEG Seizure Detection Monitor</h1>
    <p class="header-subtitle">CHB-MIT Scalp EEG Analysis & Automated Detection Protocol</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "selected_sample_idx" not in st.session_state:
    st.session_state.selected_sample_idx = 0
if "current_sequence" not in st.session_state:
    st.session_state.current_sequence = engine.REFERENCE_EXAMPLES[0]["data"]
if "sample_label" not in st.session_state:
    st.session_state.sample_label = engine.REFERENCE_EXAMPLES[0]["name"]

# 4 Test Sample Selection Buttons
st.markdown("### Test Examples")
st.write("Click any sample below to load its multi-channel EEG sequence and evaluate detection:")

cols = st.columns(4)
for i, ex in enumerate(engine.REFERENCE_EXAMPLES):
    with cols[i]:
        tag = "SEIZURE" if ex["type"] == "seizure" else "NON-SEIZURE"
        button_label = f"{ex['name']}\n[{tag}]"
        if st.button(button_label, key=f"btn_ex_{i}", use_container_width=True):
            st.session_state.selected_sample_idx = i
            st.session_state.current_sequence = ex["data"]
            st.session_state.sample_label = ex["name"]

st.caption(f"Currently active: **{st.session_state.sample_label}**")
st.markdown("---")

# Main Content Layout (Waveform left, Diagnostics right)
left_col, right_col = st.columns([1.3, 1.0], gap="large")

with left_col:
    st.markdown("### EEG Waveform Montage (4 Channels, 2.0s Window)")
    
    # Plot Stacked Clinical EEG Channels
    data_4x128 = np.array(st.session_state.current_sequence, dtype=np.float32)
    if data_4x128.shape == (128, 4):
        data_4x128 = data_4x128.T
    elif data_4x128.ndim == 1 and len(data_4x128) == 512:
        data_4x128 = data_4x128.reshape(4, 128)
        
    time_sec = np.linspace(0, 2.0, 128)
    channels = engine.CHANNELS
    offsets = [6.0, 2.0, -2.0, -6.0]
    colors = ["#0284c7", "#2563eb", "#4f46e5", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    for ch_idx in range(4):
        offset = offsets[ch_idx]
        ch_name = channels[ch_idx]
        shifted = data_4x128[ch_idx] + offset
        ax.plot(time_sec, shifted, color=colors[ch_idx], lw=1.3, label=f"{ch_name} ({offset:+0.1f} μV)")

    ax.set_xlabel("Time (seconds) @ 64 Hz", fontsize=9, color="#475569")
    ax.set_ylabel("Standardized Amplitude (μV) [Stacked]", fontsize=9, color="#475569")
    ax.set_xlim([0, 2.0])
    ax.set_ylim([-9.5, 9.5])
    ax.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, facecolor="#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.tick_params(colors="#64748b", labelsize=8)

    st.pyplot(fig)
    plt.close(fig)

    # Manual sequence input accordion
    with st.expander("Manual Sequence Input (Optional)"):
        user_input = st.text_area(
            "Paste comma-separated or JSON sequence (128 or 512 numbers):",
            value=str(st.session_state.current_sequence),
            height=120
        )
        if st.button("Analyze Custom Sequence", type="primary"):
            import json
            try:
                if user_input.strip().startswith("["):
                    parsed = json.loads(user_input)
                else:
                    nums = [float(x.strip()) for x in user_input.replace("\n", ",").split(",") if x.strip()]
                    if len(nums) == 512:
                        parsed = [nums[0:128], nums[128:256], nums[256:384], nums[384:512]]
                    elif len(nums) == 128:
                        parsed = [nums, nums, nums, nums]
                    else:
                        st.error(f"Expected 128 or 512 values, got {len(nums)}.")
                        parsed = None
                if parsed is not None:
                    st.session_state.current_sequence = parsed
                    st.session_state.sample_label = "Custom User Input"
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse input: {e}")

# Run Clinical Detection Inference
with right_col:
    st.markdown("### Detection Result")
    
    try:
        result = engine.run_inference(st.session_state.current_sequence)
        is_seizure = result["is_seizure"]
        conf = result["confidence_percentage"]
        prob = result["probability"]

        if is_seizure:
            st.markdown(f"""
            <div class="alert-seizure">
                <div class="alert-title alert-title-seizure">Seizure Activity Detected</div>
                <p class="alert-desc">Epileptiform rhythmic paroxysm identified in the evaluated EEG window.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-normal">
                <div class="alert-title alert-title-normal">No Seizure Detected (Normal EEG)</div>
                <p class="alert-desc">Baseline physiological rhythm within normal limits. No epileptiform transients.</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Card
        st.markdown(f"""
        <div class="clinical-card">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-weight:600; color:#475569; font-size:0.9rem;">Confidence</span>
                <span class="metric-pill" style="color:{'#dc2626' if is_seizure else '#16a34a'};">{conf:.2f}%</span>
            </div>
            <div style="font-size:0.78rem; color:#64748b; font-family:monospace; margin-top:8px;">
                Model Probability: <strong>{prob:.4f}</strong> &bull; Decision Threshold: <strong>0.5000</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Frequency Bands Breakdown
        bands = result.get("frequency_bands", {})
        st.markdown("#### Frequency Band Power Distribution")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Delta (0.5–4Hz)", f"{bands.get('delta', 0)}%")
        b2.metric("Theta (4–8Hz)", f"{bands.get('theta', 0)}%")
        b3.metric("Alpha (8–13Hz)", f"{bands.get('alpha', 0)}%")
        b4.metric("Beta (13–30Hz)", f"{bands.get('beta', 0)}%")

        # Channel Statistics Table
        st.markdown("#### Channel Statistics")
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
st.caption("EEG Clinical Seizure Classification Monitor • CHB-MIT Scalp EEG Database • Fast NumPy Inference")
