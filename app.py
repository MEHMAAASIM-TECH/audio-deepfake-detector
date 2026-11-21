import os
import tempfile
import subprocess

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
TARGET_SR = 16000
MAX_SECONDS = 45
PITCH_MAX_SECONDS = 25
HOP = 256
FRAME = 1024
MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

# 1. Page Configuration
st.set_page_config(
    page_title="Audio Deepfake & AI Voice Detector",
    page_icon="🎙️",
    layout="wide"
)

# 2. Clean, Minimal Enterprise Dark & Red Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .clean-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .stButton > button {
        width: 100%;
        background-color: #da3633;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover { background-color: #b62324; }
    .alert-ai {
        background-color: rgba(218, 54, 51, 0.1);
        border: 1px solid #da3633;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
    }
    .alert-human {
        background-color: rgba(46, 160, 67, 0.1);
        border: 1px solid #2ea043;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
    }
    .summary-box {
        background-color: #1c2128;
        border-left: 4px solid #da3633;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
for key, default in {
    "page": "upload",
    "predictions": None,
    "raw_predictions": None,
    "file_name": "",
    "features": None,
    "figure": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar Control Panel
with st.sidebar:
    st.markdown("### 🛡️ **Control Panel**")
    st.markdown("---")
    st.info("🤖 **Model:** Wav2Vec2-XLSR (Multilingual)\n\n🎚️ **Input:** 16 kHz mono")
    invert_labels = st.checkbox(
        "Invert model labels",
        value=False,
        help="Leave unchecked when 'fake/spoof' already maps to AI and 'real/bonafide' maps to HUMAN."
    )
    st.markdown("---")
    st.markdown("### **Supported Formats**")
    st.markdown("- `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`\n- `.opus` (WhatsApp)\n- `.mpeg`, `.mp4`, `.mov`")


@st.cache_resource(show_spinner=False)
def load_detector():
    return pipeline("audio-classification", model=MODEL_ID)


def convert_to_wav(in_path, out_path):
    cmd = [
        "ffmpeg", "-y", "-i", in_path, "-vn", "-t", str(MAX_SECONDS),
        "-af", "highpass=f=80,loudnorm",
        "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1", out_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg could not extract audio: {res.stderr.strip()[-250:]}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("No usable audio track was found in this file.")


def split_scores(raw_preds):
    if not raw_preds:
        return 0.5, 0.5
    ai, human = None, None
    for p in raw_preds:
        lab = str(p.get("label", "")).lower().strip()
        score = float(p.get("score", 0.0))
        if any(k in lab for k in ("fake", "spoof", "synthetic")) or lab in ("ai", "ai-generated"):
            ai = score
        elif any(k in lab for k in ("real", "bonafide", "human", "genuine", "authentic")):
            human = score
    if ai is None and human is None:
        ai = float(raw_preds[0].get("score", 0.0))
        human = max(0.0, 1.0 - ai)
    elif ai is None:
        ai = max(0.0, 1.0 - human)
    elif human is None:
        human = max(0.0, 1.0 - ai)
    total = ai + human
    if total > 0:
        ai, human = ai / total, human / total
    return ai, human


def extract_features_and_plots(y, sr):
    if len(y) < int(sr * 0.5):
        raise ValueError("Audio is too short (need at least 0.5 seconds).")

    y_pitch = y[:int(sr * PITCH_MAX_SECONDS)]
    f0, voiced, _ = librosa.pyin(
        y_pitch, fmin=65, fmax=400, sr=sr, frame_length=FRAME, hop_length=HOP
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    rms_pitch = librosa.feature.rms(y=y_pitch, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms_pitch), len(times))
    f0, voiced, rms_pitch, times = f0[:n], voiced[:n], rms_pitch[:n], times[:n]

    f0_v = f0[voiced & ~np.isnan(f0)]
    feats = {
        "voiced_ratio": float(np.mean(voiced)) * 100 if len(voiced) > 0 else 0.0
    }
    if len(f0_v) > 10:
        med_f0 = np.median(f0_v)
        if med_f0 > 0:
            feats["pitch_mean"] = float(np.mean(f0_v))
            feats["pitch_std_semitones"] = float(np.std(12 * np.log2(f0_v / med_f0)))
        idx = np.where(voiced[:-1] & voiced[1:] & ~np.isnan(f0[:-1]) & ~np.isnan(f0[1:]) & (f0[:-1] > 0) & (f0[1:] > 0))[0]
        if len(idx) > 5:
            periods = 1.0 / f0
            feats["jitter"] = float(np.mean(np.abs(periods[idx + 1] - periods[idx])) / (np.mean(periods[idx]) + 1e-9) * 100)
            feats["shimmer"] = float(np.mean(np.abs(rms_pitch[idx + 1] - rms_pitch[idx])) / (np.mean(rms_pitch[idx]) + 1e-9) * 100)

    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME)
    total_s = float(S.sum()) + 1e-12
    feats["hf_energy_pct"] = float(S[freqs >= 4000].sum()) / total_s * 100
    feats["flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=FRAME, hop_length=HOP)))
    feats["centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)))

    rms_full = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms_full + 1e-8)
    feats["pause_ratio"] = float(np.mean(rms_db < (np.max(rms_db) - 40))) * 100

    plt.close("all")
    bg, fg, red = "#161b22", "#c9d1d9", "#da3633"
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=bg)
    for ax in axes.ravel():
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color("#ffffff")

    t = np.arange(len(y)) / sr
    axes[0, 0].plot(t, y, color=red, linewidth=0.5)
    axes[0, 0].set(title="Waveform (loudness over time)", xlabel="Time (s)", ylabel="Amplitude")

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME, hop_length=HOP, n_mels=96),
        ref=np.max
    )
    img = axes[0, 1].imshow(
        mel, origin="lower", aspect="auto", cmap="magma",
        extent=[0, len(y) / sr, 0, sr / 2]
    )
    axes[0, 1].set(title="Mel Spectrogram (frequency content over time)", xlabel="Time (s)", ylabel="Frequency (Hz)")
    cb = fig.colorbar(img, ax=axes[0, 1], format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color=fg)
    plt.setp(cb.ax.get_yticklabels(), color=fg)

    axes[1, 0].plot(times, f0, color=red, linewidth=1.5)
    axes[1, 0].set(title=f"Pitch Contour F0 (first {min(PITCH_MAX_SECONDS, int(len(y)/sr))}s)", xlabel="Time (s)", ylabel="F0 (Hz)")
    axes[1, 0].set_ylim(50, 420)

    mean_db = 10 * np.log10(S.mean(axis=1) + 1e-12)
    axes[1, 1].plot(freqs, mean_db, color=red, linewidth=1)
    axes[1, 1].axvline(4000, color="#8b949e", linestyle="--", linewidth=0.8)
    axes[1, 1].set(title="Average Frequency Spectrum (dashed = 4 kHz)", xlabel="Frequency (Hz)", ylabel="Power (dB)")

    fig.tight_layout()
    return feats, fig


if st.session_state.page == "upload":
    st.markdown("<h1>🎙️ Audio Deep
