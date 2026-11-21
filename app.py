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
TARGET_SR = 16000          # Wav2Vec2 expects 16 kHz mono audio
MAX_SECONDS = 45           # analyse at most first 45 s for faster cloud response
PITCH_MAX_SECONDS = 25     # cap pYIN pitch tracking to keep CPU time predictable
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


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def convert_to_wav(in_path, out_path):
    """Convert audio/video file to 16 kHz mono PCM WAV with FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", in_path, "-vn", "-t", str(MAX_SECONDS),
        "-af", "highpass=f=80,loudnorm",
        "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1", out_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in PATH (check packages.txt).")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out while converting the file.")

    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg could not extract audio: {res.stderr.strip()[-250:]}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("No usable audio track was found in this file.")


def split_scores(raw_preds):
    """Return (ai_score, human_score) safely from model predictions."""
    if not raw_preds:
        return 0.5, 0.5

    ai, human = None, None
    for p in raw_preds:
        lab = str(p.get("label", "")).lower().strip()
        score = float(p.get("score", 0.0))
        if any(k in lab for k in ("fake", "spoof", "synthetic")) or lab in ("ai", "ai-generated"):
            ai = score
        elif any(k in lab for k in ("real", "
