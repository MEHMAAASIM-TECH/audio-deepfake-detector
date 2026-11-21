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
MAX_SECONDS = 60           # analyse at most the first 60 s (speed / memory)
HOP = 256
FRAME = 1024

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
        help="Leave unchecked for Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (its 'fake' and 'real' labels are already mapped correctly)."
    )
    st.markdown("---")
    st.markdown("### **Supported Formats**")
    st.markdown("- `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`\n- `.opus` (WhatsApp)\n- `.mpeg`, `.mp4`, `.mov`")


@st.cache_resource
def load_detector():
    return pipeline(
        "audio-classification",
        model="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
    )


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def convert_to_wav(in_path, out_path):
    """Convert ANY audio/video file to 16 kHz mono PCM WAV with FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", in_path, "-vn", "-t", str(MAX_SECONDS),
        "-af", "highpass=f=80,loudnorm",
        "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1", out_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in PATH.")
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg could not extract audio (no audio track or corrupt file). "
                           f"Details: {res.stderr.strip()[-300:]}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("No usable audio was found in this file.")


def split_scores(raw_preds):
    """Return (ai_score, human_score) from the model's raw output (before any inversion)."""
    ai, human = None, None
    for p in raw_preds:
        lab = p.get("label", "").lower()
        score = float(p.get("score", 0.0))
        if any(k in lab for k in ("fake", "spoof", "synthetic")) or lab in ("ai", "ai-generated"):
            ai = score
        elif any(k in lab for k in ("real", "bonafide", "human", "genuine", "authentic")):
            human = score
    if ai is None and human is None:          # unknown label names -> first label treated as 'AI'
        ai = float(raw_preds[0].get("score", 0.0))
        human = 1.0 - ai
    elif ai is None:
        ai = 1.0 - human
    elif human is None:
        human = 1.0 - ai
    return ai, human


def extract_features_and_plots(y, sr):
    """Compute simple acoustic measurements + build the 4 graphs."""
    if len(y) < sr * 0.5:
        raise ValueError("Audio is too short (need at least 0.5 s).")

    # --- Pitch (fundamental frequency) with pYIN ---
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr,
                                 frame_length=FRAME, hop_length=HOP)
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms, times = f0[:n], voiced[:n], rms[:n], times[:n]

    f0_v = f0[voiced & ~np.isnan(f0)]
    feats = {}
    feats["voiced_ratio"] = float(np.mean(voiced)) * 100
    if len(f0_v) > 10:
        feats["pitch_mean"] = float(np.mean(f0_v))
        feats["pitch_std_semitones"] = float(np.std(12 * np.log2(f0_v / np.median(f0_v))))
        # Frame-level (approximate) jitter and shimmer on consecutive voiced frames
        idx = np.where(voiced[:-1] & voiced[1:] & ~np.isnan(f0[:-1]) & ~np.isnan(f0[1:]))[0]
        if len(idx) > 5:
            periods = 1.0 / f0
            feats["jitter"] = float(np.mean(np.abs(periods[idx + 1] - periods[idx])) / np.mean(periods[idx]) * 100)
            feats["shimmer"] = float(np.mean(np.abs(rms[idx + 1] - rms[idx])) / (np.mean(rms[idx]) + 1e-9) * 100)
    # --- Spectral measurements ---
    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME)
    feats["hf_energy_pct"] = float(S[freqs >= 4000].sum() / (S.sum() + 1e-12) * 100)
    feats["flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=FRAME, hop_length=HOP)))
    feats["centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)))
    # --- Pauses ---
    rms_db = 20 * np.log10(rms + 1e-8)
    feats["pause_ratio"] = float(np.mean(rms_db < (rms_db.max() - 40))) * 100

    # --- Graphs ---
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

    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME, hop_length=HOP, n_mels=96), ref=np.max)
    img = axes[0, 1].imshow(mel, origin="lower", aspect="auto", cmap="magma",
                            extent=[0, len(y) / sr, 0, sr / 2])
    axes[0, 1].set(title="Mel Spectrogram (frequency content over time)", xlabel="Time (s)", ylabel="Frequency (Hz)")
    cb = fig.colorbar(img, ax=axes[0, 1], format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color=fg)
    plt.setp(cb.ax.get_yticklabels(), color=fg)

    axes[1, 0].plot(times, f0, color=red, linewidth=1.5)
    axes[1, 0].set(title="Pitch Contour (fundamental frequency F0)", xlabel="Time (s)", ylabel="F0 (Hz)")
    axes[1, 0].set_ylim(50, 420)

    mean_db = 10 * np.log10(S.mean(axis=1) + 1e-12)
    axes[1, 1].plot(freqs, mean_db, color=red, linewidth=1)
    axes[1, 1].axvline(4000, color="#8b949e", linestyle="--", linewidth=0.8)
    axes[1, 1].set(title="Average Frequency Spectrum (dashed = 4 kHz)", xlabel="Frequency (Hz)", ylabel="Power (dB)")

    fig.tight_layout()
    return feats, fig


# ==========================================
# SCREEN 1: UPLOAD DASHBOARD
# ==========================================
if st.session_state.page == "upload":
    st.markdown("<h1>🎙️ Audio Deepfake & <span style='color: #da3633;'>AI Voice Detector</span></h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #8b949e; font-size: 1.05rem; margin-bottom: 25px;'>"
        "Upload an audio or video file (.mpeg, .mp4, etc.) to analyze speech authenticity."
        "</p>",
        unsafe_allow_html=True
    )

    with st.spinner("Loading AI Detection Model..."):
        detector = load_detector()

    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.subheader("📁 Upload Audio or Video File")
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=["wav", "mp3", "m4a", "opus", "ogg", "flac", "mpeg", "mp4", "mov"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        video_extensions = ["mpeg", "mp4", "mov"]

        col_preview, col_action = st.columns([1.2, 1], gap="large")

        with col_preview:
            st.markdown('<div class="clean-card">', unsafe_allow_html=True)
            if file_extension in video_extensions:
                st.subheader("🎬 Video Preview")
                st.video(uploaded_file)
            else:
                st.subheader("🎧 Audio Preview")
                st.audio(uploaded_file)
            st.markdown(f"**File:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_action:
            st.markdown('<div class="clean-card" style="text-align: center; padding-top: 30px;">', unsafe_allow_html=True)
            st.subheader("🚀 Run Analysis")
            st.write("Click below to process and analyze speech patterns.")
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_clicked = st.button("Start Analysis")
            st.markdown('</div>', unsafe_allow_html=True)

        if analyze_clicked:
            temp_input_path = None
            wav_path = None
            try:
                with st.spinner("Extracting audio, running model and measuring acoustic features..."):
                    # 1) save upload to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_in:
                        temp_in.write(uploaded_file.getbuffer())
                        temp_input_path = temp_in.name

                    # 2) convert everything to 16 kHz mono WAV
                    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp_wav.close()
                    wav_path = tmp_wav.name
                    convert_to_wav(temp_input_path, wav_path)

                    # 3) load samples once; used by the model AND the graphs
                    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

                    # Trim leading/trailing silence and normalize peak amplitude
                    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
                    if len(y_trimmed) >= int(sr * 0.5):
                        y = y_trimmed
                    peak = np.max(np.abs(y))
                    if peak > 0:
                        y = y / peak * 0.9

                    # 4) model inference
                    raw_preds = detector({"raw": y, "sampling_rate": sr}, top_k=2)
                    raw_ai, raw_human = split_scores(raw_preds)

                    if invert_labels:           # calibration switch (see sidebar)
                        ai_score, human_score = raw_human, raw_ai
                    else:
                        ai_score, human_score = raw_ai, raw_human

                    formatted_preds = [
                        {"label": "AI", "score": ai_score},
                        {"label": "HUMAN", "score": human_score},
                    ]
                    formatted_preds.sort(key=lambda x: x["score"], reverse=True)

                    # 5) acoustic features + graphs
                    feats, fig = extract_features_and_plots(y, sr)

                    st.session_state.predictions = formatted_preds
                    st.session_state.raw_predictions = raw_preds
                    st.session_state.features = feats
                    st.session_state.figure = fig
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.page = "results"
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                for path in {temp_input_path, wav_path}:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass

# ==========================================
# SCREEN 2: RESULTS DASHBOARD
# ==========================================
elif st.session_state.page == "results":
    st.markdown("<h1>🎙️ Audio Deepfake & <span style='color: #da3633;'>AI Voice Detector</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #8b949e; font-size: 1rem;'>Analyzed File: <b>{st.session_state.file_name}</b></p>", unsafe_allow_html=True)

    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.subheader("📊 Confidence Breakdown")

    predictions = st.session_state.predictions

    col1, col2 = st.columns(2)
    for idx, pred in enumerate(predictions):
        label = pred.get("label", "Unknown")
        score = pred.get("score", 0.0) * 100
        target_col = col1 if idx == 0 else col2
        with target_col:
            st.markdown(f"""
                <div style="background: #21262d; border: 1px solid #30363d; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 15px;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #8b949e; margin-bottom: 5px;">{label} Confidence</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #da3633;">{score:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

    top_label = predictions[0].get("label", "").upper()
    top_score = predictions[0].get("score", 0.0) * 100
    second_score = predictions[1].get("score", 0.0) * 100
    margin = top_score - second_score

    # Certainty tiers (simple rule-based thresholds)
    if top_score >= 85.0:
        certainty_text = "High Certainty"
    elif top_score >= 65.0:
        certainty_text = "Moderate Certainty"
    else:
        certainty_text = "Low / Borderline (compression or background noise may affect the result)"

    st.markdown("<br>", unsafe_allow_html=True)

    is_ai = top_label == "AI" and top_score > 50
    if is_ai:
        st.markdown(
            f"""<div class="alert-ai">
            <h3 style="color: #ff7b72; margin: 0; font-weight: 600;">🚨 AI-GENERATED AUDIO DETECTED ({top_score:.2f}%)</h3>
            <p style="color: #ffa198; margin-top: 8px; margin-bottom: 0;">The model classifies this sample as synthetic speech.</p>
            </div>""",
            unsafe_allow_html=True
        )
        verdict_html = "<b>AI-Generated / Synthetic Speech</b>"
        border = "#da3633"
    else:
        st.markdown(
            f"""<div class="alert-human">
            <h3 style="color: #3fb950; margin: 0; font-weight: 600;">✅ AUTHENTIC HUMAN SPEECH ({top_score:.2f}%)</h3>
            <p style="color: #56d364; margin-top: 8px; margin-bottom: 0;">The model classifies this sample as natural human speech.</p>
            </div>""",
            unsafe_allow_html=True
        )
        verdict_html = "<b>Authentic Human Speech</b>"
        border = "#2ea043"

    st.markdown(f"""
        <div class="summary-box" style="border-left-color: {border};">
            <h4 style="color: #ffffff; margin-top: 0; margin-bottom: 12px;">📋 Diagnostic Summary</h4>
            <ul style="color: #c9d1d9; line-height: 1.6; margin-bottom: 0;">
                <li><b>Verdict:</b> {verdict_html}, ahead of the other class by <b>{margin:.2f}</b> percentage points.</li>
                <li><b>Certainty level:</b> {certainty_text}.</li>
                <li><b>How the verdict is made:</b> the Wav2Vec2 neural network outputs a probability for each class. The graphs and measurements below are <i>supporting acoustic evidence</i> computed from your file; they are not the direct input to the verdict.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Acoustic analysis ----------------
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.subheader("📈 Acoustic Analysis (Pitch, Frequency & Details)")
    st.pyplot(st.session_state.figure)

    f = st.session_state.features
    st.markdown("#### Measured features")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean pitch (F0)", f"{f['pitch_mean']:.0f} Hz" if "pitch_mean" in f else "n/a")
    m2.metric("Pitch variation", f"{f['pitch_std_semitones']:.2f} st" if "pitch_std_semitones" in f else "n/a",
              help="Standard deviation of pitch in semitones. Very low values mean a flat, monotone voice.")
    m3.metric("Jitter (approx.)", f"{f['jitter']:.2f} %" if "jitter" in f else "n/a",
              help="Cycle-to-cycle pitch instability. Natural voices always have some.")
    m4.metric("Shimmer (approx.)", f"{f['shimmer']:.2f} %" if "shimmer" in f else "n/a",
              help="Frame-to-frame loudness instability within voiced speech.")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Voiced speech", f"{f['voiced_ratio']:.0f} %")
    m6.metric("Pauses / silence", f"{f['pause_ratio']:.0f} %",
              help="Share of frames that are >40 dB below the loudest part.")
    m7.metric("Energy above 4 kHz", f"{f['hf_energy_pct']:.2f} %",
              help="Share of total energy in the high-frequency band, where vocoder artefacts often appear.")
    m8.metric("Spectral flatness", f"{f['flatness']:.4f}",
              help="Closer to 1 = noise-like, closer to 0 = tonal.")

    st.caption(
        "⚠️ These are descriptive measurements, not proof. Lower pitch variation, very regular jitter/shimmer "
        "or unusual energy above 4 kHz are patterns that *can* accompany synthetic speech, but compressed or "
        "noisy real recordings can show them too, and modern voice clones can avoid them. "
        "There is no fixed threshold; the neural network verdict above is the decision."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 Raw model output (use this to verify the label inversion)"):
        st.json(st.session_state.raw_predictions)
        st.caption("If a clip you KNOW is AI shows the 'fake/spoof' label with the higher score here, "
                   "the labels are already correct: turn off 'Invert model labels' in the sidebar.")

    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("🔄 Analyze Another File"):
            plt.close("all")
            st.session_state.page = "upload"
            st.session_state.predictions = None
            st.session_state.raw_predictions = None
            st.session_state.features = None
            st.session_state.figure = None
            st.session_state.file_name = ""
            st.rerun()
