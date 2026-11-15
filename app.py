import streamlit as st
from transformers import pipeline
import os
import tempfile
import subprocess

# 1. Page Configuration
st.set_page_config(
    page_title="Audio Deepfake & AI Voice Detector",
    page_icon="🎙️",
    layout="wide"
)

# 2. Clean, Minimal Enterprise Dark & Red Theme CSS
st.markdown("""
    <style>
    /* Dark background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    
    /* Clean Professional Cards */
    .clean-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Red Accent Buttons */
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
    .stButton > button:hover {
        background-color: #b62324;
    }

    /* Status Alerts */
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
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# Sidebar Control Panel
with st.sidebar:
    st.markdown("### 🛡️ **Control Panel**")
    st.markdown("---")
    st.info("🟢 **Engine:** Online\n\n🤖 **Model:** Wav2Vec2\n\n🔒 **Security:** Active")
    st.markdown("---")
    st.markdown("### **Supported Formats**")
    st.markdown("- `.wav`, `.mp3`, `.m4a`\n- `.opus` (WhatsApp)\n- `.mpeg`, `.mp4`, `.mov`")

@st.cache_resource
def load_detector():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

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
        video_extensions = ["mpeg", "mp4", "mov", "avi", "mkv", "mpg"]
        
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
            audio_file_path = None
            temp_input_path = None
            try:
                with st.spinner("Extracting audio and analyzing sample..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_in:
                        temp_in.write(uploaded_file.getbuffer())
                        temp_input_path = temp_in.name
                    
                    if file_extension in video_extensions:
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_audio.close()
                        audio_file_path = temp_audio.name
                        
                        # Extract audio using system ffmpeg via subprocess
                        cmd = ['ffmpeg', '-y', '-i', temp_input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_file_path]
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        audio_file_path = temp_input_path

                    # Run model inference
                    raw_preds = detector(audio_file_path)
                    
                    formatted_preds = []
                    for p in raw_preds:
                        orig_label = p.get("label", "").lower()
                        score = p.get("score", 0.0)
                        if "fake" in orig_label or "spoof" in orig_label or "ai" in orig_label:
                            formatted_preds.append({"label": "AI", "score": score})
                        else:
                            formatted_preds.append({"label": "HUMAN", "score": score})
                            
                    st.session_state.predictions = formatted_preds
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.page = "results"
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                if temp_input_path and os.path.exists(temp_input_path) and temp_input_path != audio_file_path:
                    try:
                        os.remove(temp_input_path)
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
    
    top_label = predictions[0].get("label", "").lower()
    top_score = predictions[0].get("score", 0.0) * 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "ai" in top_label or top_score > 50:
        st.markdown(
            f"""<div class="alert-ai">
            <h3 style="color: #ff7b72; margin: 0; font-weight: 600;">🚨 AI-GENERATED AUDIO DETECTED ({top_score:.2f}%)</h3>
            <p style="color: #ffa198; margin-top: 8px; margin-bottom: 0;">This audio sample exhibits characteristics of synthetic manipulation.</p>
            </div>""", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class="alert-human">
            <h3 style="color: #3fb950; margin: 0; font-weight: 600;">✅ AUTHENTIC HUMAN SPEECH ({top_score:.2f}%)</h3>
            <p style="color: #56d364; margin-top: 8px; margin-bottom: 0;">Spectral signatures match organic human vocal patterns.</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("🔄 Analyze Another File"):
            st.session_state.page = "upload"
            st.session_state.predictions = None
            st.session_state.file_name = ""
            st.rerun()
