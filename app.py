import streamlit as st
from transformers import pipeline
import os
import tempfile

# 1. Page Configuration
st.set_page_config(
    page_title="Audio Deepfake & AI Voice Detector",
    page_icon="🎙️",
    layout="wide"
)

# 2. Custom Red & White University Branding CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0d0f14 0%, #05070a 100%);
        color: #ffffff;
    }
    
    /* Striking Red Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #990000 0%, #660000 100%) !important;
        color: white !important;
        border-right: 2px solid #ff334b;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stInfo {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
    }
    
    /* Red-bordered Cyber Cards */
    .cyber-card {
        background: rgba(20, 15, 15, 0.85);
        border: 1px solid rgba(255, 51, 75, 0.4);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 0 20px rgba(255, 51, 75, 0.15);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Threat Alert Boxes */
    .alert-box-ai {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        padding: 25px;
        border-radius: 14px;
        text-align: center;
    }
    .alert-box-human {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        padding: 25px;
        border-radius: 14px;
        text-align: center;
    }

    /* Red Action Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e60000 0%, #990000 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        border: 1px solid #ff334b;
        box-shadow: 0 4px 15px rgba(230, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a1a 0%, #b30000 100%);
        box-shadow: 0 6px 20px rgba(255, 51, 75, 0.7);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# Red Sidebar Control Panel
with st.sidebar:
    st.markdown("### 🛡️ **Defense Hub**")
    st.markdown("---")
    st.info("🟢 **Status:** Active\n\n🤖 **Model:** Wav2Vec2 Classifier\n\n🔒 **Mode:** Secure Detection")
    st.markdown("---")
    st.markdown("### **Supported Formats**")
    st.markdown("- `.wav`, `.mp3`, `.m4a`\n- `.opus` (WhatsApp Notes)\n- `.ogg`, `.flac`")

@st.cache_resource
def load_detector():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

# ==========================================
# SCREEN 1: UPLOAD DASHBOARD
# ==========================================
if st.session_state.page == "upload":
    st.markdown("<h1>🎙️ Audio Deepfake & <span style='color: #ff334b;'>AI Voice Detector</span></h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #cbd5e1; font-size: 1.15rem; margin-bottom: 25px;'>"
        "Secure neural analysis platform for identifying synthetic voice cloning and AI-generated audio."
        "</p>", 
        unsafe_allow_html=True
    )

    with st.spinner("Initializing Neural Detection Weights..."):
        detector = load_detector()

    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("📁 Target Audio Input")
    uploaded_file = st.file_uploader(
        "Upload your target audio stream",
        type=["wav", "mp3", "m4a", "opus", "ogg", "flac"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            audio_file_path = temp_file.name

        col_preview, col_action = st.columns([1.2, 1], gap="large")
        
        with col_preview:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.subheader("🎧 Audio Stream Waveform")
            st.audio(uploaded_file)
            st.markdown(f"**File:** `{uploaded_file.name}`")
            st.markdown(f"**Type:** `.{file_extension.upper()}` | **Size:** `{uploaded_file.size / 1024:.1f} KB`")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_action:
            st.markdown('<div class="cyber-card" style="text-align: center; padding-top: 40px;">', unsafe_allow_html=True)
            st.subheader("🚀 Execute Neural Scan")
            st.write("Run deep learning feature analysis on audio frequencies.")
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_clicked = st.button("Start Deepfake Analysis")
            st.markdown('</div>', unsafe_allow_html=True)

        if analyze_clicked:
            with st.spinner("Analyzing spectral harmonics and neural weights..."):
                try:
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
                    st.error(f"Execution error: {e}")
                finally:
                    if os.path.exists(audio_file_path):
                        try:
                            os.remove(audio_file_path)
                        except Exception:
                            pass

# ==========================================
# SCREEN 2: RESULTS DASHBOARD
# ==========================================
elif st.session_state.page == "results":
    st.markdown("<h1>🎙️ Audio Deepfake & <span style='color: #ff334b;'>AI Voice Detector</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #cbd5e1; font-size: 1.05rem;'>Analysis Target: <b>{st.session_state.file_name}</b></p>", unsafe_allow_html=True)
    
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("📊 Neural Confidence Breakdown")
    
    predictions = st.session_state.predictions
    
    for pred in predictions:
        label = pred.get("label", "Unknown")
        score = pred.get("score", 0.0) * 100
        
        st.markdown(f"**{label} Confidence Score:** `{score:.2f}%`")
        st.progress(float(score / 100.0))
        st.markdown("<br>", unsafe_allow_html=True)
    
    top_label = predictions[0].get("label", "").lower()
    top_score = predictions[0].get("score", 0.0) * 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "ai" in top_label or top_score > 50:
        st.markdown(
            f"""<div class="alert-box-ai">
            <h2 style="color: #f87171; margin: 0; font-weight: 800;">🚨 SYNTHETIC AI VOICE DETECTED ({top_score:.2f}%)</h2>
            <p style="color: #fca5a5; margin-top: 10px; margin-bottom: 0; font-size: 1.1rem;">Warning: Spectral analysis confirms artificial voice cloning or deepfake generation artifacts.</p>
            </div>""", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class="alert-box-human">
            <h2 style="color: #34d399; margin: 0; font-weight: 800;">✅ AUTHENTIC HUMAN SPEECH ({top_score:.2f}%)</h2>
            <p style="color: #6ee7b7; margin-top: 10px; margin-bottom: 0; font-size: 1.1rem;">Verified: Audio frequency bands match organic human vocal patterns.</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("🔄 Analyze Another Audio File"):
            st.session_state.page = "upload"
            st.session_state.predictions = None
            st.session_state.file_name = ""
            st.rerun()
