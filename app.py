import streamlit as st
from transformers import pipeline
import os
import tempfile

# 1. Page Configuration
st.set_page_config(
    page_title="Apex Audio Forensics | Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# 2. Advanced Custom CSS for Elite Red University Dashboard Theme
st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #05070b 100%);
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1320;
        border-right: 1px solid #1e2638;
    }

    /* Main Headings */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.5px;
    }
    h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.4rem;
    }
    h1 span {
        color: #ff334b;
    }

    /* Custom Glassmorphism Containers */
    .glass-card {
        background: rgba(20, 27, 45, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 51, 75, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Styled File Uploader Area */
    div[data-testid="stFileUploader"] {
        background: rgba(26, 34, 56, 0.4);
        border: 2px dashed #ff334b;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #ff6b7d;
        background: rgba(26, 34, 56, 0.7);
    }

    /* Premium Red Action Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff334b 0%, #c9182a 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 51, 75, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff4d63. 0%, #e62235 100%);
        box-shadow: 0 6px 20px rgba(255, 51, 75, 0.6);
        transform: translateY(-2px);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 22, 38, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ff334b !important;
        font-weight: 700;
    }

    /* Status badges */
    .badge-secure {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fake {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration Panel
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=70)
    st.markdown("### **System Control Panel**")
    st.markdown("---")
    st.info("🟢 **Engine Status:** Online\n\n🤖 **Model:** Transformers Wav2Vec2\n\n🔒 **Security Level:** Maximum")
    st.markdown("---")
    st.markdown("### **Supported Formats**")
    st.markdown("- `.wav`, `.mp3`, `.m4a`\n- `.opus` (WhatsApp Notes)\n- `.ogg`, `.flac`")
    st.markdown("---")
    st.caption("Developed for Academic & Forensic Research")

# 4. Main App Interface Header
st.markdown("<h1>🛡️ Apex <span>Audio Forensics</span></h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.15rem; margin-bottom: 30px;'>"
    "Next-generation enterprise deepfake voice detection platform. Upload an audio sample "
    "to execute spectral analysis and detect synthetic neural audio manipulation."
    "</p>", 
    unsafe_allow_html=True
)

# 5. Load Deepfake Detection Model
@st.cache_resource
def load_detector():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

with st.spinner("Initializing Deep Neural Classifier... Please stand by."):
    detector = load_detector()

# 6. File Upload Card Layout
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📁 Upload Audio Sample")
uploaded_file = st.file_uploader(
    "Drag and drop your audio target here",
    type=["wav", "mp3", "m4a", "opus", "ogg", "flac"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        audio_file_path = temp_file.name

    # Preview and Action Section in Two Columns
    col_preview, col_action = st.columns([1.2, 1], gap="large")
    
    with col_preview:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎧 Audio Stream Preview")
        st.audio(uploaded_file)
        st.markdown(f"**Filename:** `{uploaded_file.name}`")
        st.markdown(f"**Format:** `.{file_extension.upper()}` | **Size:** `{uploaded_file.size / 1024:.1f} KB`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_action:
        st.markdown('<div class="glass-card" style="text-align: center; padding-top: 35px;">', unsafe_allow_html=True)
        st.subheader("🚀 Execute Scan")
        st.write("Run neural feature extraction on the uploaded voice sample.")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("Run Forensic Analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        with st.spinner("Processing spectral harmonics and inference matrix..."):
            try:
                predictions = detector(audio_file_path)
                
                st.markdown("---")
                st.markdown("<h2>📊 Forensic Classification Report</h2>", unsafe_allow_html=True)
                
                # Metrics layout
                m_col1, m_col2 = st.columns(2)
                
                for idx, pred in enumerate(predictions):
                    label = pred.get("label", "Unknown").upper()
                    score = pred.get("score", 0.0) * 100
                    
                    if idx == 0:
                        with m_col1:
                            st.metric(label="Primary Inference Class", value=f"{label}", delta=f"{score:.2f}% Confidence")
                    else:
                        with m_col2:
                            st.metric(label="Secondary Divergence", value=f"{label}", delta=f"{score:.2f}% Confidence")
                
                # Verdict banner
                top_label = predictions[0].get("label", "").lower()
                top_score = predictions[0].get("score", 0.0) * 100
                
                st.markdown("<br>", unsafe_allow_html=True)
                if "fake" in top_label or "spoof" in top_label or top_score > 50:
                    st.markdown(
                        f"""<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 15px; border-radius: 10px; text-align: center;">
                        <span class="badge-fake">🚨 SYNTHETIC AUDIO DETECTED</span>
                        <p style="color: #fca5a5; margin-top: 8px; margin-bottom: 0;">This file exhibits characteristics of AI voice cloning or synthetic manipulation.</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; padding: 15px; border-radius: 10px; text-align: center;">
                        <span class="badge-secure">✅ AUTHENTIC HUMAN SPEECH</span>
                        <p style="color: #6ee7b7; margin-top: 8px; margin-bottom: 0;">Spectral signatures match natural human vocal patterns.</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    
            except Exception as e:
                st.error(f"Inference pipeline exception encountered: {e}")
                
            finally:
                if os.path.exists(audio_file_path):
                    try:
                        os.remove(audio_file_path)
                    except Exception:
                        pass
