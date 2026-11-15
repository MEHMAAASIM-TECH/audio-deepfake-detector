import streamlit as st
from transformers import pipeline
import os
import tempfile

# 1. Page Configuration & Custom Red Branding Styling
st.set_page_config(
    page_title="Audio Deepfake & AI Voice Detector",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for Red University Theme & Professional UI
st.markdown("""
    <style>
    /* Main background & font styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Title styling */
    h1 {
        color: #ff4b4b;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: #e0e0e0;
    }

    /* File uploader container custom styling */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #ff4b4b;
        border-radius: 10px;
        padding: 15px;
        background-color: rgba(255, 75, 75, 0.03);
    }
    
    /* Custom buttons styled with red university theme */
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #e03232;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }

    /* Metric card formatting */
    div[data-testid="stMetric"] {
        background-color: #1a1c23;
        border: 1px solid #2d3139;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"] label {
        color: #a0aab5 !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ff4b4b !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# 2. App Header & Description
st.title("🎙️ Audio Deepfake & AI Voice Detector")
st.markdown(
    "<p style='color: #b0b8c4; font-size: 1.1rem;'>"
    "An enterprise-grade forensic tool leveraging deep learning transformers to analyze "
    "audio authenticity and detect synthetic voice generation."
    "</p>", 
    unsafe_allow_html=True
)

st.markdown("---")

# 3. Load Deepfake Detection Model
@st.cache_resource
def load_detector():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

with st.spinner("Initializing Deep Learning Model... Please wait."):
    detector = load_detector()

# 4. File Uploader Section
st.subheader("📁 Input Audio File")
uploaded_file = st.file_uploader(
    "Drag and drop your audio sample here (WAV, MP3, M4A, OPUS, OGG, FLAC)",
    type=["wav", "mp3", "m4a", "opus", "ogg", "flac"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        audio_file_path = temp_file.name

    st.markdown("### 🎧 Audio Preview")
    st.audio(uploaded_file)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled Submit Button inside columns for neat sizing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("🚀 Run Deepfake Analysis")

    if analyze_clicked:
        with st.spinner("Extracting spectral features and running inference..."):
            try:
                predictions = detector(audio_file_path)
                st.markdown("---")
                st.subheader("📊 Forensic Analysis Report")
                
                # Display Results in organized metrics
                res_col1, res_col2 = st.columns(2)
                
                for idx, pred in enumerate(predictions):
                    label = pred.get("label", "Unknown").upper()
                    score = pred.get("score", 0.0) * 100
                    
                    if idx == 0:
                        with res_col1:
                            st.metric(label=f"Primary Classification", value=f"{label}", delta=f"{score:.2f}% Confidence")
                    else:
                        with res_col2:
                            st.metric(label=f"Alternative Class", value=f"{label}", delta=f"{score:.2f}% Confidence")
                            
                st.success("Analysis complete. Voice characteristics mapped successfully against neural weights.")
                    
            except Exception as e:
                st.error(f"An error occurred during model inference: {e}")
                
            finally:
                if os.path.exists(audio_file_path):
                    try:
                        os.remove(audio_file_path)
                    except Exception:
                        pass
