import streamlit as st
from transformers import pipeline
import librosa
import soundfile as sf
import os

st.set_page_config(page_title="Audio Deepfake & AI Voice Detector", page_icon="🎙️")
st.title("🎙️ Audio Deepfake & AI Voice Detector")
st.write("Upload an audio file to instantly analyze whether it is authentic human speech or AI-generated deepfake audio.")

@st.cache_resource
def load_model():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

with st.spinner("Loading AI Deepfake Detection Model... Please wait."):
    detector = load_model()

uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Analyze Audio"):
        with st.spinner("Analyzing audio features..."):
            try:
                results = detector("temp_audio.wav")
                st.success("Analysis Complete!")
                
                for res in results:
                    label = res['label']
                    score = res['score'] * 100
                    st.metric(label=f"Detection: {label}", value=f"{score:.2f}%")
            except Exception as e:
                st.error(f"Error during analysis: {e}")
