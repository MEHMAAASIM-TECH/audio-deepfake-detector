import streamlit as st
from transformers import pipeline
import librosa
import soundfile as sf
import os
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="Audio Deepfake & AI Voice Detector", page_icon="🎙️")
st.title("🎙️ Audio & Video Deepfake Detector")
st.write("Upload an audio file or an MP4 video clip to instantly analyze whether the speech is authentic or AI-generated.")

@st.cache_resource
def load_model():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

with st.spinner("Loading AI Deepfake Detection Model... Please wait."):
    detector = load_model()

# Expanded file types to include mp4
uploaded_file = st.file_uploader("Upload Audio or Video File", type=["wav", "mp3", "m4a", "mp4"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Save the uploaded file temporarily
    temp_input_path = f"temp_input.{file_extension}"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio_path = "temp_audio.wav"
    
    # If it's an MP4 video, extract the audio track first
    if file_extension == "mp4":
        st.video(uploaded_file)
        with st.spinner("Extracting audio from video file..."):
            try:
                video_clip = VideoFileClip(temp_input_path)
                video_clip.audio.write_audiofile(audio_path, logger=None)
                video_clip.close()
            except Exception as e:
                st.error(f"Error extracting audio from video: {e}")
    else:
        st.audio(uploaded_file)
        # For standard audio files, copy/rename to temp_audio.wav
        os.rename(temp_input_path, audio_path)
        
    if st.button("Submit"):
        with st.spinner("Analyzing audio features..."):
            try:
                results = detector(audio_path)
                st.success("Analysis Complete!")
                
                for res in results:
                    label = res['label']
                    score = res['score'] * 100
                    st.metric(label=f"Detection: {label}", value=f"{score:.2f}%")
            except Exception as e:
                st.error(f"Error during analysis: {e}")
