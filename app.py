import streamlit as st
from transformers import pipeline
import os
import tempfile

# 1. Page Configuration
st.set_page_config(
    page_title="Audio Deepfake & AI Voice Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Audio Deepfake & AI Voice Detector")
st.write(
    "Upload an audio file (WAV, MP3, M4A, OPUS, OGG, FLAC) or an MP4 video file "
    "to analyze whether the speech is authentic human speech or AI-generated deepfake audio."
)

# 2. Load Deepfake Detection Model
@st.cache_resource
def load_detector():
    return pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")

with st.spinner("Loading AI Detection Model... Please wait."):
    detector = load_detector()

# 3. File Uploader supporting both Audio and Video
uploaded_file = st.file_uploader(
    "Upload Audio or Video File",
    type=["wav", "mp3", "m4a", "opus", "ogg", "flac", "mp4"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        input_file_path = temp_file.name

    audio_file_path = input_file_path
    
    # Render preview and extract audio if file is a video (.mp4)
    if file_extension == "mp4":
        st.video(uploaded_file)
        st.info("Video file detected. Extracting audio track for deepfake analysis...")
        
        try:
            from moviepy.editor import VideoFileClip
            video_clip = VideoFileClip(input_file_path)
            
            # Save extracted audio
            extracted_audio_path = os.path.splitext(input_file_path)[0] + "_extracted.wav"
            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(extracted_audio_path, logger=None)
                video_clip.close()
                audio_file_path = extracted_audio_path
            else:
                st.error("No audio track found in the uploaded MP4 file.")
                video_clip.close()
                audio_file_path = None
        except Exception as e:
            st.error(f"Error extracting audio from video: {e}")
            audio_file_path = None
    else:
        st.audio(uploaded_file)

    # 4. Analyze Button & Inference Execution
    if audio_file_path and st.button("Submit / Analyze"):
        with st.spinner("Analyzing speech patterns and frequencies..."):
            try:
                # Perform AI Inference
                predictions = detector(audio_file_path)
                st.success("Analysis Complete!")
                st.subheader("Results:")
                
                # Display Results
                for pred in predictions:
                    label = pred.get("label", "Unknown")
                    score = pred.get("score", 0.0) * 100
                    st.metric(label=f"Class: {label}", value=f"{score:.2f}%")
                    
            except Exception as e:
                st.error(f"An error occurred during model analysis: {e}")
                
            finally:
                # Clean up temporary extracted audio file if created
                if file_extension == "mp4" and os.path.exists(audio_file_path):
                    try:
                        os.remove(audio_file_path)
                    except Exception:
                        pass

    # Clean up original input temporary file
    if os.path.exists(input_file_path):
        try:
            os.remove(input_file_path)
        except Exception:
            pass
