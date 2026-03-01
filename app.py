import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
import io
import os
import google.generativeai as genai

# --- Load Gemini API key from .streamlit/secrets.toml ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- Configure Gemini model ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    chat = model_gemini.start_chat(history=[
        {
            "role": "user",
            "parts": ["""
    You are a smart AI assistant inside a deepfake audio detection web app.
    Your primary job is to help users understand the model, audio processing, and deep learning concepts.

    However, you can also answer general knowledge questions, explain tech concepts, or respond helpfully to any user query unless it's harmful or inappropriate.

    The model is a CNN-LSTM hybrid trained on MFCC features from 2-second audio clips.
    It performs binary classification (Real or Fake) with 95.2% test accuracy.
    """]
        },
        {
            "role": "model",
            "parts": ["Got it! I'm ready to assist with deepfake detection, AI concepts, and general questions."]
        }
    ])


else:
    model_gemini = None

# --- Custom UI ---
st.markdown("""
    <style>
    .stApp { background-color: #001F3F; color: white; }
    .header-container { display: flex; align-items: center; justify-content: center; gap: 15px; }
    .mic-symbol { font-size: 80px; color: silver; }
    .title { color: yellow; font-size: 50px; font-weight: bold; text-shadow: 2px 2px 6px black; }
    .subheader { color: white; font-size: 26px; text-align: center; font-weight: bold; padding-bottom: 10px; }
    .file-name { font-size: 22px; color: white; font-weight: bold; text-align: center; }
    .spectrogram-box { background-color: #002244; padding: 10px; border-radius: 8px; margin-top: 20px; text-align: center; }
    .spectrogram-text { color: white !important; font-size: 24px; font-weight: bold !important; text-align: center; text-transform: uppercase; }
    .prediction-box { background-color: #003366; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid white; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Load model (SavedModel format folder) ---
@st.cache_resource
def load_model():
    return tf.saved_model.load("cnn_lstm_model_final.keras")

model = load_model()

# --- Extract features from audio ---
def extract_features(audio_data, sr=16000):
    y, _ = librosa.load(io.BytesIO(audio_data), sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfccs.shape[1] < 128:
        pad_width = 128 - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
    mfccs = mfccs[:, :128]

    features = np.expand_dims(mfccs, axis=0)  # ✅ (1, 40, 128)
    return tf.constant(features, dtype=tf.float32)

# --- Generate spectrogram image ---
def generate_spectrogram(audio_data, sr=16000):
    y, _ = librosa.load(io.BytesIO(audio_data), sr=sr)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# --- Header ---
st.markdown("""
    <div class="header-container">
        <span class="mic-symbol">🎤</span>
        <span class="title">Deepfake Audio Detector</span>
    </div>
    <p class="subheader">Upload an audio file to detect whether it is real or fake</p>
""", unsafe_allow_html=True)

# --- File upload ---
uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        st.markdown(f"<div class='file-name'>📁 Uploaded File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)</div>", unsafe_allow_html=True)
        st.audio(uploaded_file, format="audio/wav")

        audio_bytes = uploaded_file.read()
        X_input = extract_features(audio_bytes)

        # --- Inference ---
        infer = model.signatures["serving_default"]
        prediction_result = infer(keras_tensor_9=X_input)
        prediction = list(prediction_result.values())[0].numpy()[0][0]

        is_fake = prediction > 0.5
        label_color = "red" if is_fake else "lime"
        label_text = f"<span style='color:{label_color}; font-size:40px; font-weight:bold;'>{'🔴 Fake' if is_fake else '🟢 Real'}</span>"
        confidence = max(prediction, 1 - prediction) * 100
        confidence_color = "red" if is_fake else "lime"

        st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color:yellow;'>Prediction: {label_text}</h2>
                <h3 style='color:{confidence_color}; font-size:30px; font-weight:bold;'>Confidence: {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        # --- Spectrogram ---
        st.markdown("""
            <div class='spectrogram-box'>
                <h2 class='spectrogram-text'>Spectrogram</h2>
            </div>
        """, unsafe_allow_html=True)
        spectrogram_img = generate_spectrogram(audio_bytes)
        st.image(spectrogram_img, caption="Spectrogram")

    except Exception as e:
        st.error(f"❌ Error processing audio:\n\n{str(e)}")

# --- Chatbot section ---
st.markdown("---")
st.subheader("💬 Ask the Deepfake Chatbot")
prompt = st.text_input("Type your question here:")

if st.button("Ask") and prompt:
    if model_gemini:
        try:
            response = chat.send_message(prompt)
            st.success(response.text)
        except Exception as e:
            st.error(f"❌ Gemini Error: {str(e)}")
    else:
        st.error("⚠️ Gemini API key not found. Please add it to `.streamlit/secrets.toml`")


