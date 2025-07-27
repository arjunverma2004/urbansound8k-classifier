import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import os

# Load the trained model and label encoder
model = tf.keras.models.load_model("model.keras")
labelencoder = pickle.load(open("labelencoder.pkl", "rb"))

# Function to extract features from uploaded audio file
def extract_features(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')  # mono by default
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Streamlit UI
st.title("UrbanSound8K Audio Classifier 🎧")
st.write("Upload a `.wav` file to classify the type of sound.")

uploaded_file = st.file_uploader("Choose a WAV audio file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Processing and predicting..."):
        try:
            features = extract_features(uploaded_file)
            features = np.expand_dims(features, axis=0)  # Reshape for prediction

            prediction = model.predict(features)
            predicted_label = labelencoder.inverse_transform([np.argmax(prediction)])

            st.success(f"Predicted Class: **{predicted_label[0]}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
