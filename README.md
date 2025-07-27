# UrbanSound8K Audio Classifier 🎧

This project is a Streamlit web app that classifies environmental sounds from audio files using a deep learning model trained on the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html).

## 🔍 Overview

The app lets users upload a `.wav` file, extracts Mel-Frequency Cepstral Coefficients (MFCCs) using `librosa`, and predicts the sound category using a pre-trained TensorFlow model.

## 🎯 Predicted Classes

The model can recognize the following 10 environmental sound classes:

| Class              | 
|--------------------|
| air_conditioner    | 
| children_playing   | 
| jackhammer         | 
| dog_bark           | 
| drilling           | 
| engine_idling      | 
| street_music       | 
| siren              | 
| car_horn           | 
| gun_shot           | 

## 🚀 Live Demo

Try the app on **Streamlit Cloud**:

👉 [Launch App](https://urbansound8k-classifier.streamlit.app)

> Upload any `.wav` audio file to see it in action!

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Feature extraction: MFCC (40 coefficients)
- Architecture: Feedforward neural network with dropout layers
- Optimizer: Adam
- Loss function: Categorical Crossentropy

## 📁 Project Structure

