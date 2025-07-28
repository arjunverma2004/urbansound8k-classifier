# UrbanSound8K Audio Classifier 🎧

This project is a Streamlit web app that classifies environmental sounds from audio files using a deep learning model trained on the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html).

## 🔍 Overview

The app lets users upload a `.wav or mp3` file, extracts Mel-Frequency Cepstral Coefficients (MFCCs) using `librosa`, and predicts the sound category using a pre-trained TensorFlow model.

## 🚀 Demo

Try the app on **Streamlit Cloud**:

👉 [Launch App](https://urbansound8k-classifier.streamlit.app)

> Upload any `.wav` audio file to see it in action!

![App Screenshot](https://github.com/arjunverma2004/urbansound8k-classifier/blob/main/screenshots/urbansound8k-classifier_ss.png)


---

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


## 🧠 Model Details

- Framework: TensorFlow / Keras
- Feature extraction: MFCC (40 coefficients)
- Architecture: Feedforward neural network with dropout layers
- Optimizer: Adam
- Loss function: Categorical Crossentropy

## 📁 Project Structure
```
urbansound8k-classifier/
│
├── app.py # Streamlit frontend
├── model.keras # Trained classification model
├── labelencoder.pkl # Fitted label encoder
├── requirements.txt # Python dependencies
├── runtime.txt # Python version for Streamlit (optional)
└── README.md # This file
```


## ⚙️ Installation & Local Run


### 📦 Clone the repository

```bash
git clone https://github.com/arjunverma2004/ANN-CustomerChurn-classification.git
cd ANN-CustomerChurn-classification
```

## 🐍 Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 🔧 Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Streamlit App Locally

```bash
streamlit run app.py
```



## 📦 Deployment

This app is deployable on Streamlit Cloud. Ensure the following files are included in the root of your GitHub repo:

- ```app.py```

- ```model.keras```

- ```labelencoder.pkl```

- ```requirements.txt```

## ✍️ Author

**Arjun Verma**

- GitHub: [@arjunverma2004](https://github.com/arjunverma2004)  
- LinkedIn: [Arjun Verma](https://www.linkedin.com/in/arjunverma2004/)
