import tensorflow as tf
import numpy as np
import librosa
from keras.models import load_model
import os

# Load the model once when the module is imported
model_path = os.path.join('model', 'model_timedistributed_cnn_lstm2.h5')
model = load_model(model_path)

# Preprocess audio file
def preprocess_audio(file_path):
    sample_rate = 22050
    length = int(sample_rate * 1.0)
    step = int(length * 0.5)
    dBFSThreshold = -96
    
    raw, rate = librosa.load(file_path, sr=sample_rate, mono=True)
    segments = []
    for i in range(0, raw.shape[0] - max(length, step), step):
        column = raw[i:i + length]
        dbFS = 10 * np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -np.inf
        if dbFS > dBFSThreshold:
            segments.append(column)
    
    data = np.array(segments).reshape(len(segments), length, 1)
    return data

# Predict audio class
def predict_audio(file_path):
    data = preprocess_audio(file_path)
    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)
    final_prediction = np.bincount(predicted_classes).argmax()
    return final_prediction