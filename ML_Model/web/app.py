#!/usr/bin/env python3
import os
# Force Agg backend _before_ importing matplotlib
os.environ['MPLBACKEND'] = 'Agg'

import io
import time
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, request, url_for
import json

# ================= Flask App Configuration =================
app = Flask(__name__)
# Folder for storing spectrogram images to be served as static files
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'spectrograms')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ================= User Configuration =================
MODEL_PATH = 'sound_classification_model.h5'
# Instead of deriving from folder, load the mapping saved during training
JSON_PATH = 'class_indices.json'
SAMPLE_RATE = 22050        # Microphone sampling rate
RECORD_SECONDS = 5         # Record for 5 seconds
IMG_SIZE = (128, 128)      # Final image size for model input
CMAP = 'magma'             # Colormap

# ================= Load Model =================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ================= Load Class Indices Mapping =================
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"JSON mapping file not found: {JSON_PATH}")
with open(JSON_PATH, "r") as f:
    class_indices = json.load(f)
# Create reverse mapping: index -> class
indices_to_class = {v: k for k, v in class_indices.items()}
print("Loaded class indices mapping:", class_indices)

# ================= Audio Processing Functions =================
def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from the microphone for the specified duration."""
    print(f"Recording {duration} seconds of audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return np.squeeze(recording)

def audio_to_log_mel_spectrogram(audio_data, sr=SAMPLE_RATE, n_mels=128):
    """Convert audio samples to a log-mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def create_color_spectrogram_image(log_mel_spec, sr=SAMPLE_RATE, cmap=CMAP):
    """
    Render a log-mel spectrogram with time and Mel-axis labels using matplotlib.
    The figure is saved to an in-memory buffer and returned as a Pillow Image.
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    librosa.display.specshow(
        log_mel_spec,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap=cmap,
        ax=ax
    )
    ax.set_title("5-second Audio Clip")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def prepare_image_for_model(pil_image, img_size=IMG_SIZE):
    """
    Convert a Pillow image to RGB, resize to img_size, normalize to [0,1],
    and add a batch dimension.
    """
    pil_image = pil_image.convert('RGB')
    pil_image = pil_image.resize(img_size, Image.BICUBIC)
    img_array = np.array(pil_image).astype(np.float32)
    img_array /= 255.0
    return np.expand_dims(img_array, axis=0)

def predict_from_audio():
    """
    Record audio, generate a colored spectrogram image, preprocess it for the model,
    perform prediction, and save the image to the static folder.
    Returns the predicted label and the filename.
    """
    audio_data = record_audio()
    log_mel_spec = audio_to_log_mel_spectrogram(audio_data, sr=SAMPLE_RATE)
    color_spec_image = create_color_spectrogram_image(log_mel_spec, sr=SAMPLE_RATE, cmap=CMAP)
    model_input = prepare_image_for_model(color_spec_image, img_size=IMG_SIZE)
    prediction = model.predict(model_input)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = indices_to_class.get(predicted_idx, "Unknown")
    print("Prediction probabilities:", prediction)
    print("Predicted index:", predicted_idx)
    print("Predicted label:", predicted_label)
    filename = f"spectrogram_{int(time.time())}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    color_spec_image.save(filepath)
    return predicted_label, filename

# ================= Flask Routes =================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        predicted_label, filename = predict_from_audio()
        image_url = url_for('static', filename=f'spectrograms/{filename}')
        return render_template('index.html', predicted_label=predicted_label, image_url=image_url)
    else:
        return render_template('index.html', predicted_label=None, image_url=None)

# ================= Main =================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
