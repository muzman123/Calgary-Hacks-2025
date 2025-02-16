#!/usr/bin/env python3
import os
import io
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt

# For saving/loading the colored spectrogram image
from PIL import Image

# ================= User Configuration =================
MODEL_PATH = 'sound_classification_model.h5'

# Folder containing subdirectories named after your classes
# e.g., spectrograms/cat, spectrograms/chainsaw, spectrograms/rain, etc.
SPECTROGRAM_DIR = './spectrograms'

SAMPLE_RATE = 22050        # Microphone sampling rate
RECORD_SECONDS = 5         # <-- Record for 5 seconds
IMG_SIZE = (128, 128)      # Final image size for model
CMAP = 'magma'             # Colormap, e.g. 'magma', 'viridis', etc.

# ================= Load Model =================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ================= Derive Classes from Folder =================
if not os.path.exists(SPECTROGRAM_DIR):
    raise FileNotFoundError(f"Spectrogram folder not found: {SPECTROGRAM_DIR}")

# Each subfolder name is a class label
class_names = sorted(
    d for d in os.listdir(SPECTROGRAM_DIR)
    if os.path.isdir(os.path.join(SPECTROGRAM_DIR, d))
)
# Map subfolder -> index
class_to_index = {name: i for i, name in enumerate(class_names)}
# Reverse mapping: index -> class
index_to_class = {i: name for name, i in class_to_index.items()}
print("Derived class indices from folder structure:", class_to_index)

# ================= Helper Functions =================
def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from microphone for `duration` seconds."""
    print(f"Recording {duration} seconds of audio...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,  # mono
        dtype='float32'
    )
    sd.wait()  # Wait until recording is done
    print("Recording finished.")
    return np.squeeze(recording)

def audio_to_log_mel_spectrogram(audio_data, sr=SAMPLE_RATE, n_mels=128):
    """Convert audio samples to a log-mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def create_color_spectrogram_image(log_mel_spec, sr=SAMPLE_RATE, cmap=CMAP):
    """
    Renders a log-mel spectrogram with labeled axes and a chosen colormap.
    Saves to an in-memory buffer (PNG) and returns a Pillow Image object.
    """
    # 1) Create a figure (slightly larger so labels are visible)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    # 2) Display the spectrogram with time and Mel labels
    librosa.display.specshow(
        log_mel_spec,
        sr=sr,
        x_axis='time',   # <-- Label the x-axis in seconds
        y_axis='mel',    # <-- Label the y-axis in Mel scale
        cmap=cmap,
        ax=ax
    )
    ax.set_title("5-second Audio Clip")
    
    # 3) Save figure to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    
    # 4) Close the figure to free memory
    plt.close(fig)
    
    # 5) Load buffer as a Pillow image
    return Image.open(buf)

def prepare_image_for_model(pil_image, img_size=IMG_SIZE):
    """
    1. Convert to RGB
    2. Resize to img_size
    3. Normalize to [0,1]
    4. Expand dims -> (1, img_size[0], img_size[1], 3)
    """
    # Force RGB
    pil_image = pil_image.convert('RGB')
    # Resize
    pil_image = pil_image.resize(img_size, Image.BICUBIC)
    # Convert to NumPy
    img_array = np.array(pil_image).astype(np.float32)
    # Normalize
    img_array /= 255.0
    # Expand dims
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================= Main Prediction Flow =================
def predict_from_mic():
    """Record audio, make a colored spectrogram, feed to model, print result."""
    # 1) Record (5 seconds, as set above)
    audio_data = record_audio()
    
    # 2) Make log-mel spectrogram
    log_mel_spec = audio_to_log_mel_spectrogram(audio_data, sr=SAMPLE_RATE)
    
    # 3) Create a color PNG in memory (with time + Mel labels)
    color_spec_image = create_color_spectrogram_image(log_mel_spec, sr=SAMPLE_RATE, cmap=CMAP)
    
    # 4) Convert that PIL image into the model input format
    model_input = prepare_image_for_model(color_spec_image, img_size=IMG_SIZE)
    
    # 5) Run the model
    prediction = model.predict(model_input)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = index_to_class.get(predicted_idx, "Unknown")
    
    # 6) Show results
    print("Prediction probabilities:", prediction)
    print(f"Predicted index: {predicted_idx}")
    print(f"Predicted label: {predicted_label}")
    
    # 7) Display the color spectrogram you just generated
    plt.figure(figsize=(5,3))
    plt.imshow(color_spec_image)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

def main():
    print("Starting live microphone classification (single run).")
    predict_from_mic()
    print("Done.")

if __name__ == "__main__":
    main()
