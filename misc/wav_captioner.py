import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd

# --- Parameters ---
DURATION = 5         # seconds to record
FS = 44100           # sampling rate (ESC-50 is recorded at 44.1 kHz)
N_MELS = 128         # number of mel bands to generate
FIXED_WIDTH = 128    # fixed time dimension used during training
CSV_PATH = "ESC-50-master/meta/esc50.csv"  # Path to your ESC-50 CSV file

# --- Additional Spectrogram Parameters ---
N_FFT = 1024         # FFT window size (try lowering for more time resolution)
HOP_LENGTH = 256     # hop length between successive frames
TOP_DB = 80          # top decibel threshold for the spectrogram

# --- Step 1: Record Audio ---
def record_audio(duration, fs):
    print(f"Recording {duration} seconds of audio...")
    # Record with 1 channel (mono) at float32 precision
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio = audio.flatten()  # Convert from 2D array to 1D
    print("Recording complete.")
    return audio

# --- Step 2: Generate Mel Spectrogram ---
def compute_mel_spectrogram(audio, fs, n_mels):
    # Compute mel spectrogram using the specified FFT parameters
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=fs,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # Convert power spectrogram to decibel (dB) units with a dynamic range limit
    S_dB = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    return S_dB

# --- Step 3: Adjust Spectrogram to Fixed Size ---
def adjust_spectrogram(spectrogram, fixed_width):
    current_width = spectrogram.shape[1]
    if current_width < fixed_width:
        pad_width = fixed_width - current_width
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif current_width > fixed_width:
        spectrogram = spectrogram[:, :fixed_width]
    return spectrogram

# --- Step 4: Load the Pre-trained CNN Model ---
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# --- Step 5: Build Mapping from Class Index to Caption ---
def build_class_mapping(csv_path):
    df = pd.read_csv(csv_path)
    mapping = {target: category for target, category in sorted(set(zip(df['target'], df['category'])))}
    return mapping

# --- Main Function ---
def main():
    # Record audio and plot the waveform for debugging
    audio = record_audio(DURATION, FS)
    plt.figure(figsize=(10, 3))
    plt.plot(audio)
    plt.title("Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Compute the mel spectrogram
    mel_spec = compute_mel_spectrogram(audio, FS, N_MELS)
    print("Mel spectrogram shape before adjust:", mel_spec.shape)
    
    # Adjust the spectrogram to match the training shape
    mel_spec = adjust_spectrogram(mel_spec, FIXED_WIDTH)
    
    # Display the adjusted mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=FS, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram (Adjusted)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
    # Prepare input for the CNN: (batch, height, width, channels)
    input_img = np.expand_dims(mel_spec, axis=-1)  # shape: (n_mels, fixed_width, 1)
    input_img = np.expand_dims(input_img, axis=0)    # shape: (1, n_mels, fixed_width, 1)
    
    # Load the pre-trained model and run inference
    model = load_model("esc50_cnn_model.h5")
    prediction = model.predict(input_img)
    predicted_index = np.argmax(prediction, axis=-1)[0]
    
    # Map predicted index to a caption
    class_mapping = build_class_mapping(CSV_PATH)
    caption = class_mapping.get(predicted_index, "Unknown")
    
    print("Prediction (class index):", predicted_index)
    print("Prediction caption:", caption)
    print("Raw prediction probabilities:", prediction)

if __name__ == "__main__":
    main()
