import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define paths (update these paths if needed)
audio_dir = '/Users/hamzakhurram/Downloads/ESC-50-master/audio'
csv_path = '/Users/hamzakhurram/Downloads/ESC-50-master/meta/esc50.csv'
output_dir = '/Users/hamzakhurram/Downloads/ESC-50-master/spectrograms'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV file to get filenames and labels
df = pd.read_csv(csv_path)

# Parameters for the spectrogram dimensions
target_height = 128   # Number of mel bands
hop_length = 512      # Define the hop length (default in librosa)
# For 5 seconds at a sample rate of 22050:
target_width = int(5 * 22050 / hop_length)  # ~215 frames

# Loop over each audio file
for idx, row in df.iterrows():
    filename = row['filename']
    label = row['category']  # e.g., "dog", "gunshot", etc.
    file_path = os.path.join(audio_dir, filename)
    
    # Check if the audio file exists; if not, skip it.
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}. Skipping.")
        continue
    
    # Create a subfolder for the label if it doesn't exist
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    try:
        # Load the audio file using a fixed sample rate (22050 Hz) for consistency
        y, sr = librosa.load(file_path, sr=22050)
        
        # Generate a mel spectrogram with 128 mel bands using the defined hop_length
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_height, hop_length=hop_length)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Adjust the spectrogram to a fixed shape (target_height x target_width)
        time_steps = S_DB.shape[1]
        if time_steps < target_width:
            pad_width = target_width - time_steps
            S_DB = np.pad(S_DB, ((0, 0), (0, pad_width)), mode='constant')
        elif time_steps > target_width:
            S_DB = S_DB[:, :target_width]
        
        # Plot the spectrogram with proper time scaling (using hop_length)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.title(f'Mel Spectrogram - {label}')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        # Save the plot as a PNG image
        output_file = os.path.join(label_dir, filename.replace('.wav', '.png'))
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved spectrogram for {filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
