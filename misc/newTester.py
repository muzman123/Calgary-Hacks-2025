import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
DATASET_PATH = "ESC-50-master"
CSV_PATH = os.path.join(DATASET_PATH, "meta", "esc50.csv")
AUDIO_PATH = os.path.join(DATASET_PATH, "audio")
SR = 22050
N_MELS = 128
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 40

# --------------------------------------------------------------------
# 1. Load CSV metadata and remap target labels
# --------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Get unique target labels in sorted order (from your filtered CSV)
unique_targets = sorted(df['target'].unique())
print("Unique original targets:", unique_targets)

# Create a mapping from original target label to new sequential label
target_mapping = {orig: new for new, orig in enumerate(unique_targets)}
print("Target mapping:", target_mapping)

# Apply the mapping to create a new column with remapped labels
df['new_target'] = df['target'].map(target_mapping)

# Update num_classes based on the new targets
num_classes = len(unique_targets)
print("Number of classes:", num_classes)

# --------------------------------------------------------------------
# 2. Function to convert audio to fixed-size Mel spectrogram
# --------------------------------------------------------------------
def audio_to_melspectrogram(file_path, sr=SR, n_mels=N_MELS, max_len=MAX_LEN):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or trim the spectrogram to have exactly max_len time frames
    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_len]
    
    return mel_spec_db

# --------------------------------------------------------------------
# 3. Build dataset
# --------------------------------------------------------------------
X_data = []
y_data = []

for i in range(len(df)):
    file_path = os.path.join(AUDIO_PATH, df.iloc[i]['filename'])
    # Use the remapped label instead of the original target
    label = df.iloc[i]['new_target']
    
    # Convert audio to Mel spectrogram
    mels = audio_to_melspectrogram(file_path)
    X_data.append(mels)
    y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Add channel dimension: shape becomes (N, 128, 128, 1)
X_data = X_data[..., np.newaxis]

# One-hot encode labels
y_data = to_categorical(y_data, num_classes=num_classes)

# --------------------------------------------------------------------
# 4. Train/test split
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_data, 
    y_data, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_data
)

# --------------------------------------------------------------------
# 5. Define CNN model
# --------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(N_MELS, MAX_LEN, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------------------------------------------
# 6. Train
# --------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# --------------------------------------------------------------------
# 7. Evaluate
# --------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy: {:.2f}%".format(test_acc * 100))

# --------------------------------------------------------------------
# 8. Save model
# --------------------------------------------------------------------
model.save("esc50_cnn_model.h5")
print("Model saved to esc50_cnn_model.h5")
