# model.py
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model(model_path="sound_model.pkl"):
    """
    Train a Random Forest classifier with three classes:
      Class 0: Calm Forest Ambience (30-50 dB)
      Class 1: Possible Human Activity (52-66 dB)
      Class 2: Chainsaw/Logging Activity (spiky values >70 dB)
    """
    window_size = 10   # number of samples per window
    n_windows = 500    # total number of windows per class

    data = []
    labels = []
    
    # Class 0: Calm Forest Ambience
    for _ in range(n_windows):
        # Simulate readings around 40 dB with low variation.
        window = np.random.normal(loc=40, scale=3, size=window_size)
        # Clip to a plausible range (30-50 dB)
        window = np.clip(window, 30, 50)
        data.append(window)
        labels.append(0)
        
    # Class 1: Possible Human Activity
    for _ in range(n_windows):
        # Simulate readings around 60 dB with low variation.
        window = np.random.normal(loc=60, scale=2, size=window_size)
        # Ensure values roughly fall in the 52-66 dB range.
        window = np.clip(window, 52, 66)
        data.append(window)
        labels.append(1)
        
    # Class 2: Chainsaw/Logging Activity
    for _ in range(n_windows):
        # Start with a base window similar to human activity.
        window = np.random.normal(loc=60, scale=2, size=window_size)
        window = np.clip(window, 52, 66)
        # Introduce a spike: randomly pick an index and replace it with a high value.
        spike_index = np.random.randint(0, window_size)
        window[spike_index] = np.random.normal(loc=90, scale=5)  # spike well above 70 dB
        data.append(window)
        labels.append(2)
    
    data = np.array(data)
    
    # Extract features: mean, std, min, and max for each window.
    features = []
    for window in data:
        features.append([np.mean(window), np.std(window), np.min(window), np.max(window)])
    features = np.array(features)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    
    # Save the trained model.
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print("Model trained and saved to", model_path)
    
    return clf

def load_model(model_path="sound_model.pkl"):
    """Load a pre-trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded from", model_path)
    return model

def extract_features(window):
    """
    Given a window (list or numpy array of decibel readings),
    extract features: mean, std, min, and max.
    Returns a 2D array (1 x 4) suitable for classification.
    """
    mean_val = np.mean(window)
    std_val = np.std(window)
    min_val = np.min(window)
    max_val = np.max(window)
    return np.array([mean_val, std_val, min_val, max_val]).reshape(1, -1)

def classify_window(window, model=None):
    """
    Extract features from a sensor window and classify it using the provided model.
    If model is None, it loads the model from disk.
    Returns the predicted label (0, 1, or 2).
    """
    if model is None:
        model = load_model()
    features = extract_features(window)
    prediction = model.predict(features)[0]
    return prediction

# When running this file directly, train and save the model.
if __name__ == "__main__":
    train_and_save_model()
