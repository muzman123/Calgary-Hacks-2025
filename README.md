# Lorax - Acoustic Anti-Logging System
Lorax is an acoustic monitoring and machine learning system designed to detect potential illegal logging activities in real time. It uses a sound sensor for continuous SPL (Sound Pressure Level) monitoring, a microphone for high-fidelity recordings when suspicious noise levels are detected, and a trained convolutional neural network to classify the audio events. The project also provides a Flask web interface for live visualization of dB readings, spectrogram images, and predicted audio class labels.

## Features
**Real-Time SPL Monitoring**: Continuously tracks dB levels using a Phidget sound sensor (or simulated data).

**Threshold-Based Audio Capture**: Records a short audio clip from a microphone whenever the SPL exceeds a preset dB threshold.

**Machine Learning Classification**: Converts audio to spectrograms and uses a CNN (trained on forest sound datasets) to detect chainsaw or other suspicious activity.

**Web Interface**: A Flask-powered dashboard that provides:
-A live chart of SPL readings
-Predicted activity labels
-Visual spectrogram images for quick reference

## Usage
Clone the Repository and pip install all modules required 

Run the Flask Web App

Navigate to src/web/:

```bash
cd src/web
python app.py
```
Open your browser at http://localhost:5000 to view the dashboard.
Observe the Dashboard

Live SPL readings update in real time (either from the actual Phidget sensor or a simulated data source).
Once the SPL crosses the threshold (default ~57 dB), the system captures and classifies a 5-second audio clip.
The predicted label and corresponding spectrogram image appear in the interface.
Retrain or Update the Model

In src/scripts/, run ```train_model.py``` (adjusting dataset paths as needed).
Replace the old .h5 file in src/web/ with the newly trained version.

## Attributions

- ESC-50: Dataset for Environmental Sound Classification used for training audio captioning model: https://github.com/karolpiczak/ESC-50
