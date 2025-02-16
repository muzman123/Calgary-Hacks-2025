# sensor_handler.py
import time
import numpy as np
from Phidget22.Phidget import *
from Phidget22.PhidgetException import PhidgetException
from Phidget22.Devices.SoundSensor import SoundSensor

# Import classification function from model.py.
from misc.model import classify_window, load_model

# Global parameters for buffering sensor data.
WINDOW_SIZE = 10
sensor_buffer = []

# Load the pre-trained model once.
model = load_model()

# Define a mapping from numeric label to human-readable string.
LABEL_MAP = {
    0: "Calm Forest Ambience",
    1: "Possible Human Activity",
    2: "Chainsaw/Logging Activity"
}

def detect_spikes(window, spike_threshold=5):
    """
    Detect spikes in the given sensor window.
    
    Parameters:
      window (list or numpy array): A sequence of decibel readings.
      spike_threshold (float): The minimum difference (in dB) above the baseline 
                               that qualifies as a spike.
    
    Returns:
      (bool, list): A tuple where the first element indicates whether any spike was 
                    detected and the second is a list of the spike values.
    """
    # Calculate a baseline using the mean (or use median if preferred)
    baseline = np.mean(window)
    # Find all readings that exceed the baseline by at least the threshold.
    spikes = [x for x in window if x - baseline >= spike_threshold]
    return (len(spikes) > 0), spikes

def on_spl_change(sound_sensor, spl, param3, param4, param5):
    """
    Callback for the sound sensor's SPL change.
    Accumulates readings in a buffer. When the buffer reaches WINDOW_SIZE,
    it extracts features and classifies the sensor window. It also checks for spikes,
    which could indicate a footstep.
    """
    global sensor_buffer
    sensor_buffer.append(spl)
    # Optionally, print each new reading:
    # print("New SPL reading:", spl)
    
    if len(sensor_buffer) >= WINDOW_SIZE:
        # Take one complete window.
        window = sensor_buffer[:WINDOW_SIZE]
        # Remove the processed window from the buffer.
        sensor_buffer = sensor_buffer[WINDOW_SIZE:]
        
        # Check for spikes (which might indicate footsteps)
        spike_detected, spike_values = detect_spikes(window, spike_threshold=5)
        
        # Extract features and classify the window.
        prediction = classify_window(np.array(window), model)
        result_label = LABEL_MAP.get(prediction, "Unknown")
        
        print("\n--- New Sensor Window ---")
        print("Window readings:", window)
        print("Features: mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
            np.mean(window), np.std(window), np.min(window), np.max(window)))
        print("Predicted sound source:", result_label)
        
        if spike_detected:
            # You might choose to treat a detected spike as a footstep, or flag it for further processing.
            print("Spike(s) detected (possible footstep):", spike_values)
        else:
            print("No significant spikes detected.")

def start_sensor():
    """
    Initialize and run the Phidget sound sensor.
    """
    try:
        sound_sensor = SoundSensor()
        sound_sensor.setOnSPLChangeHandler(on_spl_change)
        print("Opening Sound Sensor...")
        sound_sensor.openWaitForAttachment(5000)  # Wait up to 5 seconds for sensor attachment
        
        print("Sensors are attached and running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)  # Keep the script running

    except PhidgetException as e:
        print(f"Phidget Exception {e.code}: {e.details}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            sound_sensor.close()
        except Exception as e:
            print("Error closing sensor:", e)

if __name__ == "__main__":
    start_sensor()
