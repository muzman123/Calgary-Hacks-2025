# sensor_handler.py
import time
import numpy as np
from Phidget22.Phidget import *
from Phidget22.PhidgetException import PhidgetException
from Phidget22.Devices.SoundSensor import SoundSensor

# Import classification function from model.py.
from model import classify_window, load_model

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

def on_spl_change(sound_sensor, spl, param3, param4, param5):
    """
    Callback for the sound sensor's SPL change.
    Accumulates readings in a buffer. When the buffer reaches WINDOW_SIZE,
    it extracts features and classifies the sensor window.
    """
    global sensor_buffer
    sensor_buffer.append(spl)
    # For debugging, you can print each new reading:
    # print("New SPL reading:", spl)
    
    if len(sensor_buffer) >= WINDOW_SIZE:
        # Take one complete window.
        window = sensor_buffer[:WINDOW_SIZE]
        # Remove the processed window from the buffer.
        sensor_buffer = sensor_buffer[WINDOW_SIZE:]
        
        # Extract features and classify the window.
        prediction = classify_window(np.array(window), model)
        result_label = LABEL_MAP.get(prediction, "Unknown")
        print("\n--- New Sensor Window ---")
        print("Window readings:", window)
        print("Features: mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
            np.mean(window), np.std(window), np.min(window), np.max(window)))
        print("Predicted sound source:", result_label)

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
