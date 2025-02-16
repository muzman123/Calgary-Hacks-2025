#!/usr/bin/env python3
import time
from Phidget22.Phidget import *
from Phidget22.PhidgetException import PhidgetException
from Phidget22.Devices.SoundSensor import SoundSensor
from Phidget22.Devices.DistanceSensor import DistanceSensor

# Callback function for the SPL change handler of the sound sensor.
def on_spl_change(sound_sensor, spl, three, four, five):
    print(f"{spl}, {three}, {four}, {five}")

# Callback function for the distance sensor.
#def on_distance_change(distance_sensor, distance):
#    print(f"[Distance Sensor] Distance: {distance} mm")

def main():
    try:
        # Initialize the sound sensor and set the SPL change handler.
        sound_sensor = SoundSensor()
        sound_sensor.setOnSPLChangeHandler(on_spl_change)
        print("Opening Sound Sensor...")
        sound_sensor.openWaitForAttachment(5000)  # Wait up to 5 seconds for attachment

        # Initialize the distance sensor and set its change handler.
        distance_sensor = DistanceSensor()
        #distance_sensor.setOnDistanceChangeHandler(on_distance_change)
        print("Opening Distance Sensor...")
        distance_sensor.openWaitForAttachment(5000)  # Wait up to 5 seconds for attachment

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
            distance_sensor.close()
        except Exception as e:
            print("Error closing sensors:", e)

if __name__ == "__main__":
    main()
