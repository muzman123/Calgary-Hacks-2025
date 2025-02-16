#!/usr/bin/env python3
import time
import threading

# Global variable to store the current simulated SPL (dB) value.
current_spl = 40.0

def simulation_thread():
    """
    Simulate the sensor: every 0.5 seconds, increment the SPL value.
    When the value goes above 70 dB, reset it to 40 dB.
    """
    global current_spl
    while True:
        time.sleep(0.5)
        current_spl += 2.0
        if current_spl > 70.0:
            current_spl = 40.0
        # Uncomment below to see simulation in the console.
        # print(f"Simulated SPL: {current_spl:.1f} dB")

# Start the simulation thread (daemon thread so it stops when the program exits).
threading.Thread(target=simulation_thread, daemon=True).start()

def is_area_active(threshold_db=57.0):
    """
    Check if the simulated SPL reading is above the threshold.
    
    Args:
        threshold_db (float): The SPL threshold in dB.
        
    Returns:
        bool: True if the current simulated SPL is above the threshold, else False.
    """
    global current_spl
    return current_spl > threshold_db

def get_current_spl():
    """
    Return the current simulated SPL value.
    """
    global current_spl
    return current_spl
