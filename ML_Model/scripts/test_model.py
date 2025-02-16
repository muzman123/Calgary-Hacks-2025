import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# --- Load the trained model ---
model_path = 'sound_classification_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# --- Load the class indices mapping ---
json_path = "class_indices.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON mapping file not found: {json_path}")

with open(json_path, "r") as f:
    class_indices = json.load(f)

# Invert the mapping: keys become indices, values become class names
indices_to_class = {v: k for k, v in class_indices.items()}
print("Class indices mapping:", class_indices)

# --- Function to prepare an image for prediction ---
def prepare_image(image_path, target_size=(128, 128)):
    """Loads and preprocesses an image for model prediction."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- Specify the test image path ---
test_image_path = './spectrograms/crackling_fire/2-18766-A-12.png'

# If the test image doesn't exist, list available files in that folder
if not os.path.exists(test_image_path):
    print(f"Test image not found: {test_image_path}")
    test_folder = os.path.dirname(test_image_path)
    if os.path.exists(test_folder):
        available_files = os.listdir(test_folder)
        print(f"Available files in '{test_folder}':", available_files)
    else:
        print(f"Folder '{test_folder}' not found.")
    exit(1)

# --- Preprocess the test image and run prediction ---
try:
    image = prepare_image(test_image_path)
except Exception as e:
    print(f"Error preparing image: {e}")
    exit(1)

prediction = model.predict(image)
predicted_index = np.argmax(prediction, axis=1)[0]
predicted_label = indices_to_class.get(predicted_index, "Unknown")

# --- Print prediction details ---
print("Prediction probabilities:", prediction)
print("Predicted index:", predicted_index)
print("Prediction label:", predicted_label)

# --- Display the image with the predicted label ---
try:
    img_to_show = load_img(test_image_path)
    plt.imshow(img_to_show)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error displaying image: {e}")
