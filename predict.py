import json
import time
import numpy as np
import os
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Helper functions
def load_trained_model(model_dir):
    full_model_path = os.path.join('.', model_dir)
    loaded_model = tf.keras.models.load_model(full_model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    return loaded_model

def preprocess_image(image):
    target_size = 224
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (target_size, target_size))
    image /= 255.0
    return image.numpy()

def make_prediction(img_path, model, top_k=5):
    image = Image.open(img_path)
    np_image = np.asarray(image)
    processed_image = preprocess_image(np_image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    top_probs = -np.partition(-predictions[0], top_k)[:top_k]
    top_classes = np.argpartition(-predictions[0], top_k)[:top_k]
    return top_probs, top_classes

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Path to the input image')
parser.add_argument('model_dir', type=str, help='Directory of the trained model')
parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
parser.add_argument('--class_names', type=str, required=True, help='Path to the JSON file mapping labels to class names')
args = parser.parse_args()

# Model loading and prediction
model = load_trained_model(args.model_dir)
probabilities, class_indices = make_prediction(img_path=args.image_path, model=model, top_k=args.top_k)

# Load and map class names
with open(args.class_names, 'r') as json_file:
    class_labels = json.load(json_file)
    mapped_classes = {int(idx): class_labels[str(idx)] for idx in class_indices}

# Display results
print('Predicted Classes:', mapped_classes)
print('Prediction Probabilities:', probabilities)
