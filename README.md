# Flower Image Classifier

Flower-Image-Classifier is a deep learning project built using TensorFlow and Keras to classify images of flowers into different categories. The project leverages transfer learning by utilizing a pre-trained MobileNet model, fine-tuned to recognize and classify the flowers from the Oxford Flowers 102 dataset.

## Features

- **Data Loading**: Uses TensorFlow Datasets to load and preprocess the Oxford Flowers 102 dataset.
- **Data Preprocessing**: Includes resizing, normalization, and data augmentation techniques to prepare the images for model training.
- **Transfer Learning**: Utilizes the MobileNet model from TensorFlow Hub, with the classifier layers re-trained on the flower dataset.
- **Model Training**: Trains the model with appropriate hyperparameters, optimizing for both accuracy and generalization.
- **Model Evaluation**: Evaluates the trained model on a separate test set, providing metrics like accuracy and loss.
- **Command-Line Interface**: A `predict.py` script allows users to classify flower images using the trained model, supporting options to display top K predictions and flower names.

## Project Structure

- **`notebook.ipynb`**: Contains all the development work, including data loading, preprocessing, model training, and evaluation.
- **`predict.py`**: Command-line application for predicting the class of an image using the trained model.
- **`model.h5`**: The trained Keras model saved in HDF5 format.
- **`label_map.json`**: JSON file mapping class labels to flower names.

