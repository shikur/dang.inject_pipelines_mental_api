
import os

# Set cache directories for XDG and Hugging Face Hub
os.environ['XDG_CACHE_HOME'] = '/home/msds2023/jlegara/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/msds2023/jlegara/.cache'

import torch

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# from tqdm.notebook import tqdm

from moviepy.editor import VideoFileClip, ImageSequenceClip

import torch
from facenet_pytorch import (MTCNN)

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)

from PIL import Image, ImageDraw
# Initialize MTCNN model for single face cropping
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=200,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    keep_all=False,
    device=device
)

# Load the pre-trained model and feature extractor
extractor = AutoFeatureExtractor.from_pretrained(
    "trpakov/vit-face-expression"
)
model = AutoModelForImageClassification.from_pretrained(
    "trpakov/vit-face-expression"
)


def detect_emotions(image):
    """
    Detect emotions from a given image, displays the detected
    face and the emotion probabilities in a bar plot.

    Parameters:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The cropped face from the input image.
    """

    # Create a copy of the image to draw on
    temporary = image.copy()
    print(temporary)
    # Use the MTCNN model to detect faces in the image
    sample = mtcnn.detect(temporary)

    # If a face is detected
    if sample[0] is not None:

        # Get the bounding box coordinates of the face
        box = sample[0][0]

        # Crop the detected face from the image
        face = temporary.crop(box)

        # Pre-process the cropped face to be fed into the
        # emotion detection model
        inputs = extractor(images=face, return_tensors="pt")

        # Pass the pre-processed face through the model to
        # get emotion predictions
        outputs = model(**inputs)

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits,
                                                    dim=-1)

        # Retrieve the id2label attribute from the configuration
        id2label = AutoConfig.from_pretrained(
            "trpakov/vit-face-expression"
        ).id2label

        # Convert probabilities tensor to a Python list
        probabilities = probabilities.detach().numpy().tolist()[0]

        # Map class labels to their probabilities
        class_probabilities = {id2label[i]: prob for i,
                               prob in enumerate(probabilities)}

        # Define colors for each emotion
        colors = {
            "angry": "red",
            "disgust": "green",
            "fear": "gray",
            "happy": "yellow",
            "neutral": "purple",
            "sad": "blue",
            "surprise": "orange"
        }
        palette = [colors[label] for label in class_probabilities.keys()]

       

        return face, class_probabilities
    else:
        return None