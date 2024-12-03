# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:48:56 2024

@author: SRISHA
"""

import streamlit as st
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage import color, transform
import joblib

# Load the trained model
model = joblib.load('svm_model.pkl')

# Function to preprocess the image and extract HOG features
def preprocess_image(image, image_shape=(128, 128)):
    image = image.convert("L")  # Convert to grayscale
    image = np.array(image)
    image = transform.resize(image, image_shape, mode='reflect', anti_aliasing=True)
    hog_features = hog(image, block_norm='L2-Hys')
    hog_features = hog_features.reshape(1, -1)  # Reshape to fit the model input
    return hog_features

# Function to predict the class of a new image
def predict_image(image, model, image_shape=(128, 128)):
    processed_image = preprocess_image(image, image_shape)
    prediction = model.predict(processed_image)
    if prediction == 0:
        return 'Healthy Arecanut'
    elif prediction == 1:
        return 'Diseased Arecanut'
    else:
        return 'Not Arecanut'

# Streamlit app layout
st.title("Arecanut Disease Detection")
st.write("Use your mobile camera to take a photo of the arecanut and upload it to get a prediction.")

# File uploader section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    # Predict the class of the uploaded image
    result = predict_image(image, model)
    st.write(f'The uploaded image is: {result}')
