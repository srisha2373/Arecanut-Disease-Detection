# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:02:39 2024

@author: SRISHA
"""

import streamlit as st
import tensorflow as tf
from skimage import io, color, transform
import numpy as np

st.title("Arecanut Disease Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = io.imread(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    model = tf.keras.models.load_model('cnn_model.h5')

    def predict_image(image, model):
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image = transform.resize(image, (128, 128), mode='reflect', anti_aliasing=True)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_idx = np.argmax(prediction)
        class_names = ['Healthy Arecanut', 'Diseased Arecanut', 'Not Arecanut']
        return class_names[class_idx]

    prediction = predict_image(image, model)
    st.write(f'The new image is: {prediction}')
