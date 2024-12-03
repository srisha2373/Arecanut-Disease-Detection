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

# Streamlit app layout
st.title("Arecanut Disease Detection")
st.write("Use your mobile camera to take a photo of the arecanut and get a prediction.")

# Camera input section
uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = 'Healthy' if prediction == 0 else 'Diseased'
    st.write(f"The arecanut is: {result}")
