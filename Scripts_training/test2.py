# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:30:56 2024

@author: SRISHA
"""

import os
import numpy as np
from skimage import io, color, transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

def load_images_and_labels(folders, labels, image_shape=(128, 128)):
    images = []
    labels_list = []
    for folder, label in zip(folders, labels):
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                image = io.imread(filepath)
                if image.shape[-1] == 4:
                    image = color.rgba2rgb(image)
                image = transform.resize(image, image_shape, mode='reflect', anti_aliasing=True)
                images.append(image)
                labels_list.append(label)
            except Exception as e:
                print(f"Error processing image {filepath}: {e}")
    return np.array(images), np.array(labels_list)

# Define paths to your data folders
healthy_folder = 'AHealthy'
diseased_folder = 'disease'
not_arecanut_folder = 'noarecanut'

# Load data
folders = [healthy_folder, diseased_folder, not_arecanut_folder]
labels = [0, 1, 2]
X, y = load_images_and_labels(folders, labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Healthy Arecanut, Diseased Arecanut, Not Arecanut
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=25, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Save the model
model.save('cnn_model.h5')
def predict_image(image_path, model):
    image = io.imread(image_path)
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    image = transform.resize(image, (128, 128), mode='reflect', anti_aliasing=True)
    image = np.expand_dims(image, axis=0)  # Expand dims to match the input shape of the model
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    class_names = ['Healthy Arecanut', 'Diseased Arecanut', 'Not Arecanut']
    return class_names[class_idx]

# Example usage
new_image_path = 'mahali2.jpg'
model = tf.keras.models.load_model('cnn_model.h5')
prediction = predict_image(new_image_path, model)
print(f'The new image is: {prediction}')
