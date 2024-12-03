import os
import numpy as np
from skimage.feature import hog
from skimage import color, io, transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
def load_images_and_extract_features(healthy_folder, diseased_folder, image_shape=(128, 128)):
    features = []
    labels = []

    # Function to process images
    def process_image(image_path, label):
        image = io.imread(image_path)
        image = color.rgb2gray(image)
        image = transform.resize(image, image_shape, mode='reflect', anti_aliasing=True)
        hog_features = hog(image, block_norm='L2-Hys')
        features.append(hog_features)
        labels.append(label)

    # Load healthy images
    for image_name in os.listdir(healthy_folder):
        process_image(os.path.join(healthy_folder, image_name), 0)

    # Load diseased images
    for image_name in os.listdir(diseased_folder):
        process_image(os.path.join(diseased_folder, image_name), 1)

    return np.array(features), np.array(labels)

# Paths to your folders
healthy_folder = 'healthy'
diseased_folder = 'disease'
features, labels = load_images_and_extract_features(healthy_folder, diseased_folder)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
def predict_image(image_path, model, image_shape=(128, 128)):
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image = transform.resize(image, image_shape, mode='reflect', anti_aliasing=True)
    hog_features = hog(image, block_norm='L2-Hys')
    hog_features = hog_features.reshape(1, -1)  # Reshape to fit the model input
    prediction = model.predict(hog_features)
    return 'Healthy' if prediction == 0 else 'Diseased'
new_image_path = 'mahali2.jpg'
prediction = predict_image(new_image_path, classifier)
print(f'The new image is: {prediction}')
import joblib

# Train and save the model
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'svm_model.pkl')