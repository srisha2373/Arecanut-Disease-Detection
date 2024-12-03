import os
import numpy as np
from skimage.feature import hog
from skimage import color, io, transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def load_images_and_extract_features(healthy_folder, diseased_folder, not_arecanut_folder, image_shape=(128, 128)):
    features = []
    labels = []

    # Function to process images
    def process_image(image_path, label):
        try:
            image = io.imread(image_path)
            image = color.rgb2gray(image)
            image = transform.resize(image, image_shape, mode='reflect', anti_aliasing=True)
            hog_features = hog(image, block_norm='L2-Hys')
            features.append(hog_features)
            labels.append(label)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Load healthy images
    for image_name in os.listdir(healthy_folder):
        image_path = os.path.join(healthy_folder, image_name)
        if os.path.isfile(image_path):
            process_image(image_path, 0)
        else:
            print(f"Skipping non-file {image_path}")

    # Load diseased images
    for image_name in os.listdir(diseased_folder):
        image_path = os.path.join(diseased_folder, image_name)
        if os.path.isfile(image_path):
            process_image(image_path, 1)
        else:
            print(f"Skipping non-file {image_path}")

    # Load not arecanut images
    for image_name in os.listdir(not_arecanut_folder):
        image_path = os.path.join(not_arecanut_folder, image_name)
        if os.path.isfile(image_path):
            process_image(image_path, 2)
        else:
            print(f"Skipping non-file {image_path}")

    return np.array(features), np.array(labels)

# Paths to your folders
healthy_folder = 'AHealthy'
diseased_folder = 'disease'
not_arecanut_folder = 'noarecanut'

# Load images and extract features
features, labels = load_images_and_extract_features(healthy_folder, diseased_folder, not_arecanut_folder)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained model
joblib.dump(classifier, 'svm_model.pkl')

# Function to predict the class of a new image
def predict_image(image_path, model, image_shape=(128, 128)):
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image = transform.resize(image, image_shape, mode='reflect', anti_aliasing=True)
    hog_features = hog(image, block_norm='L2-Hys')
    hog_features = hog_features.reshape(1, -1)  # Reshape to fit the model input
    prediction = model.predict(hog_features)
    if prediction == 0:
        return 'Healthy Arecanut'
    elif prediction == 1:
        return 'Diseased Arecanut'
    else:
        return 'Not Arecanut'

# Test the prediction on a new image
new_image_path = 'mahali2.jpg'
prediction = predict_image(new_image_path, classifier)
print(f'The new image is: {prediction}')
