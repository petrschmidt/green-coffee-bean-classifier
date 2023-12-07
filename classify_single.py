from tensorflow.keras.models import load_model

# Load model
model = load_model('model/coffee_bean_classifier.h5')

import cv2
import numpy as np

def prepare_image(file_path):
    img_size = 128  # Same size as in model
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(-1, img_size, img_size, 1)
    return img

# Prepare a single image
test_image = prepare_image('Defect.jpg') # Example image

# Prediction
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions, axis=1)

if predicted_class[0] == 0:
    print("Model predicts: BAD")
else:
    print("Model predicts: GOOD")

