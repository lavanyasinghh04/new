from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Load and resize image to 160x160 (MobileNetV2 input size)
    img = Image.open(image_path)
    img = img.resize((160, 160))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the model
print("Loading model...")
model = load_model('cats_vs_dogs_mobilenetv2.h5')

# Test with a sample image
test_image = 'data/test/cats/cat.4000.jpg'  # You can change this to any test image
print(f"\nTesting with image: {test_image}")

# Preprocess and predict
img_array = preprocess_image(test_image)
prediction = model.predict(img_array)[0][0]

# Get class and confidence
class_name = "dog" if prediction > 0.5 else "cat"
confidence = float(prediction if prediction > 0.5 else 1 - prediction)

print(f"\nPrediction: {class_name}")
print(f"Confidence: {confidence:.2%}") 