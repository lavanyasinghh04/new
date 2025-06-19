from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Make sure the templates folder exists (for HTML files)
os.makedirs('templates', exist_ok=True)

# Load the pre-trained model (MobileNetV2)
model = load_model('cats_vs_dogs_mobilenetv2.h5')

def preprocess_image(image):
    # Resize the image to what the model expects
    image = image.resize((160, 160))
    # Convert the image to a numpy array and scale pixel values
    img_array = np.array(image) / 255.0
    # Add a batch dimension (model expects batches)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    # Show the upload form
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Read the uploaded image and preprocess it
        image = Image.open(io.BytesIO(request.files['image'].read()))
        processed_image = preprocess_image(image)
        
        # Get the model's prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Figure out if it's a cat or dog, and how confident the model is
        class_name = "dog" if prediction > 0.5 else "cat"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return jsonify({
            "class": class_name,
            "confidence": confidence
        })
    
    except Exception as e:
        # If something goes wrong, show the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app locally
    app.run(debug=True, host='127.0.0.1', port=5000) 