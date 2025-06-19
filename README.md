# Cats vs Dogs Image Classifier

## Overview
This project is a simple web app that can tell if a picture is of a cat or a dog. It uses a deep learning model (MobileNetV2) and a Flask web server. You can upload an image through the web interface and get a prediction right away. The whole thing is easy to run locally or in Docker.

## Project Structure
```
project_root/
├── data/
│   ├── train/         # Training images (cats and dogs)
│   └── test/          # Test images (cats and dogs)
├── app.py             # Flask web app for predictions
├── train_model.py     # Script to train your own model (optional)
├── cats_vs_dogs_mobilenetv2.h5  # Pre-trained model (used by default)
├── requirements.txt   # Python dependencies
├── Dockerfile         # For running everything in Docker
├── templates/
│   └── index.html     # Web interface
```

## Quick Start (Local)
1. **Install Python 3.9+** (if you don't have it already).
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   python app.py
   ```
5. **Open your browser:** Go to [http://127.0.0.1:5000](http://127.0.0.1:5000) and upload an image to test.

## Using Docker (Recommended for Deployment)
1. **Build the Docker image:**
   ```bash
   docker build -t cats-vs-dogs .
   ```
2. **Run the container:**
   ```bash
   docker run -p 5000:5000 cats-vs-dogs
   ```
3. **Go to [http://localhost:5000](http://localhost:5000) in your browser.**

## Training Your Own Model (Optional)
- If you want to train your own model, use `train_model.py`. Make sure your data is in the right folders (`data/train/cats`, `data/train/dogs`, etc.).
- The script will save a new model as `cats_vs_dogs_model.h5`.
- By default, the web app uses the MobileNetV2 model (`cats_vs_dogs_mobilenetv2.h5`).

## Troubleshooting
- **Model not found?** Make sure `cats_vs_dogs_mobilenetv2.h5` is in the project folder.
- **Dependency errors?** Double-check your Python version and run `pip install -r requirements.txt` again.
- **Docker build slow?** The first build can take a while because it downloads all dependencies.
- **App not loading?** Check the terminal for errors and make sure port 5000 is free.

## Credits
- Model architecture: MobileNetV2 (TensorFlow/Keras)
- Web app: Flask
- Student project, feel free to use or modify! 