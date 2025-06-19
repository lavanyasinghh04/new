# Start from a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (so Docker can cache dependencies)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-trained model file
COPY cats_vs_dogs_mobilenetv2.h5 .

# Copy the main app and templates
COPY app.py .
COPY templates/ templates/

# Expose port 5000 for the web app
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Start the app using Gunicorn (good for production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 