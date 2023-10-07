from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load model
model_path = 'bestModel.h5'  
model = tf.keras.models.load_model(model_path)

# Preprocess input image
def preprocess_image(image):
    try:
        image = Image.open(image)
        image = image.resize((168, 168))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError("Error processing image: " + str(e))

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            raise ValueError('No image found')
        
        image = request.files['image']
        image_data = preprocess_image(image)
        prediction = model.predict(image_data)[0]
        
        category = 'Segar' if np.argmax(prediction) == 1 else 'Mengantuk'
        confidence = float(prediction[np.argmax(prediction)])

        response = {
            'prediction': category,
            'confidence': confidence
        }

        return jsonify(response), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'error': str(e)}), 400, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
