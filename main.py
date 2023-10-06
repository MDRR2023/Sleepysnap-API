from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load model
model_path = 'bestModel.h5'  # Sesuaikan dengan path model di Render
model = tf.keras.models.load_model(model_path)

# Preprocess input image
def preprocess_image(image):
    image = image.resize((168, 168))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    image = Image.open(image)
    image = preprocess_image(image)

    try:
        prediction = model.predict(image)[0]
        if prediction > 0.5:
            result = 'Segar'
        else:
            result = 'Mengantuk'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
