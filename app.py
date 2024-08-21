from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
from tensorflow.keras.preprocessing import image
import numpy as np
import io


app = Flask(__name__)

# Load the model
model = load_model('densenet_model.h5')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load and preprocess the image
        img = PILImage.open(io.BytesIO(file.read())).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class
        predictions = model.predict(img_array)
        class_label = 'HP' if predictions[0][0] >= 0.5 else 'SSA'

        return jsonify({'class': class_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
