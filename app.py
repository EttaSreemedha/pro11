from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image  # Import keras_image properly
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
from PIL import Image
import numpy as np
from flask_cors import CORS
import json
import io
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained models
deep_learning_model = load_model('C:/Users/sreem/Downloads/connect1/connect1/connect/flask-backend/model/model(1) .h5')
caption_model = load_model('C:/Users/sreem/Downloads/connect1/connect1/connect/flask-backend/model/model1006qqqqqq.keras')

# Load the word-to-index and index-to-word mappings
with open("C:/Users/sreem/Downloads/connect1/connect1/connect/flask-backend/model/wordtoix.json", 'r') as f:
    wordtoix = json.load(f)

with open("C:/Users/sreem/Downloads/connect1/connect1/connect/flask-backend/model/ixtoword.json", 'r') as f:
    ixtoword = json.load(f)

base_model = InceptionV3(weights='imagenet')
model1 = Model(base_model.input, base_model.layers[-2].output)

def preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(image):
    predicted_prob = deep_learning_model.predict(image)[0][0]
    predicted_class = 'Normal' if predicted_prob > 0.5 else 'Effusion'
    return predicted_class

def preprocess_image_caption(image):
    image = image.resize((299, 299))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def encode(image):
    image = preprocess_image_caption(image)
    vec = model1.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

def generate_caption(image):
    caption = 'startseq'
    max_length = 31
    for i in range(max_length):
        sequence = [wordtoix[w] for w in caption.split() if w in wordtoix]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        sequence = np.array([sequence])
        yhat = caption_model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[str(yhat)]
        caption += ' ' + word
        if word == 'endseq':
            break
    caption = caption.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)
    return caption

@app.route('/predict/<model_type>', methods=['POST'])
def upload_image(model_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if model_type == 'deep-learning':
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        image = preprocess_image(file_path)
        predicted_class = predict_image(image)
        os.remove(file_path)
        return jsonify({'predicted_class': predicted_class})
    elif model_type == 'llm':
        image = Image.open(io.BytesIO(file.read()))
        image_features = encode(image).reshape(1, 2048)
        caption = generate_caption(image_features)
        return jsonify({'caption': caption})
    else:
        return jsonify({'error': 'Invalid model type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
