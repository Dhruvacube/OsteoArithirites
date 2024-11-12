import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import secrets
import tensorflow as tf
from tensorflow import keras # type: ignore

# Importing models
from models.KLGrade.alexnet import load_model as load_alexnet # type: ignore
from models.KLGrade.densenet201 import load_model as load_densenet201 # type: ignore
from models.KLGrade.googlenet import load_model as load_googlenet # type: ignore
from models.KLGrade.inceptionresnetv2 import load_model as load_inceptionresnetv2 # type: ignore
from models.WithoutKLGrade.googlenet import load_model as load_googlenet_without_klgrade # type: ignore

alexnet = load_alexnet()
densenet201 = load_densenet201()
googlenet = load_googlenet()
inceptionresnetv2 = load_inceptionresnetv2()
googlenet_without_klgrade = load_googlenet_without_klgrade()

img_height, img_width = 224, 224

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_NAMES = ['alexnet', 'densenet201', 'googlenet', 'inceptionresnetv2', 'googlenet_without_klgrade']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/<name>', methods=['POST'])
def upload_file(name: str):
    if name.lower() not in MODEL_NAMES:
        return jsonify({'error': 'Model not found'}), 400
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file and not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename+secrets.token_urlsafe(16)) # type: ignore
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    if name.lower() == 'alexnet':
        model = alexnet
    elif name.lower() == 'densenet201':
        model = densenet201
    elif name.lower() == 'googlenet':
        model = googlenet
    elif name.lower() == 'inceptionresnetv2':
        model = inceptionresnetv2
    elif name.lower() == 'googlenet_without_klgrade':
        model = googlenet_without_klgrade
    
    img = tf.keras.utils.load_img( # type: ignore
        os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img) # type: ignore
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return jsonify({'message': 'File uploaded successfully'}), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)