import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import secrets
import tensorflow as tf
from tensorflow import keras # type: ignore
import numpy as np
from flasgger import Swagger
from typing import Literal
import cv2

# Importing models
import models.KLGrade.alexnet
import models.KLGrade.densenet201
import models.KLGrade.googlenet
import models.KLGrade.inceptionresnetv2 
import models.WithoutKLGrade.googlenet

alexnet = models.KLGrade.alexnet.load_model()
densenet201 = models.KLGrade.densenet201.load_model()
googlenet = models.KLGrade.googlenet.load_model()
inceptionresnetv2 = models.KLGrade.inceptionresnetv2.load_model()
googlenet_without_klgrade = models.WithoutKLGrade.googlenet.load_model()

img_height, img_width = 224, 224

app = Flask(__name__)
swagger = Swagger(app, template={
    "info": {
        "title": "API Endpoint Documentation",
        "version": "1.0.0"
    }
})

UPLOAD_FOLDER = 'uploads'
MODEL_NAMES = ['alexnet', 'densenet201', 'googlenet', 'inceptionresnetv2', 'googlenet_without_klgrade']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1000 * 1000


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/<name>', methods=['POST'])
def upload_file(name: str):
    """The model prediction and KL Grading for the osteoarthritis X-Rays
        ---
        parameters:
           - name: name
             in: path
             type: string
             enum: ['alexnet', 'densenet201', 'googlenet', 'inceptionresnetv2', 'googlenet_without_klgrade']
             required: true
             default: alexnet
           - name: image
             in: formData
             required: true
             type: file
        consumes:
          - multipart/form-data
        responses:
          200:
            description: The model prediction and KL Grade if applicable.
            schema:
                $ref: '#/definitions/Prediction'
            examples:
                class: 'patient'
                confidence: '73.105835'
                message: 'This image most likely belongs to patient with a 73.11 percent confidence.'
        definitions:
          Prediction:
            type: object
            properties:
              message:
                type: string
              class:
                type: string
              confidence:
                type: string
        """
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
        class_names = ["0", "1", "2", "3", "4"]
    elif name.lower() == 'densenet201':
        model = densenet201
        class_names = ["0", "1", "2", "3", "4"]
    elif name.lower() == 'googlenet':
        model = googlenet
        class_names = ["0", "1", "2", "3", "4"]
    elif name.lower() == 'inceptionresnetv2':
        model = inceptionresnetv2
        class_names = ["0", "1", "2", "3", "4"]
    elif name.lower() == 'googlenet_without_klgrade':
        model = googlenet_without_klgrade
        class_names = ["normal", "patient"]
    
    img = tf.keras.utils.load_img( # type: ignore
        os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img) # type: ignore
    img_array = cv2.resize(img_array, (224, 224))  # resize image to match model's expected sizing
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    
    return jsonify(
        {
            'message': "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)),
            'class': str(class_names[np.argmax(score)]),
            'confidence': str(100 * np.max(score))
        }
    ), 200
    
@app.route("/")
def main():
    return jsonify(
        {
            'apidocs': "gfg.dhruvashaw.in/apidocs",
            "dataset": "https://del1.vultrobjects.com/datasets/sorted.7z",
            "message": "Use postman to try out the api, or visit the apidocs for more information.",
            "Research Paper": "https://del1.vultrobjects.com/datasets/Detection_of_Osteoarthritis_and_doing_KL_Grade_for_Knee_Osteoarthritis_using_deep_learning_techniques.pdf",
            "Github Repo": "https://github.com/Dhruvacube/OsteoArithirites"
        }
    ), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host="0.0.0.0")