from flask import Flask,render_template,request
from PIL import Image
import numpy as numpy
import io
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('website.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'inputImage' not in request.files:
        return "No file part"

    file = request.files['inputImage']

    if file.filename == '':
        return "No selected file"

    # Save the file to a desired location (e.g., in the same directory as the script)
    file.save(file.filename)
    model = load_model('model.h5')
    img = Image.open(file.filename).convert('L')  # Convert to grayscale
    img = img.resize((160, 160))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 160, 160, 1))  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming a classification model, get the class with the highest probability
    predicted_class = np.argmax(predictions[0])
    if(predicted_class == 0):
        return "Model detected MRI image does not contain Tumor"
    if(predicted_class == 1):
        return "Model detected MRI image contains Tumor"
    #return f"Predicted class: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
