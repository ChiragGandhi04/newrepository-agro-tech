import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
#from tensorflow.keras.preprocessing.image import image, load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load models and encoders
crop_model = pickle.load(open('model.pkl', 'rb'))
crop_scaler = pickle.load(open('standscaler.pkl', 'rb'))
crop_minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

fertilizer_model = pickle.load(open('classifier.pkl', 'rb'))
fertilizer_encoder = pickle.load(open('fertilizer.pkl', 'rb'))
soil_encoder = {'Black': 0, 'Clayey': 1, 'Loamy': 2, 'Red': 3, 'Sandy': 4}
crop_encoder = {'Barley': 0, 'Cotton': 1, 'Groundnut': 2, 'Maize': 3, 'Millets': 4,
                'Oil seeds': 5, 'Paddy': 6, 'Pulses': 7, 'Sugarcane': 8, 'Tobacco': 9, 'Wheat': 10}

disease_model = load_model('model.h5')
# Labels for plant disease detection
disease_labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Home page
@app.route('/')
def home():
    return render_template('global.html')

# Crop Recommendation Page
@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('Index_crop.html')

# Fertilizer Prediction Page
@app.route('/fertilizer_prediction')
def fertilizer_prediction():
    return render_template('Index_fertilizer.html')

# Predict Crop
@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        # Get form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare features
        features = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)
        scaled_features = crop_scaler.transform(crop_minmax_scaler.transform(features))

        # Predict crop
        crop_prediction = crop_model.predict(scaled_features)[0]

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        crop = crop_dict.get(crop_prediction, "Unknown crop")
        result = f"The best crop to grow is: {crop}"

        return render_template('Index_crop.html', result=result)
    except Exception as e:
        return render_template('Index_crop.html', result=f"Error: {e}")

# Predict Fertilizer
@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        # Get form data
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])

        # Encode soil and crop type
        soil_encoded = soil_encoder.get(soil_type.capitalize())
        crop_encoded = crop_encoder.get(crop_type.capitalize())

        if soil_encoded is None or crop_encoded is None:
            return render_template('Index_fertilizer.html', result="Invalid soil type or crop type.")

        # Prepare input data
        input_data = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, N, P, K]])
        prediction = fertilizer_model.predict(input_data)

        # Decode fertilizer name
        fertilizer_name = fertilizer_encoder.classes_[prediction[0]]
        result = f"The recommended fertilizer is: {fertilizer_name}"

        return f"The recommended fertilizer is: {fertilizer_name}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Plant Disease Detection routes
@app.route('/plant_disease', methods = ['GET'])
def plant_disease():
    return render_template('example.html')

@app.route('/predict_disease', methods=['GET','POST'])
def predict_disease():
    try:
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        #file_path = os.makedirs(os.path.join(basepath, 'uploads'), exist_ok=True)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Predict plant disease
        img = load_img(file_path, target_size=(225, 225))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = disease_model.predict(img_array)
        label = disease_labels[np.argmax(prediction)]

        result1 = f"Plant disease detected: {label}"
        return render_template('example.html', result=result1)
    except Exception as e:
        return f"An error occurred: {str(e)}"



if __name__ == '__main__':
    app.run(debug=True)