
from flask import Flask, render_template, request
import numpy as np
import pickle

# Load Crop Recommendation Model
crop_model = pickle.load(open('model.pkl', 'rb'))
crop_sc = pickle.load(open('standscaler.pkl', 'rb'))
crop_mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Load Fertilizer Prediction Model
fertilizer_model = pickle.load(open('classifier.pkl', 'rb'))
fertilizer_encoder = pickle.load(open('fertilizer.pkl', 'rb'))
soil_encoder = {'Black': 0, 'Clayey': 1, 'Loamy': 2, 'Red': 3, 'Sandy': 4}
crop_encoder = {'Barley': 0, 'Cotton': 1, 'Groundnut': 2, 'Maize': 3, 'Millets': 4,
                'Oil seeds': 5, 'Paddy': 6, 'Pulses': 7, 'Sugarcane': 8, 'Tobacco': 9, 'Wheat': 10}

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('Main_page.html')

# Crop recommendation page
@app.route("/crop_recommendation", methods=['GET', 'POST'])
def crop_recommendation():
    result = ""
    if request.method == 'POST':
        # Get form data
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        ph = request.form['pH']
        rainfall = request.form['Rainfall']

        # Prepare input data for the model
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale the features and make prediction
        mx_features = crop_mx.transform(single_pred)
        sc_mx_features = crop_sc.transform(mx_features)
        prediction = crop_model.predict(sc_mx_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated.".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('Index_crop.html', result=result)

# Fertilizer prediction page
@app.route("/fertilizer_prediction", methods=['GET', 'POST'])
def fertilizer_prediction():
    result = ""
    if request.method == 'POST':
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

            # Convert soil and crop type to encoded values
            soil_encoded = soil_encoder.get(soil_type.capitalize())
            crop_encoded = crop_encoder.get(crop_type.capitalize())

            if soil_encoded is None or crop_encoded is None:
                return "Invalid soil type or crop type. Please check your inputs."

            # Prepare the input data
            input_data = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, N, P, K]])

            # Predict fertilizer
            prediction = fertilizer_model.predict(input_data)
            fertilizer_name = fertilizer_encoder.classes_[prediction[0]]

            result = f"The recommended fertilizer is: {fertilizer_name}"
        except Exception as e:
            result = f"An error occurred: {str(e)}"
    return render_template('Index_fertilizer.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)