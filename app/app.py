# ============================================================
# Harvestify - Smart Crop & Disease Recommendation System
# ============================================================

from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from sklearn.ensemble import RandomForestClassifier
import importlib
import sys
import os

# ------------------------------------------------------------
# Print environment details
# ------------------------------------------------------------
print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Flask:", importlib.import_module('flask').__version__)

# ------------------------------------------------------------
# Setup paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# Load Disease Model
# ------------------------------------------------------------
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_model_path = os.path.join(BASE_DIR, '..', 'models', 'plant_disease_model.pth')
disease_model = ResNet9(3, len(disease_classes))

try:
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
    disease_model.eval()
    print("✅ Disease model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load disease model — {e}")

# ------------------------------------------------------------
# Load and Train Crop Recommendation Model
# ------------------------------------------------------------
crop_recommendation_model = RandomForestClassifier(n_estimators=20, random_state=42)

try:
    crop_data_path = os.path.join(BASE_DIR, '..', 'Data-processed', 'crop_recommendation.csv')
    crop_data = pd.read_csv(crop_data_path, encoding='utf-8')
    X = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = crop_data['label']
    crop_recommendation_model.fit(X, y)
    print("✅ Crop recommendation model trained successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not train crop model — {e}")

# ------------------------------------------------------------
# WEATHER FETCH FUNCTION (Fixed)
# ------------------------------------------------------------
def weather_fetch(city_name):
    """
    Fetch temperature and humidity using OpenWeatherMap API.
    Returns None if city or key is invalid.
    """
    api_key = getattr(config, "weather_api_key", "").strip()
    if not api_key:
        print("⚠️ Weather API key missing in config.py")
        return None

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": api_key}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        print(f"\n🌤️ Weather API raw response for '{city_name}': {data}\n")

        if response.status_code != 200:
            print(f"❌ Weather API error {response.status_code}: {data.get('message', 'Unknown error')}")
            return None

        if "main" not in data:
            print(f"⚠️ No 'main' field in weather data for '{city_name}' — invalid city or key")
            return None

        temp_c = round(data["main"]["temp"] - 273.15, 2)
        humidity = data["main"]["humidity"]
        return temp_c, humidity

    except Exception as e:
        print(f"⚠️ Exception while fetching weather: {e}")
        return None

# ------------------------------------------------------------
# IMAGE PREDICTION
# ------------------------------------------------------------
def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert('RGB')
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        yb = model(img_u)
        _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# ------------------------------------------------------------
# FLASK APP ROUTES
# ------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', title='Harvestify - Home')

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='Harvestify - Crop Recommendation')

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='Harvestify - Fertilizer Suggestion')

@app.route('/disease')
def disease():
    return render_template('disease.html', title='Harvestify - Disease Detection')

# ------------------------------------------------------------
# CROP PREDICTION
# ------------------------------------------------------------
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'
    try:
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        weather = weather_fetch(city)
        if weather:
            temperature, humidity = weather
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)[0]
            return render_template('crop-result.html', prediction=prediction, title=title)
        else:
            return render_template('try_again.html', title=title)
    except Exception as e:
        print("⚠️ Error in crop prediction:", e)
        return render_template('try_again.html', title=title)

# ------------------------------------------------------------
# FERTILIZER RECOMMENDATION
# ------------------------------------------------------------
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'
    try:
        crop_name = str(request.form['cropname'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        df = pd.read_csv(os.path.join(BASE_DIR, 'Data', 'fertilizer.csv'))

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        key = f"{max_value}{'High' if eval(max_value.lower()) < 0 else 'low'}"

        response = Markup(str(fertilizer_dic[key]))
        return render_template('fertilizer-result.html', recommendation=response, title=title)
    except Exception as e:
        print("⚠️ Fertilizer error:", e)
        return render_template('try_again.html', title=title)

# ------------------------------------------------------------
# DISEASE PREDICTION
# ------------------------------------------------------------
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print("⚠️ Disease prediction error:", e)
            return render_template('try_again.html', title=title)
    return render_template('disease.html', title=title)

# ------------------------------------------------------------
# MAIN APP START
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
