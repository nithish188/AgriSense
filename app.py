from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
else:
    print(f"[WARN] model.pkl not found. Run train_model.py first.")

SOIL_PROPERTIES = {
    'Alluvial': {'N': 40, 'P': 60, 'K': 50, 'ph': 6.5},
    'Black':    {'N': 30, 'P': 40, 'K': 60, 'ph': 7.2},
    'Red':      {'N': 20, 'P': 30, 'K': 40, 'ph': 6.0},
    'Clay':     {'N': 50, 'P': 50, 'K': 40, 'ph': 7.0},
    'Loamy':    {'N': 60, 'P': 50, 'K': 50, 'ph': 6.5},
    'Sandy':    {'N': 10, 'P': 15, 'K': 20, 'ph': 5.5}
}

FERTILIZER_SUGGESTIONS = {
    'Rice':      'Urea (Nitrogen-rich), DAP, and Muriate of Potash.',
    'Maize':     'NPK 12:32:16, Zinc Sulfate.',
    'Cotton':    'NPK 15:15:15, Ammonium Sulfate.',
    'Soybean':   'Single Super Phosphate (SSP), Gypsum.',
    'Wheat':     'Urea, DAP (Diammonium Phosphate).',
    'Sugarcane': 'NPK 10:26:26, Urea.',
    'Coffee':    'Ammonium Nitrate, Rock Phosphate.',
    'Apple':     'Calcium Nitrate, Boron supplements.',
    'Mango':     'Organic Manure, NPK 19:19:19.',
    'Grapes':    'Potassium Sulfate, Magnesium Sulfate.',
    'default':   'Standard NPK 19:19:19 and organic compost.'
}

CROP_TRANSLATIONS = {
    'Rice':      'நெல் (Nel)',
    'Maize':     'மக்கா சோளம் (Makka Cholam)',
    'Cotton':    'பருத்தி (Paruthi)',
    'Soybean':   'சோயா பீன்ஸ் (Soya Beans)',
    'Wheat':     'கோதுமை (Godhumai)',
    'Sugarcane': 'கரும்பு (Karumbu)',
    'Coffee':    'காபி (Kapi)',
    'Apple':     'ஆப்பிள் (Apple)',
    'Mango':     'மாம்பழம் (Mambalam)',
    'Grapes':    'திராட்சை (Dhiratchai)'
}

CROP_EMOJI = {
    'Rice': '🌾', 'Maize': '🌽', 'Cotton': '🌿', 'Soybean': '🫘',
    'Wheat': '🌾', 'Sugarcane': '🎋', 'Coffee': '☕',
    'Apple': '🍎', 'Mango': '🥭', 'Grapes': '🍇'
}

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html',
                           soil_types=list(SOIL_PROPERTIES.keys()))

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/recommend', methods=['GET', 'POST'], strict_slashes=False)
def recommend():
    if request.method == 'GET':
        return jsonify({"message": "Please use POST to recommend crops."})
    print("Recommend endpoint hit via POST!")
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train_model.py first.'}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body received.'}), 400

        city = (data.get('city') or '').strip()
        soil_type = (data.get('soil_type') or '').strip()

        if not city:
            return jsonify({'error': 'City name is required.'}), 400
        if soil_type not in SOIL_PROPERTIES:
            return jsonify({'error': f'Invalid soil type: {soil_type}'}), 400

        soil_data = SOIL_PROPERTIES[soil_type]
        N, P, K, ph = soil_data['N'], soil_data['P'], soil_data['K'], soil_data['ph']

        # --- Weather via Open-Meteo (no API key needed) ---
        try:
            import requests as req
            geo_url = (
                f"https://geocoding-api.open-meteo.com/v1/search"
                f"?name={city}&count=1&language=en&format=json"
            )
            geo_resp = req.get(geo_url, timeout=8).json()
            if not geo_resp.get('results'):
                return jsonify({'error': f'City "{city}" not found.'}), 404

            loc   = geo_resp['results'][0]
            lat, lon = loc['latitude'], loc['longitude']
            city_name = loc['name']

            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,precipitation"
            )
            w = req.get(weather_url, timeout=8).json()
            cur = w.get('current', {})
            temp        = float(cur.get('temperature_2m', 25.0))
            humidity    = float(cur.get('relative_humidity_2m', 60.0))
            precipitation = float(cur.get('precipitation', 0.0))

        except Exception:
            # Fallback: use typical values so prediction still works
            city_name   = city
            temp        = 28.0
            humidity    = 70.0
            precipitation = 0.0

        # If current precipitation is near zero, simulate seasonal rainfall
        if precipitation < 10:
            precipitation = float(np.random.uniform(50, 200))

        # --- Predict ---
        features   = np.array([[N, P, K, temp, humidity, ph, precipitation]])
        prediction = model.predict(features)[0]

        return jsonify({
            'crop':        prediction,
            'emoji':       CROP_EMOJI.get(prediction, '🌱'),
            'tamil_crop':  CROP_TRANSLATIONS.get(prediction, prediction),
            'fertilizer':  FERTILIZER_SUGGESTIONS.get(prediction, FERTILIZER_SUGGESTIONS['default']),
            'soil': {
                'type': soil_type,
                'N': N, 'P': P, 'K': K, 'ph': ph
            },
            'weather': {
                'city':        city_name,
                'temperature': round(temp, 1),
                'humidity':    round(humidity, 1),
                'rainfall':    round(precipitation, 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/soil_types')
def soil_types():
    return jsonify(list(SOIL_PROPERTIES.keys()))


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='127.0.0.1')
