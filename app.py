import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None

try:
    with open('optimized_diabetes_model.sav', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.sav', 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Model and scaler loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")


@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes Prediction API Running"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                "error": "Model or scaler not loaded"
            }), 500

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        if 'features' not in data:
            return jsonify({
                "error": "Missing 'features' key"
            }), 400

        features = np.array(
            data['features'],
            dtype=float
        ).reshape(1, -1)

        if features.shape[1] != 8:
            return jsonify({
                "error": f"Expected 8 features but received {features.shape[1]}"
            }), 400

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features_scaled)
            confidence = round(float(np.max(probability)), 4)

        return jsonify({
            "prediction": int(prediction),
            "label": "Diabetes" if prediction == 1 else "No Diabetes",
            "confidence": confidence
        })

    except ValueError:
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
