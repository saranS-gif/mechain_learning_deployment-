import pickle
import joblib
import numpy as np
from flask import Flask, request, jsonify
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load model and scaler with error handling"""
    global model, scaler
    
    model_paths = [
        'optimized_diabetes_model.sav',
        'best_diabetes_model_lr.sav'
    ]
    
    scaler_path = 'scaler.sav'
    
    # Try to load model from different possible paths
    for model_file in model_paths:
        if os.path.exists(model_file):
            try:
                # Try joblib first (recommended for sklearn models)
                model = joblib.load(model_file)
                logger.info(f"✅ Model loaded successfully from {model_file} using joblib!")
                break
            except Exception as e:
                try:
                    # Fallback to pickle
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"✅ Model loaded successfully from {model_file} using pickle!")
                    break
                except Exception as pickle_error:
                    logger.warning(f"Failed to load {model_file}: {pickle_error}")
                    continue
    
    if model is None:
        logger.error(f"❌ Model not found. Tried: {', '.join(model_paths)}")
    
    # Load scaler
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler loaded successfully using joblib!")
        except Exception as e:
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("✅ Scaler loaded successfully using pickle!")
            except Exception as pickle_error:
                logger.error(f"❌ Error loading scaler with pickle: {pickle_error}")
    else:
        logger.error(f"❌ Scaler file not found: {scaler_path}")

# Load model and scaler on startup
load_model_and_scaler()

@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes Prediction API Running",
        "model_status": "loaded" if model is not None else "not loaded",
        "scaler_status": "loaded" if scaler is not None else "not loaded"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                "error": "Model or scaler not loaded",
                "model_status": "loaded" if model is not None else "not loaded",
                "scaler_status": "loaded" if scaler is not None else "not loaded"
            }), 503

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

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False") == "True"
    app.run(host='0.0.0.0', port=port, debug=debug)
