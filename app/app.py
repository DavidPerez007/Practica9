import os
import queue
import threading
import pickle
from flask import Flask, jsonify, request
import joblib
import numpy as np
from threading import Semaphore, Lock
from model.adaboost_custom import SimpleAdaBoost 
from joblib import load
from utils.DataTransformation import *




app = Flask(__name__)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "modelo.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

model = load("./model/modelo.pkl")



if model is None:
    raise ValueError("El modelo cargado es None")

def get_model_params():
    if model is None:
        raise ValueError("The model is not valid")
    params = model.get_params()
    if not isinstance(params, dict):
        raise TypeError("get_params() didn't return a valid dictionary")
    return {
        "team": "Boosters",
        "model": type(model).__name__,
        "n_estimators": params.get("n_estimators", "unknown"),
        "base_estimator": params.get("base_estimator", "unknown"),
    }


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_columns = load("./model/feature_columns.pkl")
        scaler = load("./model/scaler.pkl")
        poly = load("./model/poly.pkl")
        data = request.get_json()
        features = data.get("features")

        if features is None:
            return jsonify({"error": "No se proporcionaron características"}), 400

        df = pd.DataFrame([features])
        df_processed = preprocess_for_prediction(df, scaler=scaler, poly=poly, feature_columns=feature_columns)
        prediction = model.predict(df_processed)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/info', methods=['GET'])
def info():
    model_params = get_model_params()
    return model_params


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)