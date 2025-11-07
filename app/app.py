import os
import queue
import threading
import pickle
from flask import Flask, jsonify, request
import numpy as np
from threading import Semaphore, Lock
from model.SimpleAdaBoost import SimpleAdaBoost 



app = Flask(__name__)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "modelo.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

with open("./model/modelo.pkl", "rb") as f:
    model = pickle.load(f)

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
        # Obtiene los datos enviados en JSON
        data = request.get_json()
        features = data.get("features")

        if features is None:
            return jsonify({"error": "No se proporcionaron características"}), 400

        sex_map = {"male": 0, "female": 1}
        sex = sex_map.get(features.get("sex"), 0)

        embarked_q = 1 if features.get("embarked") == "Q" else 0
        embarked_s = 1 if features.get("embarked") == "S" else 0

        features_array = np.array([
            features.get("pclass"),
            sex,
            features.get("age"),
            features.get("sibsp"),
            features.get("parch"),
            features.get("fare"),
            embarked_q,
            embarked_s
        ]).reshape(1, -1)

        # ==== 2. Predicción ====
        prediction = model.predict(features_array)

        # ==== 3. Respuesta ====
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    model_params = get_model_params()
    return model_params


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)