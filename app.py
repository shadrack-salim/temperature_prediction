from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime

# Load the trained model
model = joblib.load("random_forest_temperature_model.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return "Temperature Prediction API (with datetime input) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # Parse datetime input
        dt_str = data["datetime"]
        dt = datetime.fromisoformat(dt_str)

        # Extract features
        hour = dt.hour
        day = dt.day
        month = dt.month
        weekday = dt.weekday()

        # Optional: Assume constant or default values for humidity and precipitation
        humidity = data.get("Humidity_%", 75)  # default: 75%
        precip = data.get("Precip_mm", 0.0)    # default: 0 mm

        # Predict using model
        features = [hour, day, month, weekday, humidity, precip]
        prediction = model.predict([features])[0]

        return jsonify({"predicted_temperature_c": round(prediction, 2)})

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
