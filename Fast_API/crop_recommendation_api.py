from fastapi import FastAPI
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from joblib import load 
from dotenv import load_dotenv
import os 

app = FastAPI()

crop_model = load("models/crop_recommendation.joblib")
scaler = load("encoders_scalers/scaler.joblib")
crop_label_encoder = load("encoders_scalers/crop_encoder.joblib")

load_dotenv()  
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

if WEATHERAPI_KEY is None:
    raise ValueError("WEATHERAPI_KEY is not set in the .env file")

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

SOIL_TYPES = ["Acidic Soil", "Alkaline Soil", "Loamy Soil", "Neutral Soil", "Peaty Soil"]

def get_historical_weather(latitude, longitude, days=30):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    url = f"http://api.weatherapi.com/v1/history.json?key={WEATHERAPI_KEY}&q={latitude},{longitude}&dt={start_date.strftime('%Y-%m-%d')}"
    response = requests.get(url)
    data = response.json()

    rainfall_values = []
    temperature_values = []
    humidity_values = []
    
    try:
        for day in data["forecast"]["forecastday"]:
            for hour in day["hour"]:
                temperature_values.append(hour["temp_c"])  
                humidity_values.append(hour["humidity"])  
                rainfall_values.append(hour["precip_mm"])  
        
        avg_temp = sum(temperature_values) / len(temperature_values) if temperature_values else None
        avg_humidity = sum(humidity_values) / len(humidity_values) if humidity_values else None
        total_rainfall = sum(rainfall_values) if rainfall_values else None
    except KeyError:
        avg_temp, avg_humidity, total_rainfall = None, None, None

    return avg_temp, avg_humidity, total_rainfall

def get_nasa_weather(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "parameters": "T2M, RH2M, PRECTOTCORR",
        "start": (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y%m%d"),
        "end": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "community": "AG",
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_URL, params=params).json()
    try:
        data = response["properties"]["parameter"]
        avg_temp = sum(data["T2M"].values()) / len(data["T2M"]) if "T2M" in data else None
        avg_humidity = sum(data["RH2M"].values()) / len(data["RH2M"]) if "RH2M" in data else None
        total_rainfall = sum(data["PRECTOTCORR"].values()) if "PRECTOTCORR" in data else None
    except KeyError:
        avg_temp, avg_humidity, total_rainfall = None, None, None
    return avg_temp, avg_humidity, total_rainfall

@app.get("/crop/predict_crop")
def predict_crop(latitude: float = None, longitude: float = None, soil_type: str = None,
                 nitrogen: float = None, phosphorus: float = None, 
                 potassium: float = None, carbon: float = None,
                 rainfall: float = None, temperature: float = None, 
                 humidity: float = None, PH: float = None):
    
    if rainfall is None or temperature is None or humidity is None:
        print("Fetching weather data from API...")
        temp_auto, humid_auto, rain_auto = get_historical_weather(latitude, longitude)
        if temp_auto is None or humid_auto is None or rain_auto is None:
            temp_auto, humid_auto, rain_auto = get_nasa_weather(latitude, longitude)
        temperature = temperature if temperature is not None else temp_auto
        humidity = humidity if humidity is not None else humid_auto
        rainfall = rainfall if rainfall is not None else rain_auto

    required_inputs = {
        "Temperature": temperature,
        "Humidity": humidity,
        "PH": PH,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorus,
        "Potassium": potassium,
        "Carbon": carbon
    }
    missing = {k: v for k, v in required_inputs.items() if v is None or v == 0}
    if missing:
        return {"message": "Invalid input: The following values must be provided and nonzero", "missing_or_zero": missing}

    print(f"Final Values → Temperature: {temperature}, Humidity: {humidity}, Rainfall: {rainfall}, PH: {PH}")

    numeric_feature_names = ["Temperature", "Humidity", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon"]
    numeric_values = [temperature, humidity, rainfall, PH, nitrogen, phosphorus, potassium, carbon]
    numeric_df = pd.DataFrame([numeric_values], columns=numeric_feature_names)
    
    soil_encoded = [1 if soil_type == s else 0 for s in ["Acidic Soil", "Alkaline Soil", "Loamy Soil", "Neutral Soil", "Peaty Soil"]]
    soil_df = pd.DataFrame([soil_encoded], columns=["Acidic Soil", "Alkaline Soil", "Loamy Soil", "Neutral Soil", "Peaty Soil"])

    input_df = pd.concat([numeric_df, soil_df], axis=1)

    expected_features = scaler.feature_names_in_
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    print("Input Data Before Scaling:\n", input_df)
    input_features_scaled = scaler.transform(input_df)
    print("Input Data After Scaling:\n", input_features_scaled)

    predicted_index = int(crop_model.predict(input_features_scaled)[0])
    predicted_crop_name = crop_label_encoder.inverse_transform([predicted_index])[0]
    print(f"Predicted Crop Index: {predicted_index} → Crop Name: {predicted_crop_name}")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "numeric_inputs": dict(zip(numeric_feature_names, numeric_values)),
        "soil_type": soil_type,
        "recommended_crop": predicted_crop_name
    }
