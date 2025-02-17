from fastapi import FastAPI
import requests
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

app = FastAPI()

fertilizer_model = load("models/fertilizer_recommendation.joblib")
scaler = load("encoders_scalers/fertilizer_scaler.joblib")
fertilizer_encoder = load("encoders_scalers/fertilizer_encoder.joblib")
crop_encoder = load("encoders_scalers/crop_encoder.joblib")
feature_names = load("encoders_scalers/fertilizer_feature_names.joblib")

df_remark = pd.read_csv('data/fertilizer_recommendation_dataset.csv')
fertilizer_to_remark = df_remark.groupby('Fertilizer')['Remark'].first().to_dict()

load_dotenv()  
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

if WEATHERAPI_KEY is None:
    raise ValueError("WEATHERAPI_KEY is not set in the .env file")


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

SOIL_TYPES = ["Acidic Soil", "Alkaline Soil", "Loamy Soil", "Neutral Soil", "Peaty Soil"]

def get_historical_weather(latitude: float, longitude: float, days: int = 30):
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    url = (f"http://api.weatherapi.com/v1/history.json?key={WEATHERAPI_KEY}"
           f"&q={latitude},{longitude}&dt={start_date.strftime('%Y-%m-%d')}")
    response = requests.get(url)
    data = response.json()
    
    rainfall_values = []
    temperature_values = []
    moisture_values = []
    try:
        for day in data["forecast"]["forecastday"]:
            for hour in day["hour"]:
                temperature_values.append(hour["temp_c"])
                moisture_values.append(hour["moisture"])  
                rainfall_values.append(hour["precip_mm"])
        avg_temp = np.mean(temperature_values) if temperature_values else None
        avg_moisture = np.mean(moisture_values) if moisture_values else None
        total_rainfall = np.sum(rainfall_values) if rainfall_values else None
    except KeyError:
        avg_temp, avg_moisture, total_rainfall = None, None, None

    return avg_temp, avg_moisture, total_rainfall

def get_nasa_weather(latitude: float, longitude: float):
    
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
        avg_temp = np.mean(list(data["T2M"].values())) if "T2M" in data else None
        avg_moisture = np.mean(list(data["RH2M"].values())) if "RH2M" in data else None
        total_rainfall = np.sum(list(data["PRECTOTCORR"].values())) if "PRECTOTCORR" in data else None
    except KeyError:
        avg_temp, avg_moisture, total_rainfall = None, None, None
    return avg_temp, avg_moisture, total_rainfall

@app.get("/recommend_fertilizer")
def recommend_fertilizer(
    latitude: float = None,
    longitude: float = None,
    soil_type: str = None,
    crop: str = None,
    nitrogen: float = None,
    phosphorus: float = None,
    potassium: float = None,
    carbon: float = None,
    PH: float = None,
    rainfall: float = None,
    temperature: float = None,
    moisture: float = None
):
   
    if rainfall is None or temperature is None or moisture is None:
        print("Fetching weather data from API...")
        temp_auto, moist_auto, rain_auto = get_historical_weather(latitude, longitude)
        if temp_auto is None or moist_auto is None or rain_auto is None:
            temp_auto, moist_auto, rain_auto = get_nasa_weather(latitude, longitude)
        temperature = temperature if temperature is not None else temp_auto
        moisture = moisture if moisture is not None else moist_auto
        rainfall = rainfall if rainfall is not None else rain_auto

    required_inputs = {
        "Temperature": temperature,
        "Moisture": moisture,
        "PH": PH,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorus,
        "Potassium": potassium,
        "Carbon": carbon
    }
    missing = {k: v for k, v in required_inputs.items() if v is None or v == 0}
    if missing:
        return {
            "message": "Invalid input: The following values must be provided and nonzero",
            "missing_or_zero": missing
        }
    
    print(f"Final Values → Temperature: {temperature}, Moisture: {moisture}, Rainfall: {rainfall}, PH: {PH}")
    
    numeric_feature_names = ["Temperature", "Moisture", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon"]
    numeric_values = [temperature, moisture, rainfall, PH, nitrogen, phosphorus, potassium, carbon]
    
    input_dict = {
        "Temperature": temperature,
        "Moisture": moisture,
        "Rainfall": rainfall,
        "PH": PH,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorus,
        "Potassium": potassium,
        "Carbon": carbon
    }
    
    for s in SOIL_TYPES:
        col_name = f"Soil_{s}"
        input_dict[col_name] = 1.0 if soil_type == s else 0.0
    
    crop_encoded = int(crop_encoder.transform([crop])[0])
    input_dict["Crop"] = crop_encoded

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    input_scaled = scaler.transform(input_df)
    
    predicted_index = int(fertilizer_model.predict(input_scaled)[0])
    recommended_fertilizer = fertilizer_encoder.inverse_transform([predicted_index])[0]
    
    remark = fertilizer_to_remark.get(recommended_fertilizer, "No remark available")
    
    return {
        "latitude": latitude,
        "longitude": longitude,
        "numeric_inputs": dict(zip(numeric_feature_names, numeric_values)),
        "soil_type": soil_type,
        "crop": crop,
        "recommended_fertilizer": recommended_fertilizer,
        "remark": remark
    }
