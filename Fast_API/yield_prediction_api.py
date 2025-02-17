from fastapi import FastAPI, HTTPException
import requests
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta, timezone
from joblib import load
import ee
from dotenv import load_dotenv
import os
from google.oauth2 import service_account
load_dotenv()  

app = FastAPI()

service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if service_account_file is None:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set. Add it to your .env file.")

credentials = service_account.Credentials.from_service_account_file(
    service_account_file,
    scopes=["https://www.googleapis.com/auth/earthengine"]
)

project_id = os.getenv("project_id")

if project_id is None:
    raise ValueError("project_id is not set in the .env file")
 

try:
    ee.Initialize(credentials, project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(credentials, project=project_id)

yield_model = load("models/yield_prediction.joblib")
yield_scaler = load("encoders_scalers/yield_scaler.joblib")
yield_feature_names = load("encoders_scaler/yield_feature_names.joblib")

load_dotenv()  
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

if WEATHERAPI_KEY is None:
    raise ValueError("WEATHERAPI_KEY is not set in the .env file")

  
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

DEFAULT_BUFFER_M = 500
SEARCH_WINDOW_DAYS = 5  



def get_field_geometry(lat: float, lon: float, buffer_m: int = DEFAULT_BUFFER_M):
    point = ee.Geometry.Point(lon, lat)
    return point.buffer(buffer_m)

def get_sentinel2_collection(aoi, start: str, end: str):
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(ee.Date(start), ee.Date(end))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .sort('system:time_start', False))
    return collection

def compute_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    L = 0.5
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': L}
    ).rename('SAVI')
    return image.addBands([ndvi, gndvi, ndwi, savi])

def get_soil_moisture(aoi, date_str: str):
    try:
        date = ee.Date(date_str)
        sm_collection = ee.ImageCollection('NASA/SMAP/SPL4SMGP') \
            .filterDate(date.advance(-2, 'day'), date.advance(2, 'day'))
        image = sm_collection.first()
        ssm_dict = image.select('ssm').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=36000,
            maxPixels=1e9
        )
        ssm = ssm_dict.get('ssm')
        return ssm.getInfo() if ssm is not None else None
    except Exception as e:
        print(f"Error retrieving soil moisture: {e}")
        return None

def get_temperature(aoi, date_str: str):
    try:
        date = ee.Date(date_str)
        modis_collection = ee.ImageCollection("MODIS/061/MOD11A1")
        image = modis_collection.filterDate(date, date.advance(1, 'day')).first()
        lst_dict = image.select('LST_Day_1km').multiply(0.02).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        )
        lst = lst_dict.get('LST_Day_1km')
        return lst.getInfo() if lst is not None else None
    except Exception as e:
        print(f"Error retrieving temperature: {e}")
        return None

def get_rainfall(aoi, date_str: str):
    try:
        date = ee.Date(date_str)
        chirps_collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        image = chirps_collection.filterDate(date, date.advance(1, 'day')).first()
        precip_dict = image.select('precipitation').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=5000,
            maxPixels=1e9
        )
        precip = precip_dict.get('precipitation')
        return precip.getInfo() if precip is not None else None
    except Exception as e:
        print(f"Error retrieving rainfall: {e}")
        return None

def export_image(image, aoi, filename: str, scale: int = 10, format: str = 'PNG'):
    vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
    image_rgb = image.visualize(**vis_params)
    url = image_rgb.getThumbURL({'region': aoi, 'scale': scale, 'format': format.lower()})
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as img_file:
            img_file.write(response.content)
        return filename
    else:
        print(f"Error downloading image: {response.status_code}")
        return None

def process_field(latitude: float, longitude: float, crop: str, target_date: str,
                  soil_moisture_override: float = None,
                  temperature_override: float = None,
                  rainfall_override: float = None):
    
    aoi = get_field_geometry(latitude, longitude, buffer_m=DEFAULT_BUFFER_M)
    
    target = datetime.datetime.strptime(target_date, '%Y-%m-%d')
    window_start = (target - timedelta(days=SEARCH_WINDOW_DAYS)).strftime('%Y-%m-%d')
    window_end = (target + timedelta(days=SEARCH_WINDOW_DAYS)).strftime('%Y-%m-%d')
    
    s2_collection = get_sentinel2_collection(aoi, window_start, window_end)
    image_count = s2_collection.size().getInfo()
    if image_count == 0:
        raise Exception("No Sentinel-2 images found in the search window.")
    
    image = ee.Image(s2_collection.toList(image_count).get(0))
    image_with_indices = compute_indices(image)
    
    stats = image_with_indices.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e9
    )
    stats_dict = stats.getInfo()
    
    soil_moisture = get_soil_moisture(aoi, target_date)
    temperature = get_temperature(aoi, target_date)
    rainfall = get_rainfall(aoi, target_date)
    
    if soil_moisture_override is not None:
        soil_moisture = soil_moisture_override
    if temperature_override is not None:
        temperature = temperature_override
    if rainfall_override is not None:
        rainfall = rainfall_override
    
    features = {
        "NDVI": stats_dict.get('NDVI', None),
        "GNDVI": stats_dict.get('GNDVI', None),
        "NDWI": stats_dict.get('NDWI', None),
        "SAVI": stats_dict.get('SAVI', None),
        "Soil_Moisture": soil_moisture,
        "Temperature": temperature,
        "Rainfall": rainfall,
        "Crop_Type": crop
    }
    return features

@app.get("/predict_yield")
def predict_yield(
    latitude: float = None,
    longitude: float = None,
    crop: str = None,
    target_date: str = None,
    soil_moisture: float = None,
    temperature: float = None,
    rainfall: float = None):
    
    if target_date is None:
        target_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        features = process_field(
            latitude, longitude, crop, target_date,
            soil_moisture_override=soil_moisture,
            temperature_override=temperature,
            rainfall_override=rainfall
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing field: {e}")
    
    required_keys = ["NDVI", "GNDVI", "NDWI", "SAVI", "Soil_Moisture", "Temperature", "Rainfall"]
    missing = [k for k in required_keys if features.get(k) is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing data for features: {missing}")
    
    input_df = pd.DataFrame([features])
    input_df = input_df.reindex(columns=yield_feature_names, fill_value=0)
    
    scaled_input = yield_scaler.transform(input_df)
    predicted_yield = yield_model.predict(scaled_input)[0]
    
    return {
        "input_features": features,
        "predicted_yield": predicted_yield
    }