import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import io
import csv
import json
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

router = APIRouter()

MODEL_PATH = "models/model.h5"
CLASS_NAMES_PATH = "data/class_names.json"
CURE_CSV_PATH = "data/cure_info.csv"

print("Loading disease model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes

print("Loading class names...")
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    print(f"Class names loaded from {CLASS_NAMES_PATH}")
except Exception as e:
    print(f"Error loading class names: {e}")
    class_names = {}

print("Loading cure mapping...")
def load_cure_mapping(csv_path: str):
    cure_mapping = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cure_mapping[row["disease"]] = row["cure"]
    except Exception as e:
        print(f"Error loading cure mapping: {e}")
    return cure_mapping

cure_mapping = load_cure_mapping(CURE_CSV_PATH)
print(f"Cure mapping loaded from {CURE_CSV_PATH}")

# Helper functions
def read_imagefile(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes))

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]  
    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)  
    return image

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handles image upload and predicts crop disease."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PNG or JPG image.")

    try:
        if model is None:
            raise RuntimeError("Model not loaded. Check server logs for errors.")

        contents = await file.read()
        image = read_imagefile(contents)
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])

        predicted_class = class_names.get(str(predicted_index), "Unknown") if isinstance(class_names, dict) else class_names[predicted_index]
        cure_info = cure_mapping.get(predicted_class, "Cure information not available for this disease.")

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "cure": cure_info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@router.get("/")
def read_root():
    return {"message": "Crop Disease Classifier API is up and running."}
