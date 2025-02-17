from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fertilizer_recommendation_api import app as fertilizer_api
from crop_recommendation_api import app as crop_api
from disease_detection_api import router as disease_router
from yield_prediction_api import app as yield_api

app = FastAPI(title="Samrudha Agricultural & Chatbot API Gateway")

origins = [
    " http://localhost:5173", 
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/fertilizer", fertilizer_api)   
app.mount("/crop", crop_api)                 
app.include_router(disease_router, prefix="/disease")         
app.mount("/yield", yield_api)               


@app.get("/")
def root():
    return {"message": "Welcome to the Samrudha API Gateway"}
