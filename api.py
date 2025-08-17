from fastapi import FastAPI
from HarvestPrediction import run_for_farmer

app = FastAPI()

@app.get("/predict")
def predict(pin: str = None, lat: float = None, lon: float = None, crop: str = "wheat", planting: str = "2025-05-14"):
    try:
        if pin:
            return run_for_farmer(pin=pin, planting_date_str=planting, crop_name=crop)
        elif lat and lon:
            return run_for_farmer(latlon=(lat, lon), planting_date_str=planting, crop_name=crop)
        return {"error": "Must provide either pin or lat/lon"}
    except Exception as e:
        return {"error": str(e)}
