from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

router = APIRouter()
@router.get("/")
def welcome():
    return {"message": "Welcome to Iris Classification API",
            "docs": "/api/v1/docs",
            "predict": "/api/v1/predict",
            "health": "/api/v1/health"}

def get_model():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(root_dir, 'model.pkl')
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Model file not found at {model_path}")
    except pickle.UnpicklingError:
        raise HTTPException(status_code=500, detail="Error loading model file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict")
def predict(iris_data: IrisInput):
    try:
        model = get_model()
        features = np.array([
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width
        ]).reshape(1, -1)
        prediction = model.predict(features)[0]
        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        return {"prediction": int(prediction), "class": class_names[prediction]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
