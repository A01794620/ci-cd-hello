from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn
from sklearn.datasets import load_wine

model = None

# Recuperar el modelo
#with open("../model_pkl/wine_model_.pkl", "rb") as f:
with open("wine_model_.pkl", "rb") as f:
    model = pickle.load(f)

data = load_wine()
target_names = data.target_names


class WineData(BaseModel):
    features: List[float]


app = FastAPI()


@app.post("/predict-nish")
def predict(wine_data: WineData):

    if len(wine_data.features) != model.n_features_in_:

        raise HTTPException(
            status_code=400,
            detail="The input data has not the right number of features.",
        )

    # predict :-0
    prediction = model.predict([wine_data.features])[0]
    prediction_name = target_names[prediction]
    return {"prediction": int(prediction), "prediction_name": prediction_name}


@app.get("/")
def read_root():
    return {"message": "OK Yeah!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
