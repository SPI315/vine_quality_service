import io
import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dvc.api

# Инициируем приложение
app = FastAPI()

# Приложение готово, но модель еще не подгружена
is_ready = False
is_alive = True


class VinoData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


model = dvc.api.read("model/model.pkl", mode="rb")
model = io.BytesIO(model)
model = joblib.load(model)

is_ready = True


@app.get("/liveness")
def liveness():
    if is_alive:
        return {"status": "OK"}
    raise HTTPException(status_code=503, detail="service is not alive")


@app.get("/readiness")
def readiness():
    if is_ready:
        return {"status": "OK"}
    raise HTTPException(status_code=503, detail="service is not ready")


@app.post("/predict")
def predict(vino_data: VinoData) -> float:
    vino_data = vino_data.dict()
    vino_data = pd.DataFrame(vino_data, index=[0])
    predict = model.predict(vino_data)[0]
    return predict


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
