import pickle

import pandas as pd
from fastapi import FastAPI  # type: ignore

from src.data_loader import load_data
from src.preprocessing import wrangle

app = FastAPI()

# Load your model
with open("models/predictor.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
async def main_page():
    test_df = load_data("./data/raw/test.csv")
    test_df = wrangle(test_df)
    prediction = model.predict(test_df)
    return {"prediction": prediction.tolist()}