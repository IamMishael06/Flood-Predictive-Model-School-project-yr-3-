from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
import asyncio
import json
import joblib

app = FastAPI()

model = joblib.load("../data/random_forest_model.joblib")
data = pd.read_csv("../data/test_set1_cleaned.csv").drop(columns=["class"])

class_labels = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Incipient Leak in Production Choke",
    3: "Flow Instability",
    4: "Rapid Productivity Loss",
    5: "Quick Restriction in Production Choke",
    6: "Scaling in Production Choke",
    7: "Hydrate in Production Line",
    8: "Hydrate in Service Line",
    101: "Transient - Abrupt Increase of BSW",
    102: "Transient - Incipient Leak in Production Choke",
    103: "Transient - Flow Instability",
    104: "Transient - Rapid Productivity Loss",
    105: "Transient - Quick Restriction in Production Choke",
    106: "Transient - Scaling in Production Choke",
    107: "Transient - Hydrate in Production Line",
    108: "Transient - Hydrate in Service Line",
}


async def well_stream():
    for _, row in data.iterrows():
        row_df = row.to_frame().T  # single-row DataFrame for predict()

        prediction = model.predict(row_df)
        pred_value = int(prediction[0])
        prediction_label = class_labels.get(pred_value, f"Unknown ({pred_value})")

        payload = {
            "timestamp": str(row.get("timestamp", "")),
            "data": row.to_dict(),
            "prediction": prediction_label,
        }

        yield f"data: {json.dumps(payload, default=str)}\n\n"
        await asyncio.sleep(2)


@app.get("/stream")
async def stream():
    return StreamingResponse(well_stream(), media_type="text/event-stream")


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}