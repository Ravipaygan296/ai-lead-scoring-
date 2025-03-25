from fastapi import FastAPI
import pickle
import shap
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load AI Model
model = pickle.load(open("lead_model.pkl", "rb"))
shap_values = pickle.load(open("shap_values.pkl", "rb"))

class LeadRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(lead: LeadRequest):
    prob = model.predict_proba([lead.features])[0, 1]
    return {
        "lead_conversion_probability": prob,
        "suggestion": "Follow-up in 3 days" if prob > 0.7 else "Send nurture email"
    }

@app.get("/explain")
def explain():
    return {"top_features": shap_values.feature_names}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
