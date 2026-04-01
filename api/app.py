from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CustomerInput(BaseModel):
    tenure: int
    monthly_charges: float
    contract: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerInput):
    return {"prediction": "yes", "churn_probability": 0.78}