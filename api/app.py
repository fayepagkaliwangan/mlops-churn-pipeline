import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#load the correct trained model
model = joblib.load("models/RandomForest_model.pkl")
print(f"Loaded model: {type(model)}")

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerInput):
    try:
        df = pd.DataFrame([data.dict()])
        df = pd.get_dummies(df)
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return {"prediction": "Churn" if prediction == 1 else "No churn","churn_probability": float(probability)}

    except Exception as e:
        return {"error": str(e)}