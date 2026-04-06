import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#load the correct trained model
try:
    model_path = "models/model.pkl"
    model = joblib.load(model_path)
    encoder = joblib.load("models/encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("Successfully loaded model, encoder, and scaler.")
except Exception as e:
    print(f"Error loading models: {e}")

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
        df = pd.DataFrame([data.model_dump()])
        
        #using the saved ENCODER on categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        #using the saved SCALER on numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        #ensure column order perfectly matches what the model trained on
        df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return {"prediction": "Churn" if prediction == 1 else "No churn","churn_probability": float(probability)}

    except Exception as e:
        return {"error": str(e)}