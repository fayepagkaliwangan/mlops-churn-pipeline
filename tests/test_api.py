# To test API endpoints [Used AI to add @pytest feature, to make test robust]

import os
import sys
import pytest

# Add the project root to sys.path so 'api' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# Check if model files exist
MODEL_DIR = "models"
MODEL_FILES_PRESENT = (
    os.path.exists(os.path.join(MODEL_DIR, "model.pkl")) and
    os.path.exists(os.path.join(MODEL_DIR, "encoder.pkl")) and
    os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl"))
)

@pytest.fixture
def valid_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 800.25
    }

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        """Verify the API is online and responding to basic GET requests."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_predict_missing_field_returns_422(self):
        """Test that the /predict endpoint returns a 422 error when a required field is missing."""
        payload = {"gender": "Female"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(not MODEL_FILES_PRESENT, reason="Model artifacts not found in 'models/' directory")
    def test_predict_returns_200(self, valid_payload):
        """Confirm the /predict endpoint accepts a valid payload without crashing."""
        response = client.post("/predict", json=valid_payload)
        
        # If the model fails to load in app.py, this will return a 200 containing {"error": ...}
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "error" not in response.json(), f"Endpoint returned an internal error: {response.json().get('error')}"

    @pytest.mark.skipif(not MODEL_FILES_PRESENT, reason="Model artifacts not found in 'models/' directory")
    def test_predict_returns_valid_prediction(self, valid_payload):
        """Test that the /predict endpoint returns a valid prediction and probability for a realistic input."""
        response = client.post("/predict", json=valid_payload)
        result = response.json()
        
        assert "error" not in result, f"Endpoint returned an internal error: {result.get('error')}"
        assert "prediction" in result
        assert result["prediction"] in ["Churn", "No churn"]
        assert "churn_probability" in result
        assert 0.0 <= result["churn_probability"] <= 1.0