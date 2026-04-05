import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Import centralized MLflow setup
from src.mlflow_tracking import setup_mlflow_experiment

# mock models for simulation
# If more mock models are to be added, add a class here and add it to the models dictionary.
class MockRandomModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        probs = np.random.rand(len(X))
        return np.column_stack((1 - probs, probs))


class MockRuleBasedModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        # Very simple rule: predict churn (1) if tenure is below average, else no churn (0)
        tenure = X.iloc[:, 4]                    
        return (tenure < tenure.mean()).astype(int)

    def predict_proba(self, X):
        # Return simple probability based on tenure
        tenure = X.iloc[:, 4]
        # Lower tenure = higher churn probability (between 0 and 1)
        prob = (tenure.max() - tenure) / (tenure.max() - tenure.min() + 1e-8)
        prob = np.clip(prob, 0.0, 1.0)
        return np.column_stack((1 - prob, prob))


# defining paths
DATA_PATH = "data/splits/"


# load data
def load_data():
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))

    # Convert Yes/No to 1/0
    y_train = y_train["Churn"].map({"Yes": 1, "No": 0}).values
    y_test = y_test["Churn"].map({"Yes": 1, "No": 0}).values

    return X_train, X_test, y_train, y_test



# train and evaluate
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    return model, accuracy, precision, f1



# main training pipeline
def train_and_select_model():
    setup_mlflow_experiment()
    X_train, X_test, y_train, y_test = load_data()
    # if new models are to be added, add it to the models dictionary
    models = {
        "MockRandom": MockRandomModel(),
        "MockRuleBased": MockRuleBasedModel(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_model = None
    best_model_name = None
    best_score = -1

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            trained_model, accuracy, precision, f1 = train_and_evaluate(model, X_train, X_test, y_train, y_test)

            # log parameters
            mlflow.log_param("model_name", model_name)
            if "Mock" in model_name:
                mlflow.log_param("model_type", "mock")
            else:
                mlflow.log_param("model_type", "real")

            # log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)

            # log model to MLflow
            mlflow.sklearn.log_model(trained_model, name="model")

            print(f"{model_name} ---- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

            # select best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = trained_model
                best_model_name = model_name

    return best_model, best_model_name



# save model  and encoder and scaler 
def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/model.pkl"
    joblib.dump(model, model_path)
    try:
        encoder = joblib.load("models/encoder.pkl")
        scaler = joblib.load("models/scaler.pkl")
        print("Encoder and Scaler loaded successfully.")
    except FileNotFoundError:
        print("Encoder or Scaler not found.")
    print(f"Best model ({model_name}) saved to {model_path}")



if __name__ == "__main__":
    best_model, best_model_name = train_and_select_model()
    save_model(best_model, best_model_name)