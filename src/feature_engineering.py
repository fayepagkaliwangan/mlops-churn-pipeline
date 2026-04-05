"""
Feature engineering

This script prepares machine learning features by:
- encoding categorical variables
- normalizing numeric variables
- saving the final feature dataset
"""

# Import libraries
import pandas as pd
import os
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# File paths
clean_dataset_path = "data/processed/clean_data.csv"
features_output_path = "data/features/features.csv"


def load_clean_dataset():

    print("Loading cleaned dataset...")

    dataset_dataframe = pd.read_csv(clean_dataset_path)

    return dataset_dataframe


def encode_categorical_columns(dataset_dataframe):

    print("Encoding categorical columns...")

    categorical_columns = dataset_dataframe.select_dtypes(include=["object"]).columns

    # Remove churn columns if present
    if "Churn" in categorical_columns:
        categorical_columns = categorical_columns.drop("Churn")
    
    encoder = OrdinalEncoder()

    dataset_dataframe[categorical_columns] = encoder.fit_transform(
        dataset_dataframe[categorical_columns]
    )

    return dataset_dataframe, encoder


def normalize_numeric_columns(dataset_dataframe):

    print("Normalizing numeric columns...")

    scaler = StandardScaler()

    numeric_columns = dataset_dataframe.select_dtypes(include=["int64", "float64"]).columns
    
    # Remove churn columns if present
    if "Churn" in numeric_columns:
        numeric_columns = numeric_columns.drop("Churn")
    
    dataset_dataframe[numeric_columns] = scaler.fit_transform(
        dataset_dataframe[numeric_columns]
    )
    
    return dataset_dataframe, scaler


def save_features(dataset_dataframe):

    print("Saving engineered features...")

    os.makedirs("data/features", exist_ok=True)

    dataset_dataframe.to_csv(features_output_path, index=False)

    print("Feature dataset saved to:", features_output_path)


def run_feature_engineering_pipeline():

    dataset_dataframe = load_clean_dataset()

    dataset_dataframe, encoder = encode_categorical_columns(dataset_dataframe)

    dataset_dataframe, scaler = normalize_numeric_columns(dataset_dataframe)

    save_features(dataset_dataframe)

    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Encoder and Scaler are saved to models/ folder")

if __name__ == "__main__":
    run_feature_engineering_pipeline()