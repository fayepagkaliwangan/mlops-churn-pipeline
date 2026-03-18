"""
Feature engineering

This script prepares machine learning features by:
- encoding categorical variables
- normalizing numeric variables
- saving the final feature dataset
"""

# Import libraries
import pandas as pandas_library
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


# File paths
clean_dataset_path = "data/processed/clean_data.csv"
features_output_path = "data/features/features.csv"


def load_clean_dataset():

    print("Loading cleaned dataset...")

    dataset_dataframe = pandas_library.read_csv(clean_dataset_path)

    return dataset_dataframe


def encode_categorical_columns(dataset_dataframe):

    print("Encoding categorical columns...")

    label_encoder = LabelEncoder()

    categorical_columns = dataset_dataframe.select_dtypes(include=["object"]).columns

    for column_name in categorical_columns:
        dataset_dataframe[column_name] = label_encoder.fit_transform(
            dataset_dataframe[column_name]
        )

    return dataset_dataframe


def normalize_numeric_columns(dataset_dataframe):

    print("Normalizing numeric columns...")

    scaler = StandardScaler()

    numeric_columns = dataset_dataframe.select_dtypes(include=["int64", "float64"]).columns

    dataset_dataframe[numeric_columns] = scaler.fit_transform(
        dataset_dataframe[numeric_columns]
    )

    return dataset_dataframe


def save_features(dataset_dataframe):

    print("Saving engineered features...")

    os.makedirs("data/features", exist_ok=True)

    dataset_dataframe.to_csv(features_output_path, index=False)

    print("Feature dataset saved to:", features_output_path)


def run_feature_engineering_pipeline():

    dataset_dataframe = load_clean_dataset()

    dataset_dataframe = encode_categorical_columns(dataset_dataframe)

    dataset_dataframe = normalize_numeric_columns(dataset_dataframe)

    save_features(dataset_dataframe)


if __name__ == "__main__":
    run_feature_engineering_pipeline()