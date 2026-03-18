"""
Data cleaning script

This script prepares the dataset for machine learning by:
- converting numeric columns
- handling problematic values
- saving a cleaned dataset
"""

# Import libraries
import pandas as pd
import os


# File paths
raw_dataset_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
clean_dataset_output_path = "data/processed/clean_data.csv"


def load_raw_dataset():
    """
    Load the raw dataset
    """

    print("Loading raw dataset...")

    dataset_dataframe = pd.read_csv(raw_dataset_path)

    return dataset_dataframe


def clean_dataset(dataset_dataframe):
    """
    Perform data cleaning operations
    """

    print("Cleaning dataset...")

    # Convert TotalCharges to numeric
    dataset_dataframe["TotalCharges"] = pd.to_numeric(
        dataset_dataframe["TotalCharges"], errors="coerce"
    )

    # Remove rows with missing values
    dataset_dataframe = dataset_dataframe.dropna()

    # Drop customerID because it is not useful for ML
    dataset_dataframe = dataset_dataframe.drop(columns=["customerID"])

    return dataset_dataframe


def save_clean_dataset(dataset_dataframe):
    """
    Save cleaned dataset
    """

    print("Saving cleaned dataset...")

    os.makedirs("data/processed", exist_ok=True)

    dataset_dataframe.to_csv(clean_dataset_output_path, index=False)

    print("Clean dataset saved to:", clean_dataset_output_path)


def run_cleaning_pipeline():

    dataset_dataframe = load_raw_dataset()

    cleaned_dataset = clean_dataset(dataset_dataframe)

    save_clean_dataset(cleaned_dataset)


if __name__ == "__main__":
    run_cleaning_pipeline()