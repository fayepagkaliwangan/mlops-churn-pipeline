"""
Data ingestion script

This script downloads the Telco Customer Churn dataset
from Kaggle and stores it in the raw data folder.
"""

# Import required libraries
import subprocess
import os
import pandas as pd


def download_dataset_from_kaggle():
    """
    Download dataset from Kaggle using the Kaggle CLI
    """

    print("Starting data ingestion from Kaggle...")

    # Ensure raw data folder exists
    os.makedirs("data/raw", exist_ok=True)

    # Run Kaggle download command
    subprocess.run([
        "kaggle",
        "datasets",
        "download",
        "blastchar/telco-customer-churn",
        "-p",
        "data/raw",
        "--unzip"
    ])

    print("Dataset successfully downloaded to data/raw")

def load_raw_dataset():
    """
    Load the raw dataset into a dataframe
    """
    
    file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print("Loading dataset from:", file_path)
    df = pd.read_csv(file_path)
    print("Raw dataset loaded successfully. Shape:", df.shape)
    return df

if __name__ == "__main__":
    download_dataset_from_kaggle()
    df = load_raw_dataset()