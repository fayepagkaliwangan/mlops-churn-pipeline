"""
Data ingestion script

This script downloads the Telco Customer Churn dataset
from Kaggle and stores it in the raw data folder.
"""

# Import required libraries
import subprocess
import os


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


if __name__ == "__main__":
    download_dataset_from_kaggle()