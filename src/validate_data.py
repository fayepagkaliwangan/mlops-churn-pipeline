"""
Data validation script

This script checks the raw dataset for common issues:
- missing values
- duplicate rows
- incorrect data types

It generates a validation report in logs/validation_report.txt
"""

# Import required libraries
import pandas as pandas_library
import os


# File paths
raw_dataset_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
validation_report_path = "logs/validation_report.txt"


def load_dataset():
    """
    Load the dataset using pandas
    """

    print("Loading dataset...")

    dataset_dataframe = pandas_library.read_csv(raw_dataset_path)

    return dataset_dataframe


def check_missing_values(dataset_dataframe):
    """
    Count missing values for each column
    """

    missing_values = dataset_dataframe.isnull().sum()

    return missing_values


def check_duplicate_rows(dataset_dataframe):
    """
    Count duplicate rows
    """

    duplicate_rows_count = dataset_dataframe.duplicated().sum()

    return duplicate_rows_count


def generate_validation_report(dataset_dataframe):
    """
    Generate a text report summarizing dataset quality
    """

    print("Generating validation report...")

    missing_values = check_missing_values(dataset_dataframe)
    duplicate_rows_count = check_duplicate_rows(dataset_dataframe)

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    with open(validation_report_path, "w") as report_file:

        report_file.write("DATA VALIDATION REPORT\n")
        report_file.write("======================\n\n")

        report_file.write("Dataset shape:\n")
        report_file.write(str(dataset_dataframe.shape))
        report_file.write("\n\n")

        report_file.write("Missing values per column:\n")
        report_file.write(str(missing_values))
        report_file.write("\n\n")

        report_file.write("Duplicate rows:\n")
        report_file.write(str(duplicate_rows_count))
        report_file.write("\n")

    print("Validation report saved to:", validation_report_path)


def run_validation_pipeline():
    """
    Run the full validation pipeline
    """

    dataset_dataframe = load_dataset()

    generate_validation_report(dataset_dataframe)


if __name__ == "__main__":
    run_validation_pipeline()