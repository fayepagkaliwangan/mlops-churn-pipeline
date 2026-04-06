# Data validation tests for the Telco Customer Churn dataset.

import os
import pytest
import pandas as pd


# Path to the raw dataset
RAW_DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


class TestRawDataExists:
    """Verify that the raw dataset is present and accessible"""

    def test_raw_csv_file_exists(self):
        """Test that the raw Telco churn CSV file exists on disk"""
        assert os.path.exists(RAW_DATA_PATH), (f"Raw dataset not found at: {RAW_DATA_PATH}. ")

    def test_raw_csv_is_not_empty(self):
        """Test that the raw CSV file is not an empty file"""
        assert os.path.getsize(RAW_DATA_PATH) > 0, ("Raw dataset file exists but is empty.")

    def test_raw_csv_is_loadable(self):
        """Test that the CSV can be loaded into a DataFrame"""
        df = pd.read_csv(RAW_DATA_PATH)
        assert len(df) > 0, "Dataset loaded but contains zero rows."

   