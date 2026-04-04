# Tests for cleaned dataset validation.

import os
import pandas as pd
import pytest


CLEAN_DATA_PATH = os.path.join("data", "processed", "clean_data.csv")

@pytest.mark.skip(reason="Data not on GitHub. Waiting to inject Kaggle Secrets into CI.")
class TestCleanDataQuality:
    # Verify that cleaning steps were applied correctly.

    def test_no_null_values(self):
        df = pd.read_csv(CLEAN_DATA_PATH)
        assert df.isnull().sum().sum() == 0, "Clean data still contains null values"

    def test_customer_id_dropped(self):
        df = pd.read_csv(CLEAN_DATA_PATH)
        assert "customerID" not in df.columns, "customerID should be dropped"

    def test_total_charges_is_numeric(self):
        df = pd.read_csv(CLEAN_DATA_PATH)
        assert df["TotalCharges"].dtype in ["float64", "int64"], (
            "TotalCharges should be numeric after cleaning"
        )

    def test_churn_column_exists(self):
        df = pd.read_csv(CLEAN_DATA_PATH)
        assert "Churn" in df.columns, "Target column Churn is missing"

    def test_row_count_is_reasonable(self):
        df = pd.read_csv(CLEAN_DATA_PATH)
        assert len(df) > 7000, "Too many rows dropped during cleaning"