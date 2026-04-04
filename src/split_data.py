# Split feature-engineered data into train/test sets.

import pandas as pd
import os
from sklearn.model_selection import train_test_split


# File paths
FEATURES_PATH = os.path.join("data", "features", "features.csv")
OUTPUT_DIR = os.path.join("data", "splits")


def load_features():
    # Load the feature-engineered dataset.
    print("Loading features from:", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH)
    print("Dataset shape:", df.shape)
    return df


def split_data(df, test_size=0.2, random_state=42):
    # Split data into train and test sets.
    # Uses a fixed random_state for reproducibility every team member gets the same split.
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Train set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set:  {X_test.shape[0]} rows, {X_test.shape[1]} features")

    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test):
    # Save train/test splits to CSV files.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    print("Splits saved to:", OUTPUT_DIR)


def run_split_pipeline():
    # Run the full split pipeline.
    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_split_pipeline()