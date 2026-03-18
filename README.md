# mlops-churn-pipeline
End-to-end MLOps pipeline for telecom customer churn prediction

## System Architecture

Data flows through the following pipeline:
```
Raw Dataset (Kaggle)
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
API Deployment
```
## Project Structure
```
mlops-churn-pipeline
│
├── data
│   ├── raw
│   ├── processed
│   └── features
│
├── src
│   ├── ingest_data.py
│   ├── validate_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── split_data.py
│   ├── train_model.py
│   ├── tune_model.py
│   └── evaluate_model.py
│
├── api
│   └── app.py
│
├── models
├── logs
├── docker
└── .github/workflows
```
## Data Engineering Pipeline

The data engineering stage prepares the dataset for machine learning.

Steps:

1. **Data Ingestion**  
   Downloads the Telco Customer Churn dataset from Kaggle.

2. **Data Validation**  
   Checks dataset quality:
   - missing values
   - duplicate rows

3. **Data Cleaning**  
   - converts numeric fields
   - removes invalid values
   - drops unnecessary columns

4. **Feature Engineering**  
   - encodes categorical variables
   - normalizes numeric columns

Final dataset:

data/features/features.csv

## Team Roles

Fatima – Data Engineering  
Responsible for:
- data ingestion
- data validation
- data cleaning
- feature engineering

Navneeth – Machine Learning  
Responsible for:
- train/test split
- model training
- hyperparameter tuning
- model evaluation

Anirudh – MLOps  
Responsible for:
- MLflow experiment tracking
- model registry
- CI/CD pipeline

Sarah – Deployment  
Responsible for:
- API development
- Docker containerization
- prediction serving

## Dataset

Telco Customer Churn Dataset  
Source: Kaggle

https://www.kaggle.com/datasets/blastchar/telco-customer-churn
