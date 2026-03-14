# MLOps Assignment 2: Titanic Survival Prediction

## Overview
This project implements an end-to-end ML pipeline using Apache Airflow and MLflow. The pipeline predicts survival on the Titanic dataset.

## Prerequisites
- Python 3.12+
- Apache Airflow 2.9.3
- MLflow
- Other dependencies listed in `requirements.txt`

## Setup
1. Clone this repository.
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set Airflow home: `export AIRFLOW_HOME=$(pwd)/airflow`
6. Initialize Airflow database: `airflow db init`
7. Create an admin user: `airflow users create --username admin --firstname ...`
8. Copy the DAG file to `$AIRFLOW_HOME/dags/` (or symlink).
9. Download the Titanic dataset: `wget -O titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
10. Start Airflow services:
    - `airflow webserver -p 8080`
    - `airflow scheduler`
11. Start MLflow UI: `mlflow ui`

## Usage
- In Airflow UI, trigger the DAG manually. To change hyperparameters, edit the Airflow Variable `model_params` (Admin → Variables) with a JSON config (e.g., `{"model_type": "RandomForest", "n_estimators": 150, "max_depth": 8}`).
- View experiment results in MLflow UI at `http://localhost:5000`.

## Deliverables
- DAG Python file
- requirements.txt
- Screenshots (in `screenshots/` folder)
- Technical report (PDF)
