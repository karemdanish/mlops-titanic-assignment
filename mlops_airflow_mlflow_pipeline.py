"""
MLOps Assignment 2: End-to-End ML Pipeline with Airflow and MLflow
Titanic Survival Prediction
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import os
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'danish',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'mlops_titanic_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for Titanic survival prediction',
    schedule_interval=None,  # Only manual trigger
    catchup=False,
    tags=['mlops', 'titanic'],
)

# Path to Titanic dataset (update this to your actual path)
DATA_PATH = '/home/dell/mlops-assignment2/titanic.csv'  # Adjust if needed

# Task 1: Start
def start():
    print("Starting the pipeline...")

start_task = PythonOperator(
    task_id='start_pipeline',
    python_callable=start,
    dag=dag,
)

# Task 2: Data Ingestion
def ingest_data(**context):
    import pandas as pd
    import logging
    from airflow.exceptions import AirflowException

    data_path = DATA_PATH
    logging.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise AirflowException(f"Dataset not found at {data_path}")
    
    # Print dataset shape
    shape = df.shape
    logging.info(f"Dataset shape: {shape}")
    print(f"Dataset shape: {shape}")
    
    # Log missing values count per column
    missing_counts = df.isnull().sum()
    logging.info("Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            logging.info(f"  {col}: {count}")
            print(f"  {col}: {count}")
    
    # Push dataset path to XCom
    context['ti'].xcom_push(key='data_path', value=data_path)
    
    return "Data ingestion completed."

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    provide_context=True,
    dag=dag,
)

# Task 3: Data Validation (with retry demonstration)
def validate_data(**context):
    import pandas as pd
    import logging

    data_path = context['ti'].xcom_pull(task_ids='ingest_data', key='data_path')
    df = pd.read_csv(data_path)
    
    # Check missing percentage in Age and Embarked
    missing_age_pct = df['Age'].isnull().mean() * 100
    missing_embarked_pct = df['Embarked'].isnull().mean() * 100
    
    logging.info(f"Missing Age: {missing_age_pct:.2f}%")
    logging.info(f"Missing Embarked: {missing_embarked_pct:.2f}%")
    
    # Raise exception if missing > 30%
    if missing_age_pct > 30 or missing_embarked_pct > 30:
        raise Exception(f"Missing values too high: Age {missing_age_pct:.2f}%, Embarked {missing_embarked_pct:.2f}%")
    
    print("Validation passed.")
    return "Validation passed."

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    provide_context=True,
    retries=3,  # Demonstrate retry
    retry_delay=timedelta(seconds=10),
    dag=dag,
)

# Task 4: Parallel Processing
# 4a. Handle missing values
def handle_missing_values(**context):
    import pandas as pd
    import logging

    data_path = context['ti'].xcom_pull(task_ids='ingest_data', key='data_path')
    df = pd.read_csv(data_path)
    
    # Fill missing Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Save processed data temporarily (or push to XCom, but XCom has size limits)
    # Instead, save to a temporary file and push the path
    temp_path = '/tmp/titanic_missing_handled.csv'
    df.to_csv(temp_path, index=False)
    context['ti'].xcom_push(key='missing_handled_path', value=temp_path)
    
    logging.info("Missing values handled and saved to temp file.")
    return "Missing values handled."

missing_task = PythonOperator(
    task_id='handle_missing_values',
    python_callable=handle_missing_values,
    provide_context=True,
    dag=dag,
)

# 4b. Feature engineering
def feature_engineering(**context):
    import pandas as pd
    import logging

    data_path = context['ti'].xcom_pull(task_ids='ingest_data', key='data_path')
    df = pd.read_csv(data_path)
    
    # Create FamilySize and IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Save engineered data temporarily
    temp_path = '/tmp/titanic_feature_engineered.csv'
    df.to_csv(temp_path, index=False)
    context['ti'].xcom_push(key='engineered_path', value=temp_path)
    
    logging.info("Feature engineering completed and saved to temp file.")
    return "Feature engineering completed."

feature_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
)

# Task 5: Data Encoding (depends on both parallel tasks)
def encode_features(**context):
    import pandas as pd
    import logging

    # Pull paths from both parallel tasks
    missing_path = context['ti'].xcom_pull(task_ids='handle_missing_values', key='missing_handled_path')
    engineered_path = context['ti'].xcom_pull(task_ids='feature_engineering', key='engineered_path')
    
    # Load both dataframes (they have same rows but different columns)
    df_missing = pd.read_csv(missing_path)
    df_engineered = pd.read_csv(engineered_path)
    
    # Merge on common columns (or just use one and add features from the other)
    # For simplicity, we'll use the missing-handled df and add the new features
    df = df_missing.copy()
    df['FamilySize'] = df_engineered['FamilySize']
    df['IsAlone'] = df_engineered['IsAlone']
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Drop irrelevant columns (e.g., Name, Ticket, Cabin, PassengerId)
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # Save final processed data
    final_path = '/tmp/titanic_final.csv'
    df.to_csv(final_path, index=False)
    context['ti'].xcom_push(key='final_data_path', value=final_path)
    
    logging.info("Encoding completed and final data saved.")
    return "Encoding completed."

encode_task = PythonOperator(
    task_id='encode_features',
    python_callable=encode_features,
    provide_context=True,
    dag=dag,
)

# Task 6: Model Training with MLflow
def train_model(**context):
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import logging
    import json
    from airflow.models import Variable

    # Read parameters from Airflow Variable
    try:
        params_json = Variable.get("model_params", default_var='{"model_type":"RandomForest","n_estimators":100,"max_depth":5}')
        params = json.loads(params_json)
        logging.info(f"Using Variable params: {params}")
    except Exception as e:
        logging.error(f"Failed to read Variable, using defaults: {e}")
        params = {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 5}

    # Load final data
    final_path = context['ti'].xcom_pull(task_ids='encode_features', key='final_data_path')
    df = pd.read_csv(final_path)
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    mlflow.set_experiment("Titanic_Experiment")
    with mlflow.start_run():
        # Log all parameters from params
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Choose model based on params
        model_type = params.get("model_type", "RandomForest")
        if model_type.lower() == "randomforest":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                random_state=42
            )
        elif model_type.lower() == "logisticregression":
            model = LogisticRegression(
                C=params.get("C", 1.0),
                solver=params.get("solver", "liblinear"),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Log dataset sizes
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model and test data for evaluation
        import joblib
        model_path = "/tmp/titanic_model.pkl"
        joblib.dump(model, model_path)
        context['ti'].xcom_push(key='model_path', value=model_path)
        X_test.to_csv('/tmp/X_test.csv', index=False)
        y_test.to_csv('/tmp/y_test.csv', index=False)
        context['ti'].xcom_push(key='X_test_path', value='/tmp/X_test.csv')
        context['ti'].xcom_push(key='y_test_path', value='/tmp/y_test.csv')
        context['ti'].xcom_push(key='mlflow_run_id', value=mlflow.active_run().info.run_id)
    
    logging.info("Model training completed and logged to MLflow.")
    return "Training completed."

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

# Task 7: Model Evaluation
def evaluate_model(**context):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import mlflow
    import logging

    model_path = context['ti'].xcom_pull(task_ids='train_model', key='model_path')
    X_test_path = context['ti'].xcom_pull(task_ids='train_model', key='X_test_path')
    y_test_path = context['ti'].xcom_pull(task_ids='train_model', key='y_test_path')
    
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    
    # Log metrics to MLflow (resume the run)
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='mlflow_run_id')
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
    
    # Push accuracy for branching
    context['ti'].xcom_push(key='accuracy', value=accuracy)
    
    return f"Evaluation completed. Accuracy: {accuracy}"

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

# Task 8: Branching Logic
def decide_branch(**context):
    accuracy = context['ti'].xcom_pull(task_ids='evaluate_model', key='accuracy')
    if accuracy >= 0.80:
        return 'register_model'
    else:
        return 'reject_model'

branch_task = BranchPythonOperator(
    task_id='branch_decision',
    python_callable=decide_branch,
    provide_context=True,
    dag=dag,
)

# Task 9: Model Registration (if approved)
def register_model(**context):
    import mlflow
    from mlflow.tracking import MlflowClient
    import logging

    run_id = context['ti'].xcom_pull(task_ids='train_model', key='mlflow_run_id')
    accuracy = context['ti'].xcom_pull(task_ids='evaluate_model', key='accuracy')
    
    client = MlflowClient()
    model_name = "Titanic_Survival_Model"
    
    # Register the model
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    logging.info(f"Model registered as version {result.version} in '{model_name}'")
    
    # Optionally, transition to "Staging" or "Production" (not required)
    
    return f"Model registered with accuracy {accuracy}"

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    provide_context=True,
    dag=dag,
)

# Task 9 (reject branch)
def reject_model(**context):
    accuracy = context['ti'].xcom_pull(task_ids='evaluate_model', key='accuracy')
    rejection_reason = f"Accuracy {accuracy:.4f} is below threshold (0.80)"
    logging.info(f"Model rejected: {rejection_reason}")
    print(f"Rejection reason: {rejection_reason}")
    return f"Model rejected: {rejection_reason}"

reject_task = PythonOperator(
    task_id='reject_model',
    python_callable=reject_model,
    provide_context=True,
    dag=dag,
)

# Task 10: End
end_task = DummyOperator(
    task_id='end_pipeline',
    trigger_rule=TriggerRule.NONE_FAILED,
    dag=dag,
)

# Set dependencies to create the DAG graph
start_task >> ingest_task >> validate_task
validate_task >> [missing_task, feature_task]  # Parallel tasks
[missing_task, feature_task] >> encode_task
encode_task >> train_task >> evaluate_task
evaluate_task >> branch_task
branch_task >> [register_task, reject_task]
[register_task, reject_task] >> end_task