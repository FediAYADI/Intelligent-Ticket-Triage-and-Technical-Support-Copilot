"""Airflow DAG: ticket triage pipeline.

Schedule: daily ingestion+cleaning, weekly retrain, hourly KB reindex.

Deploy: copy this file to $AIRFLOW_HOME/dags/ and set PYTHONPATH to the
project root (or install the project as a package). All tasks call the
project's CLI entry points via BashOperator.
"""
from __future__ import annotations

import datetime as dt

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/ticket-triage"  # adjust to your deployment path
PY = "python"

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=5),
}

dag = DAG(
    dag_id="ticket_triage_pipeline",
    description="Ingest, clean, retrain classifier, reindex RAG KB, detect anomalies",
    start_date=dt.datetime(2026, 1, 1),
    schedule_interval="0 6 * * *",   # daily 06:00
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["triage", "mlops"],
)

ingest = BashOperator(
    task_id="ingest_and_clean",
    bash_command=f"cd {PROJECT_DIR} && {PY} -m src.data_processing.pipeline",
    dag=dag,
)

train = BashOperator(
    task_id="train_classifier",
    bash_command=f"cd {PROJECT_DIR} && {PY} -m src.classification.train",
    dag=dag,
)

reindex = BashOperator(
    task_id="reindex_rag_kb",
    bash_command=f"cd {PROJECT_DIR} && {PY} -m src.rag.indexer",
    dag=dag,
)

anomalies = BashOperator(
    task_id="detect_anomalies",
    bash_command=f"cd {PROJECT_DIR} && {PY} -m src.anomaly_detection.detector",
    dag=dag,
)

evidently = BashOperator(
    task_id="evidently_drift_report",
    bash_command=f"cd {PROJECT_DIR} && {PY} -m src.monitoring.evidently_report",
    dag=dag,
)

# weekly retrain variant (Sundays) — same task, separate schedule doc:
# In production, trigger `train` on a weekly DAG or via a second DAG with
# schedule_interval="0 6 * * 0". Kept in one file for readability.

with dag:
    ingest >> train >> reindex >> anomalies >> evidently
