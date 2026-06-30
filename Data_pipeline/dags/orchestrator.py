"""
dags/orchestrator.py
--------------------
Mental Health ETL pipeline.

Task 1  scrape_and_clean   → script1_scrape_clean.py
Task 2  vectorise_and_save → script2_vectorise_save.py   (runs after Task 1)

Each task dynamically loads its script from /opt/airflow/scripts/ (mounted
from the host ./scripts/ folder) and calls the script's run() function.
This means you can edit scripts on the host without rebuilding the image.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Helper: load a .py file from disk and call its run() function
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path("/opt/airflow/scripts")


def _load_and_run(script_name: str) -> None:
    """Dynamically import *script_name* from the mounted scripts folder and
    call its ``run()`` entry point.  Using importlib keeps each task isolated
    even if both scripts share a module name."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so relative imports inside the script work
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise AttributeError(f"{script_name} must expose a top-level run() function")

    module.run()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=60),
    "email_on_failure": False,
}

with DAG(
    dag_id="mental_health_etl",
    default_args=default_args,
    description="Beyond Blue scrape → clean → encode → store in Azure Blob",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mental-health", "etl", "beyondblue"],
) as dag:

    task_scrape = PythonOperator(
        task_id="scrape_and_clean",
        python_callable=_load_and_run,
        op_args=["script1_scrape_clean.py"],
        doc_md="""
        **Task 1 — Scrape & Clean**

        Scrapes Beyond Blue forum posts using Selenium, cleans text
        (emoji / emoticon conversion, normalisation), and uploads the
        result as a Parquet file to Azure Blob Storage under `raw/`.
        """,
    )

    task_vectorise = PythonOperator(
        task_id="vectorise_and_save",
        python_callable=_load_and_run,
        op_args=["script2_vectorise_save.py"],
        doc_md="""
        **Task 2 — Vectorise & Save**

        Reads the latest Parquet from Azure Blob, encodes posts into
        sentence embeddings, builds a FAISS index, and saves the index
        + metadata pickle back to Blob under `faiss/`.
        """,
    )

    # Task 1 must complete before Task 2 starts
    task_scrape >> task_vectorise