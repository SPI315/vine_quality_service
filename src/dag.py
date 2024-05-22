from datetime import timedelta, datetime
from loguru import logger

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator

from data import prepare_data
from model import train, test, check_metrics
from checker import check_files

local_path = "/src"
logger.info(f"Sys path is {local_path}")
data_path = "/app/data"
logger.info(f"Data saved in {data_path}")
model_path = "/app/model/model.pkl"
logger.info(f"Model saved in {model_path}")
metrics_path = "/app/metrics"
logger.info(f"Metrics saved in {metrics_path}")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["your.email@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "my_dag",
    default_args=default_args,
    description="WineQuality",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 4, 14),
    tags=["WineQuality"],
)

check_data_task = ShortCircuitOperator(
    task_id="check_data",
    python_callable=check_files,
    dag=dag,
)

pull_data_task = BashOperator(
    task_id="pull_data",
    bash_command="cd /app && dvc pull",
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    op_kwargs={"path": data_path},
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train,
    op_kwargs={"model_path": model_path, "data_path": data_path},
    dag=dag,
)

test_model_task = PythonOperator(
    task_id="test_model",
    python_callable=test,
    op_kwargs={
        "model_path": model_path,
        "data_path": data_path,
        "metrics_path": metrics_path,
    },
    dag=dag,
)

check_metrics_task = ShortCircuitOperator(
    task_id="check_metrics",
    python_callable=check_metrics,
    op_kwargs={
        "metrics_path": metrics_path,
    },
    dag=dag,
)

update_model_task = BashOperator(
    task_id="update_model",
    bash_command="cd /app && dvc add model && dvc add data && dvc push",
    dag=dag,
)

(
    check_data_task
    >> pull_data_task
    >> prepare_data_task
    >> train_model_task
    >> test_model_task
    >> check_metrics_task
    >> update_model_task
)
