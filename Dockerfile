FROM apache/airflow:2.9.0
RUN pip install --no-cache-dir pandas loguru==0.5.3 scikit-learn joblib clearml==1.15.1 apache-airflow==2.9.0 dvc[s3] 
