import json
import logging
import os  # Pour os.path.exists et os.environ
from datetime import datetime

import pandas as pd
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator


# Plugin custom (doit être placé dans plugins/s3_to_postgres.py)
from s3_to_postgres import S3ToPostgresOperator


default_args = {
    "owner": "airflow",
    "start_date": datetime(2022, 6, 1),
}

# Coordonnées de Paris
LAT = 48.8566
LON = 2.3522

def setup_aws_environment():
    """Configure les credentials AWS via Variables Airflow (similaire au DAG ML)"""
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"] = Variable.get("AWS_DEFAULT_REGION")
        logging.info("✓ AWS environment configured from Airflow Variables")
    except Exception as e:
        logging.error(f"❌ AWS setup failed: {str(e)}")
        raise

# Appeler l’API OpenWeatherMap pour obtenir les données météo actuelles de Paris.
# Le JSON reste local (pas d'upload S3 redondant).

def _fetch_weather_data(**context):
    """Appelle l'API OpenWeatherMap et retourne le JSON local"""
    logging.info("Fetching weather data from OpenWeatherMap")
    api_key = Variable.get("OPEN_WEATHER_API_KEY")  # Assumé comme Variable Airflow

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={api_key}&units=metric"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Erreur API : {resp.status_code} - {resp.text}")

    filename = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}_weather.json"
    local_path = f"/tmp/{filename}"
   
    with open(local_path, "w") as f:
        json.dump(resp.json(), f)
    
    # Push le chemin local du JSON via XCom (pour la tâche suivante)
    context["ti"].xcom_push(key="local_json_path", value=local_path)
    logging.info(f"JSON saved locally: {local_path}")
    # Pas d'upload S3 pour JSON → allégé !

# Télécharger/transformer le JSON local, append au CSV existant sur S3, et upload le CSV mis à jour.
def _transform_and_append_weather_data(**context):
    """Transforme le JSON en ligne, append au CSV S3, et upload le CSV final"""
    # Setup AWS en premier (pour S3Hook)
    setup_aws_environment()
    
    bucket = Variable.get("BUCKET")
    csv_key = "weather_paris_fect.csv"  # Key fixe pour accumulation
    
    # Pull le chemin JSON local via XCom
    local_json = context["ti"].xcom_pull(task_ids="fetch_weather_data", key="local_json_path")
    
    if not local_json or not os.path.exists(local_json):
        raise ValueError("Impossible de récupérer le JSON local")
    
    logging.info(f"DEBUG: Processing {local_json} | Bucket: {bucket} | CSV Key: {csv_key}")
    
    # Téléchargement du JSON local
    with open(local_json, "r") as f:
        raw_data = json.load(f)

    # Mapping direct vers colonnes compatibles ML (OpenWeatherMap natif)
    new_row = {
        "datetime": pd.to_datetime(raw_data["dt"], unit="s"),  # dt est timestamp UTC
        "temp": raw_data["main"]["temp"],
        "feels_like": raw_data["main"]["feels_like"],
        "pressure": raw_data["main"]["pressure"],
        "humidity": raw_data["main"]["humidity"],
        "dew_point": None,  # Non direct → NaN (calculable mais simplifié)
        "clouds": raw_data["clouds"]["all"],
        "visibility": raw_data.get("visibility", None),  # En mètres
        "wind_speed": raw_data["wind"]["speed"],
        "wind_deg": raw_data["wind"]["deg"],
        "rain_1h": raw_data.get("rain", {}).get("1h", 0.0),  # 0 si absent
        "weather_main": raw_data["weather"][0]["main"],
        "weather_description": raw_data["weather"][0]["description"],
    }
    new_df = pd.DataFrame([new_row])

    # Download CSV existant de S3 (ou gérer premier run)
    local_csv = f"/tmp/{csv_key}"
    try:
        S3Hook(aws_conn_id="aws_default").download_file(key=csv_key, bucket_name=bucket, local_path="/tmp")
        existing_df = pd.read_csv(local_csv)
        logging.info(f"CSV existant chargé : {len(existing_df)} lignes")
        
        # Check doublon : éviter append si datetime déjà présent (convert to str pour comparaison)
        new_datetime_str = new_df['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        existing_datetimes = pd.to_datetime(existing_df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if new_datetime_str in existing_datetimes.values:
            logging.warning("Ligne déjà présente (doublon évité)")
            existing_df.to_csv(local_csv, index=False, header=True)  # Ré-upload inchangé
            updated_df = existing_df  # Correction: set updated_df ici
        else:
            # Append
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(local_csv, index=False, header=True)
            logging.info(f"Nouvelle ligne appendée : total {len(updated_df)} lignes")
    except Exception as e:  # Premier run ou fichier absent
        logging.info(f"Premier run ou CSV absent : création avec {len(new_df)} lignes | Erreur: {e}")
        new_df.to_csv(local_csv, index=False, header=True)
        updated_df = new_df  # Pour logs

    # Upload du CSV mis à jour sur S3
    S3Hook(aws_conn_id="aws_default").load_file(
        filename=local_csv, key=csv_key, bucket_name=bucket, replace=True
    )

    # Push la key CSV fixe via XCom (pour DB)
    context["ti"].xcom_push(key="weather_csv_key", value=csv_key)
    logging.info(f"CSV mis à jour uploadé sur S3: s3://{bucket}/{csv_key}")
    logging.info(f"Colonnes: {list(updated_df.columns)}")

with DAG(
    dag_id="meteo_01_etl",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["weather"],
) as dag:

    fetch_weather_data = PythonOperator(
        task_id="fetch_weather_data", 
        python_callable=_fetch_weather_data
    )

    transform_and_append_weather_data = PythonOperator(
        task_id="transform_and_append_weather_data", 
        python_callable=_transform_and_append_weather_data
    )
    
    # Créer la table avec UNIQUE sur datetime pour éviter doublons à l'insert.
    create_weather_table = SQLExecuteQueryOperator(
        task_id="create_weather_table",
        sql="""
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            datetime TIMESTAMP UNIQUE,  -- UNIQUE pour dedup
            temp DECIMAL(5, 2),
            feels_like DECIMAL(5, 2),
            pressure DECIMAL(5, 2),
            humidity DECIMAL(5, 2),
            clouds DECIMAL(5, 2),
            visibility DECIMAL(5, 2),
            wind_speed DECIMAL(5, 2),
            wind_deg DECIMAL(5, 2),
            rain_1h DECIMAL(5, 2),
            weather_main VARCHAR,
            weather_description VARCHAR
        )
        """,
        conn_id="neon_db_conn",
    )
    
    # Insert du CSV full, mais avec dedup via UNIQUE constraint (assume custom operator gère INSERT ... ON CONFLICT DO NOTHING)
    transfer_weather_data_to_postgres = S3ToPostgresOperator(
        task_id="transfer_weather_data_to_postgres",
        table="weather_data",
        bucket="{{ var.value.BUCKET }}",
        key="{{ ti.xcom_pull(task_ids='transform_and_append_weather_data', key='weather_csv_key') }}",
        postgres_conn_id="neon_db_conn",
        aws_conn_id="aws_default",
    )

    trigger_evidently = TriggerDagRunOperator(
        task_id="meteo_02_evidently",
        trigger_dag_id="meteo_02_evidently",
        wait_for_completion=False,  # mettre True si tu veux que ce DAG attende sa fin
        reset_dag_run=True,
    )



    # Flux sans trigger_ml_dag
    fetch_weather_data >> transform_and_append_weather_data >> create_weather_table >> transfer_weather_data_to_postgres >> trigger_evidently