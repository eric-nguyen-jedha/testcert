# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

import mlflow
import boto3

# ----------------------------
# Configuration DAG
# ----------------------------
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    'meteo_04_real_time_weather',
    default_args=default_args,
    description='Prédiction météo en temps réel avec MLflow et OpenWeather API',
    schedule=None,
    catchup=False,
    tags=['ml', 'weather', 'prediction', 'real-time', 'openweather', 'mlflow'],
)

# ----------------------------
# Variables Airflow / config
# ----------------------------
CITIES = {
    'paris': {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
    'toulouse': {'lat': 43.6047, 'lon': 1.4442, 'name': 'Toulouse'},
    'lyon': {'lat': 45.7640, 'lon': 4.8357, 'name': 'Lyon'},
    'marseille': {'lat': 43.2965, 'lon': 5.3698, 'name': 'Marseille'},
    'nantes': {'lat': 47.2184, 'lon': -1.5536, 'name': 'Nantes'},
    'bordeaux': {'lat': 44.833328, 'lon': -0.56667, 'name': 'Bordeaux'},
    'strasbourg': {'lat': 48.5734053, 'lon': 7.7521113, 'name': 'Strasbourg'},
    'brest': {'lat': 48.400002, 'lon': -4.48333, 'name': 'Brest'},
    'roubaix': {'lat': 50.683636, 'lon': 3.165532, 'name': 'Roubaix'},
    'biarritz': {'lat': 43.48333, 'lon': -1.56667, 'name': 'Biarritz'},
    'clermont-ferrand': {'lat': 45.783329, 'lon': 3.08333, 'name': 'Clermont Ferrand'},
    'besançon': {'lat': 47.25, 'lon': 6.03333, 'name': 'Besançon'},
    'ajaccio': {'lat': 41.919229, 'lon': 8.738635, 'name': 'Ajaccio'},
}

BUCKET = Variable.get("BUCKET")

# Mapping des codes vers les labels météo
# Basé sur l'ordre alphabétique : Clear, Clouds, Fog, Rain, Snow
WEATHER_CODE_MAPPING = {
    0: 'Clear',
    1: 'Clouds',
    2: 'Fog',
    3: 'Rain',
    4: 'Snow'
}

# ----------------------------
# Fonctions utilitaires
# ----------------------------
def setup_aws_environment():
    """Configure les credentials AWS depuis les Variables Airflow"""
    os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = Variable.get("AWS_DEFAULT_REGION")

def setup_mlflow():
    mlflow_uri = Variable.get("mlflow_uri", default_var="https://ericjedha-aia.hf.space/")
    os.environ["APP_URI"] = mlflow_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Meteo")

def fetch_weather(lat, lon):
    api_key = Variable.get("OPEN_WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Erreur API : {resp.status_code} - {resp.text}")
    return resp.json()

def preprocess_weather_json(raw_data, model_type='historical'):
    """
    Prétraite les données météo de l'API OpenWeather
    
    Args:
        raw_data: JSON retourné par l'API OpenWeather
        model_type: 'historical' ou 'forecast_6h' (même features pour les deux maintenant)
    """
    timestamp_dt = pd.to_datetime(raw_data["dt"], unit="s")
    
    # Features météo de base
    features = {
        "temp": raw_data["main"]["temp"],
        "feels_like": raw_data["main"]["feels_like"],
        "pressure": raw_data["main"]["pressure"],
        "humidity": raw_data["main"]["humidity"],
        "clouds": raw_data["clouds"]["all"],
        "visibility": raw_data.get("visibility", None),
        "wind_speed": raw_data["wind"]["speed"],
        "wind_deg": raw_data["wind"]["deg"],
        "rain_1h": raw_data.get("rain", {}).get("1h", 0.0)
    }
    
    # Features temporelles
    hour = timestamp_dt.hour
    month = timestamp_dt.month
    weekday = timestamp_dt.weekday()
    
    features["hour"] = hour
    features["month"] = month
    features["weekday"] = weekday
    features["is_weekend"] = int(weekday in [5, 6])
    
    # Features cycliques
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # ✅ MÊME ordre de colonnes pour les deux modèles (17 features, SANS timestamp ni dew_point)
    column_order = [
        "temp", "feels_like", "pressure", "humidity", "clouds",
        "visibility", "wind_speed", "wind_deg", "rain_1h",
        "hour", "month", "weekday", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]
    
    df = pd.DataFrame([features])[column_order]
    
    # Forcer les types numériques pour éviter les problèmes avec XGBoost
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def upload_to_s3(local_file, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, BUCKET, s3_key)

# ----------------------------
# Fonction générique de prédiction
# ----------------------------
def predict_weather(model_uri, output_filename, model_type='historical'):
    """
    Fonction générique pour faire des prédictions météo
    
    Args:
        model_uri: URI du modèle MLflow (ex: 'runs:/xxx/model')
        output_filename: Nom du fichier CSV de sortie
        model_type: 'historical' ou 'forecast_6h'
    """
    setup_aws_environment()
    setup_mlflow()
    
    print(f"🔄 Chargement du modèle {model_type}...")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # ✅ Essayer de charger le LabelEncoder depuis plusieurs sources
    import tempfile
    import pickle
    
    label_encoder = None
    label_encoder_name = "label_encoder_historical.pkl" if model_type == 'historical' else "label_encoder_6h.pkl"
    
    # Méthode 1 : Depuis MLflow
    try:
        run_id = model_uri.split('/')[1]
        client = mlflow.tracking.MlflowClient()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, label_encoder_name, tmp_dir)
            with open(local_path, 'rb') as f:
                label_encoder = pickle.load(f)
        print(f"✅ LabelEncoder chargé depuis MLflow : {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"⚠️  Impossible de charger depuis MLflow: {e}")
        
        # Méthode 2 : Depuis un fichier local (si le DAG d'entraînement a tourné)
        try:
            local_model_path = f"/opt/airflow/models/{label_encoder_name}"
            if os.path.exists(local_model_path):
                with open(local_model_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                print(f"✅ LabelEncoder chargé depuis fichier local : {list(label_encoder.classes_)}")
        except Exception as e2:
            print(f"⚠️  Impossible de charger depuis fichier local: {e2}")
            
            # Méthode 3 : Hardcodé en dernier recours (à adapter selon vos classes)
            print("⚠️  Utilisation du mapping hardcodé (à mettre à jour selon vos données)")
            class FakeLabelEncoder:
                def __init__(self):
                    # ⚠️ ADAPTER CES CLASSES SELON VOS DONNÉES RÉELLES
                    self.classes_ = np.array(['Clear', 'Clouds', 'Rain', 'Fog'])
                
                def inverse_transform(self, codes):
                    return [self.classes_[int(code)] if int(code) < len(self.classes_) else f"Unknown_{code}" for code in codes]
            
            label_encoder = FakeLabelEncoder()

    results = []
    for city, info in CITIES.items():
        print(f"📍 Prédiction pour {info['name']}...")
        raw_data = fetch_weather(info['lat'], info['lon'])
        df = preprocess_weather_json(raw_data, model_type=model_type)
        
        pred_encoded = model.predict(df)[0]
        
        # ✅ Décoder la prédiction
        if label_encoder is not None:
            try:
                pred_weather = label_encoder.inverse_transform([int(pred_encoded)])[0]
            except Exception as e:
                print(f"⚠️  Erreur lors du décodage: {e}")
                pred_weather = WEATHER_CODE_MAPPING.get(int(pred_encoded), f"Code_{int(pred_encoded)}")
        else:
            # Utiliser le mapping par défaut
            pred_weather = WEATHER_CODE_MAPPING.get(int(pred_encoded), f"Unknown_{int(pred_encoded)}")
        
        results.append({
            "ville": info['name'],
            "lat": info['lat'],
            "lon": info['lon'],
            "prediction": pred_weather,
            "prediction_code": int(pred_encoded),
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        print(f"  → {pred_weather} (code: {int(pred_encoded)})")

    df_out = pd.DataFrame(results)
    local_file = f"/tmp/{output_filename}"
    df_out.to_csv(local_file, index=False)
    upload_to_s3(local_file, output_filename)
    
    print(f"✅ Prédictions {model_type} sauvegardées : {output_filename}")
    print(f"   Exemples: {df_out[['ville', 'prediction']].to_dict('records')}")

# ----------------------------
# Définition des tâches
# ----------------------------
historical_task = PythonOperator(
    task_id='historical',
    python_callable=lambda **kwargs: predict_weather(
        'runs:/217007d943dd4c20988bd54ff0227be3/model', 
        'historical.csv',
        model_type='historical'
    ),
    dag=dag,
)

forecast_6h_task = PythonOperator(
    task_id='forecast_6h',
    python_callable=lambda **kwargs: predict_weather(
        'runs:/bf2b6f0c84d245f0a6eac3d60ad0a23d/model',  # ← Remplacez par le nouveau run ID après réentraînement
        'forecast_6h.csv',
        model_type='forecast_6h'
    ),
    dag=dag,
)

# ----------------------------
# Opérateur d'envoi d'e-mail personnalisé
# ----------------------------
from airflow.models import BaseOperator
from airflow.hooks.base import BaseHook
import smtplib
from email.mime.text import MIMEText

class GmailEmailOperator(BaseOperator):
    def __init__(self, to: str, subject: str, html_content: str, conn_id: str = "smtp_default", **kwargs):
        super().__init__(**kwargs)
        self.to = to
        self.subject = subject
        self.html_content = html_content
        self.conn_id = conn_id

    def execute(self, context):
        conn = BaseHook.get_connection(self.conn_id)
        smtp_host = conn.extra_dejson.get("smtp_host", "smtp.gmail.com")
        smtp_port = conn.extra_dejson.get("smtp_port", 587)
        use_tls = conn.extra_dejson.get("smtp_starttls", True)

        msg = MIMEText(self.html_content, "html")
        msg["Subject"] = self.subject
        msg["From"] = f"Airflow  <{conn.login}>"
        msg["To"] = self.to

        server = smtplib.SMTP(smtp_host, smtp_port)
        if use_tls:
            server.starttls()
        server.login(conn.login, conn.password)
        server.send_message(msg)
        server.quit()
        self.log.info(f"✅ E-mail envoyé à {self.to}")

# ----------------------------
# Tâche de notification finale
# ----------------------------
send_success_email = GmailEmailOperator(
    task_id="send_weather_update_email",
    to="en9.eric@gmail.com",
    subject="- 𖣘 - Airflow : Météo France à jour",
    #from_email="Airflow Météo <enguyen.fr@gmail.com>",  # 👈 Ajoutez cette ligne
    html_content="""
    <H1> 𖣘 Airflow </h1>
    <h2> Toutes les DAGs exécutées avec réussite ✅️ </H2>
    <H2>La météo du moment et son forecast +6h est à consulter ici :
    🌦️ <a href="https://huggingface.co/spaces/ericjedha/meteo-france"><strong> Météo France </strong></a></H2>
    <p><em>Votre Service Météo</em></p>
    """,
    dag=dag,
)




historical_task >> forecast_6h_task >> send_success_email
