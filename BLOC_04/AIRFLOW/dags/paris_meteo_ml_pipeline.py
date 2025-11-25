# -*- coding: utf-8 -*-
# Pipeline ML Airflow - Exercice : modèle historique + prévision à 6h

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import boto3
from airflow.exceptions import AirflowSkipException


# Configuration
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'meteo_03_ml_pipeline',
    default_args=default_args,
    description='Modèle historique + prévision à 6h (exercice)',
    schedule=None,
    catchup=False,
    tags=['ml', 'weather', 'xgboost', 'mlflow', 'exercise'],
)

# Chemins
DATA_PATH = '/opt/airflow/data'
MODEL_PATH = '/opt/airflow/models'
WEATHER_CSV_FILE = 'weather_paris.csv'


def setup_environment():
    """Configure l'environnement AWS et MLflow une seule fois"""
    try:
        # Variables AWS
        os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"] = Variable.get("AWS_DEFAULT_REGION")
        os.environ["ARTIFACT_STORE_URI"] = Variable.get("ARTIFACT_STORE_URI", default_var="")
        
        # MLflow
        mlflow_uri = Variable.get("mlflow_uri", default_var="https://ericjedha-AIA.hf.space/")
        os.environ["APP_URI"] = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Meteo")
        
        print(f"✓ Environment configured successfully")
        print(f"✓ MLflow URI: {mlflow_uri}")
        return True
    except Exception as e:
        print(f"❌ Environment setup failed: {str(e)}")
        raise

# Aller chercher le dernier fichier CSV sur S3

def download_weather_csv_from_s3(**context):
    """Télécharger le fichier CSV depuis S3 et le placer dans DATA_PATH"""
    print("⬇️ Téléchargement du fichier météo depuis S3...")

    setup_environment()
    s3 = boto3.client('s3')

    # 🔹 Récupère la clé du DAG parent si transmise
    dag_conf = context.get('dag_run').conf if context.get('dag_run') else {}
    csv_key = dag_conf.get("csv_key", WEATHER_CSV_FILE)

    # 🔹 Bucket S3
    bucket_name = Variable.get("BUCKET")

    # 🔹 Chemin local de destination
    local_path = os.path.join(DATA_PATH, WEATHER_CSV_FILE)
    os.makedirs(DATA_PATH, exist_ok=True)

    # 🔹 Téléchargement
    try:
        s3.download_file(bucket_name, csv_key, local_path)
        print(f"✅ Fichier téléchargé depuis s3://{bucket_name}/{csv_key}")
        print(f"   → Local: {local_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du téléchargement S3 : {e}")

    # 🔹 Partage du chemin avec les tâches suivantes
    context["ti"].xcom_push(key="local_weather_csv", value=local_path)




def test_aws_connection(**context):
    """Tâche 0: Tester la connexion AWS"""
    print("Testing AWS connection...")
    setup_environment()
    
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"Missing AWS variables: {missing_vars}")
    
    print(f"✓ AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID')[:10]}...")
    
    # Test optionnel boto3
    try:
        import boto3
        boto3.client('s3')
        print("✓ boto3 S3 client created successfully")
    except ImportError:
        print("Note: boto3 not available")
    except Exception as e:
        print(f"Warning: Could not create S3 client: {str(e)}")


task_download_s3_csv = PythonOperator(
    task_id='download_weather_csv_from_s3',
    python_callable=download_weather_csv_from_s3,
    dag=dag,
)


# =============== TÂCHE 1 : Modèle historique (pas de décalage) ===============
def prepare_data_historical(**context):
    """Préparer les données pour le modèle historique (classification instantanée)"""
    print("🔄 Préparation données - modèle historique...")
    
    csv_path = context["ti"].xcom_pull(task_ids="download_weather_csv_from_s3", key="local_weather_csv")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Données vides")
        
    # CORRECTION : Supprimer la colonne 'dew_point' si elle existe
    if 'dew_point' in df.columns:
        df = df.drop('dew_point', axis=1)
        print("✓ Colonne 'dew_point' supprimée pour le modèle historique.")

    df['weather_main'] = df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
    min_samples = 2
    valid_classes = df['weather_main'].value_counts()
    valid_classes = valid_classes[valid_classes >= min_samples].index
    df = df[df['weather_main'].isin(valid_classes)]
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df = df.drop(['datetime', 'weather_description'], axis=1, errors='ignore')
    
    le = LabelEncoder()
    df['weather_main_encoded'] = le.fit_transform(df['weather_main'])
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    df.to_pickle(f"{MODEL_PATH}/data_historical.pkl")
    with open(f"{MODEL_PATH}/label_encoder_historical.pkl", 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Données historiques prêtes : {df.shape}")


def train_historical_model(**context):
    """Entraîner le modèle historique (classification du temps actuel)"""
    print("🧠 Entraînement modèle historique...")
    
    setup_environment()
    
    df = pd.read_pickle(f"{MODEL_PATH}/data_historical.pkl")
    with open(f"{MODEL_PATH}/label_encoder_historical.pkl", 'rb') as f:
        le = pickle.load(f)
    
    # Cette partie fonctionne maintenant car 'dew_point' a déjà été enlevée
    feature_cols = [col for col in df.columns if col not in ['weather_main', 'weather_main_encoded']]
    X = df[feature_cols]
    y = df['weather_main_encoded']
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'accuracy_train': accuracy_score(y_train, y_train_pred),
        'precision_train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall_train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1_train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'accuracy_test': accuracy_score(y_test, y_test_pred),
        'precision_test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'recall_test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'f1_test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
    }
    
    print("=== Modèle historique ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # ✅ Classification report et matrice de confusion
    labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = le.inverse_transform(labels)
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    cr = classification_report(
        y_test, y_test_pred,
        labels=labels,
        target_names=target_names,
     zero_division=0
    )
    
    print("\n=== Matrice de confusion (test) ===")
    print(cm)
    print("\n=== Classification Report (test) ===")
    print(cr)
    
    with open(f"{MODEL_PATH}/xgboost_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # ✅ MLflow complet
    experiment = mlflow.get_experiment_by_name("Meteo")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"Historical_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.log_params({
            'model_type': 'historical',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'n_classes': len(le.classes_),
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate
        })
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # ✅ Log des artefacts textuels
        mlflow.log_text(str(cm), "confusion_matrix_test.txt")
        mlflow.log_text(cr, "classification_report_test.txt")
        
        mlflow.xgboost.log_model(model, "model")
    
    print("✅ Modèle historique loggué avec métriques, CM et rapport complet")


# =============== TÂCHE 2 : Modèle de prévision à 6h ===============
def prepare_data_6h(**context):
    """Préparer les données pour la prévision à 6h"""
    print("🔄 Préparation données - prévision à 6h...")
    csv_path = context["ti"].xcom_pull(task_ids="download_weather_csv_from_s3", key="local_weather_csv")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Fichier CSV introuvable : {csv_path}")
    # Charger les données brutes
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Données vides")
    
    # Nettoyer les classes météo
    df['weather_main'] = df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
    min_samples = 2  # bas seuil pour l'exercice
    valid_classes = df['weather_main'].value_counts()
    valid_classes = valid_classes[valid_classes >= min_samples].index
    df = df[df['weather_main'].isin(valid_classes)]
    
    # Convertir datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Créer la cible : weather_main dans 6 pas de temps
    # ⚠️ Ajuste le décalage selon la fréquence de tes données :
    # - Toutes les 1h → shift(-6)
    # - Toutes les 30min → shift(-12)
    # - Toutes les 10min → shift(-36)
    df['weather_6h'] = df['weather_main'].shift(-6)
    df = df.dropna(subset=['weather_6h']).reset_index(drop=True)
    
    # --- Créer TOUTES les features nécessaires ---
    
    # 1. Timestamp (Unix)
    df['timestamp'] = df['datetime'].apply(lambda x: int(x.timestamp()))
    
    # 2. Features temporelles
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.dayofweek  # propriété pandas → entier
    df['weekday'] = df['weekday'].astype('int64')
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 3. Features trigonométriques
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 4. Vérifier que rain_1h existe (sinon mettre NaN)
    if 'rain_1h' not in df.columns:
        df['rain_1h'] = np.nan
    
    # --- Liste EXACTE des features utilisées (doit matcher generate_6h_forecast) ---
    # CORRECTION : 'dew_point' a été enlevée de cette liste
    feature_cols = [
        'timestamp',
        'clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h',
        'hour', 'month', 'weekday', 'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    # Après avoir créé toutes les features, avant de sauvegarder :
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Vérifier que toutes les features sont présentes
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Colonnes manquantes dans les données : {missing_features}")
    
    # Encoder la cible
    le = LabelEncoder()
    df['weather_6h_encoded'] = le.fit_transform(df['weather_6h'])
    
    # Sauvegarder uniquement les colonnes nécessaires
    save_cols = feature_cols + ['weather_6h', 'weather_6h_encoded']
    df_final = df[save_cols].copy()
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    df_final.to_pickle(f"{MODEL_PATH}/data_6h.pkl")
    with open(f"{MODEL_PATH}/label_encoder_6h.pkl", 'wb') as f:
        pickle.dump(le, f)
    
    print(f"✓ Données 6h prêtes : {df_final.shape} | Features : {len(feature_cols)}")
    print(f"  → Classes : {list(le.classes_)}")


def train_6h_model(**context):
    """Entraîner le modèle de prévision à 6h"""
    print("🧠 Entraînement modèle à 6h...")
    
    setup_environment()
    
    df = pd.read_pickle(f"{MODEL_PATH}/data_6h.pkl")
    with open(f"{MODEL_PATH}/label_encoder_6h.pkl", 'rb') as f:
        le = pickle.load(f)
    
    feature_cols = [col for col in df.columns if col not in ['weather_6h', 'weather_6h_encoded']]
    X = df[feature_cols]
    y = df['weather_6h_encoded']
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'accuracy_train': accuracy_score(y_train, y_train_pred),
        'precision_train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall_train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1_train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'accuracy_test': accuracy_score(y_test, y_test_pred),
        'precision_test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'recall_test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'f1_test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
    }
    
    print("=== Modèle 6h ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # ✅ Classification report et matrice de confusion
    labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = le.inverse_transform(labels)

    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    cr = classification_report(
        y_test, y_test_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    
    print("\n=== Matrice de confusion (test) ===")
    print(cm)
    print("\n=== Classification Report (test) ===")
    print(cr)
    
    with open(f"{MODEL_PATH}/xgboost_model_6h.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # ✅ MLflow complet
    experiment = mlflow.get_experiment_by_name("Meteo")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"Forecast_6h_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.log_params({
            'model_type': 'forecast_6h',
            'horizon': '6h',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'n_classes': len(le.classes_),
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate
        })
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # ✅ Log des artefacts textuels
        mlflow.log_text(str(cm), "confusion_matrix_test.txt")
        mlflow.log_text(cr, "classification_report_test.txt")
        
        mlflow.xgboost.log_model(model, "model")
    
    print("✅ Modèle 6h loggué avec métriques, CM et rapport complet")

def generate_6h_forecast(**context):
    print("🔮 Prévision à 6h...")
    
    df_raw = pd.read_csv(os.path.join(DATA_PATH, WEATHER_CSV_FILE))
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    latest = df_raw.iloc[-1]
    dt = pd.to_datetime(latest['datetime'])
    
    # Colonnes EXACTEMENT comme utilisées pour le modèle 6h
    expected_features = [
        'timestamp',
        'clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h',
        'hour', 'month', 'weekday', 'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    features = {}
    
    # Remplir uniquement les features attendues
    for col in ['clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h']:
        features[col] = latest.get(col, np.nan)
    
    # Timestamp
    features['timestamp'] = int(dt.timestamp())
    
    # Features temporelles
    features['hour'] = dt.hour
    features['month'] = dt.month
    features['weekday'] = dt.dayofweek
    features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
    
    # Features trigonométriques
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Créer DataFrame avec les colonnes correctes
    X_input = pd.DataFrame([{col: features[col] for col in expected_features}])
    
    # Forcer les types numériques
    for col in expected_features:
        X_input[col] = pd.to_numeric(X_input[col], errors='coerce')
    
    # Charger modèle et label encoder
    with open(f"{MODEL_PATH}/xgboost_model_6h.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"{MODEL_PATH}/label_encoder_6h.pkl", 'rb') as f:
        le = pickle.load(f)
    
    # Prédiction
    pred_enc = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    weather_pred = le.inverse_transform([pred_enc])[0]
    confidence = float(np.max(proba))
    
    # Sauvegarder le résultat
    output = {
        'prediction_time': datetime.utcnow().isoformat(),
        'based_on_time': dt.isoformat(),
        'predicted_weather': weather_pred,
        'confidence': confidence,
        'horizon': '6h'
    }
    
    pd.DataFrame([output]).to_csv(os.path.join(DATA_PATH, 'weather_forecast_output.csv'), index=False)
    print(f"✅ Prévision : {weather_pred} ({confidence:.1%})")



# =============== DÉFINITION DES TÂCHES ===============
task_prepare_hist = PythonOperator(
    task_id='prepare_data_historical',
    python_callable=prepare_data_historical,
    dag=dag,
)

task_train_hist = PythonOperator(
    task_id='train_historical_model',
    python_callable=train_historical_model,
    dag=dag,
)

task_prepare_6h = PythonOperator(
    task_id='prepare_data_6h',
    python_callable=prepare_data_6h,
    dag=dag,
)

task_train_6h = PythonOperator(
    task_id='train_6h_model',
    python_callable=train_6h_model,
    dag=dag,
)

task_forecast = PythonOperator(
    task_id='generate_6h_forecast',
    python_callable=generate_6h_forecast,
    dag=dag,
)

from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_real_time_weather = TriggerDagRunOperator(
    task_id="meteo_04_real_time_weather",
    trigger_dag_id="meteo_04_real_time_weather",  # DAG cible
    wait_for_completion=False,  # ne bloque pas l'exécution
    poke_interval=60,  # intervalle de vérification si wait_for_completion=True
    reset_dag_run=True,  # optionnel : recrée un DAG run si déjà existant pour la même execution_date
    dag=dag
)

# =============== ORDONNANCEMENT ===============
task_download_s3_csv >> task_prepare_hist >> task_train_hist
task_download_s3_csv >> task_prepare_6h >> task_train_6h >> task_forecast
task_forecast >> trigger_real_time_weather