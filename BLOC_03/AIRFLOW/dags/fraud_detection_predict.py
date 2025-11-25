# -*- coding: utf-8 -*- Version 13:41
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.models import Variable
import pandas as pd
import numpy as np
import requests
import json
import os
import mlflow
import mlflow.sklearn
import boto3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import warnings
from sqlalchemy import create_engine, text
import uuid
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
DATA_PATH = "/opt/airflow/data"

# -------------------------
# DAG settings
# -------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "fraud_detection_03_prediction_api",
    default_args=default_args,
    description="Prédiction de fraude (API -> MLflow -> S3 -> Mail)",
    schedule="*/2 * * * *",  # toutes les 2 minutes
    start_date=datetime(2025, 10, 27),
    catchup=False,
    tags=["fraud", "prediction", "mlflow"],
)

# -------------------------
# Utilities / credentials
# -------------------------
def setup_credentials():
    """Configure AWS / MLflow credentials from Airflow Variables."""
    os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = Variable.get("AWS_DEFAULT_REGION", default_var="eu-west-3")
    os.environ["ARTIFACT_STORE_URI"] = Variable.get("ARTIFACT_STORE_URI", default_var="")
    os.environ["BACKEND_STORE_URI_FP"] = Variable.get("BACKEND_STORE_URI_FP", default_var="")
    
    mlflow_uri = Variable.get("mlflow_uri", default_var="https://ericjedha-fraud-detection.hf.space/")
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info("✅ Credentials AWS et MLflow configurés")


def get_neon_engine():
    """Récupère le moteur SQLAlchemy pour Neon DB"""
    database_url = Variable.get("NEON_DATABASE_URL_SECRET")
    engine = create_engine(database_url)
    return engine


def init_neon_table():
    """Crée la table fraud_predictions si elle n'existe pas"""
    engine = get_neon_engine()
    
    create_table_sql = text("""
    CREATE TABLE IF NOT EXISTS fraud_predictions (
        id SERIAL PRIMARY KEY,
        trans_num VARCHAR(255) UNIQUE,
        merchant VARCHAR(255),
        category VARCHAR(100),
        amt NUMERIC(10, 2),
        gender VARCHAR(10),
        city VARCHAR(100),
        zip INTEGER,
        city_pop INTEGER,
        job VARCHAR(255),
        merch_lat NUMERIC(10, 6),
        merch_long NUMERIC(10, 6),
        hour INTEGER,
        day INTEGER,
        month INTEGER,
        year INTEGER,
        day_of_week INTEGER,
        is_weekend INTEGER,
        amt_log NUMERIC(10, 6),
        merchant_freq NUMERIC(10, 6),
        amt_category NUMERIC(10, 6),
        merchant_city VARCHAR(255),
        pred_is_fraud INTEGER,
        is_fraud_ground_truth INTEGER,
        transaction_time BIGINT,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_trans_num ON fraud_predictions(trans_num);
    CREATE INDEX IF NOT EXISTS idx_pred_is_fraud ON fraud_predictions(pred_is_fraud);
    CREATE INDEX IF NOT EXISTS idx_created_at ON fraud_predictions(created_at);
    """)

    with engine.begin() as conn:
        conn.execute(create_table_sql)
    
    logger.info("✅ Table fraud_predictions vérifiée/créée dans Neon DB")


# -------------------------
# 1) Fetch from API
# -------------------------
def fetch_transactions(**context):
    """Récupère les transactions depuis l'API"""
    setup_credentials()
    
    # Générer un execution_id unique pour cette exécution
    execution_id = str(uuid.uuid4())
    context['ti'].xcom_push(key='execution_id', value=execution_id)
    
    url = Variable.get("API_URL", default_var="https://ericjedha-real-time-fraud-detection.hf.space/current-transactions")
    headers = {"User-Agent": "Airflow-Fraud-Detector/1.0"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data_json = resp.json()
    except Exception as e:
        logger.error(f"❌ Erreur API: {e}")
        raise

    if isinstance(data_json, str):
        data_json = json.loads(data_json)

    if not isinstance(data_json, dict) or "columns" not in data_json or "data" not in data_json:
        raise ValueError("Format inattendu depuis l'API — attendu dict avec 'columns' et 'data'")

    df = pd.DataFrame(
        data=data_json["data"], 
        columns=data_json["columns"], 
        index=data_json.get("index", None)
    )

    os.makedirs(DATA_PATH, exist_ok=True)
    raw_file_path = os.path.join(DATA_PATH, f"raw_{execution_id}.csv")
    df.to_csv(raw_file_path, index=False)
    
    context['ti'].xcom_push(key='raw_file_path', value=raw_file_path)
    logger.info(f"✅ {len(df)} transaction(s) récupérée(s) depuis l'API → {raw_file_path}")


# -------------------------
# 2) Preprocessing - FIX : Ne pas perdre les colonnes originales
# -------------------------
def preprocess_data(**context):
    """Prétraite les données pour qu'elles correspondent exactement aux features d'entraînement"""
    setup_credentials()
    
    raw_file_path = context['ti'].xcom_pull(key='raw_file_path', task_ids='fetch_transactions')
    execution_id = context['ti'].xcom_pull(key='execution_id', task_ids='fetch_transactions')
    
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Fichier brut manquant : {raw_file_path}")
    
    df = pd.read_csv(raw_file_path)
    
    # 🔧 FIX : Sauvegarder TOUTES les colonnes originales avant transformation
    original_df = df.copy()
    
    # Sauvegarder trans_num et ground truth pour plus tard
    trans_nums = df["trans_num"].copy() if "trans_num" in df.columns else pd.Series([None]*len(df))
    current_times = df["current_time"].copy() if "current_time" in df.columns else pd.Series([None]*len(df))
    ground_truth = df["is_fraud"].copy() if "is_fraud" in df.columns else pd.Series([None]*len(df))

    # ===== RENOMMAGE : lat/long -> merch_lat/merch_long =====
    if "lat" in df.columns:
        df = df.rename(columns={"lat": "merch_lat"})
    if "long" in df.columns:
        df = df.rename(columns={"long": "merch_long"})

    # ===== TEMPORAL FEATURES =====
    if "current_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["current_time"], unit="ms")
    elif "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    else:
        raise KeyError("Aucun champ timestamp trouvé")

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["year"] = df["trans_date_trans_time"].dt.year
    df["day_of_week"] = df["trans_date_trans_time"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df = df.drop(columns=["trans_date_trans_time"], errors='ignore')

    # ===== MERCHANT FEATURES =====
    if "merchant" in df.columns:
        df["merchant"] = df["merchant"].astype(str).str.replace(r"^fraud_", "", regex=True)
    
    df["amt_log"] = np.log1p(df["amt"].astype(float))

    # ===== CHARGER MERCHANT_FREQ DEPUIS MLFLOW =====
    try:
        MODEL_URI = Variable.get("MLFLOW_MODEL_URI", default_var="models:/fraud_detector_xgboost_@champion")
        
        try:
            model_info = mlflow.sklearn.load_model(MODEL_URI)
        except:
            logger.warning("⚠️ Alias @champion non trouvé, essai avec Production stage...")
            MODEL_URI = "models:/fraud_detector_xgboost_pipeline/Production"
            model_info = mlflow.sklearn.load_model(MODEL_URI)
        
        from mlflow import MlflowClient
        client = MlflowClient()
        
        if "@champion" in MODEL_URI:
            model_name = MODEL_URI.split("/")[-1].split("@")[0]
            alias = "champion"
            model_version = client.get_model_version_by_alias(model_name, alias).version
        elif "/Production" in MODEL_URI:
            model_name = MODEL_URI.split("/")[-2]
            versions = client.get_latest_versions(model_name, stages=["Production"])
            model_version = versions[0].version if versions else None
        else:
            raise ValueError(f"Format MODEL_URI non reconnu: {MODEL_URI}")
        
        model_details = client.get_model_version(model_name, model_version)
        run_id = model_details.run_id
        
        artifact_path = client.download_artifacts(run_id, "merchant_freq_map.json")
        
        with open(artifact_path, 'r') as f:
            merchant_freq_map = json.load(f)
        
        df["merchant_freq"] = df["merchant"].map(merchant_freq_map).fillna(0.001)
        logger.info("✅ merchant_freq chargé depuis MLflow")
        
    except Exception as e:
        logger.warning(f"⚠️ Impossible de charger merchant_freq depuis MLflow: {e}")
        logger.warning("⚠️ Utilisation d'une valeur par défaut (0.001)")
        df["merchant_freq"] = 0.001

    # ===== AUTRES FEATURES =====
    if "category" in df.columns:
        df["amt_category"] = df["amt_log"] * df["category"].astype("category").cat.codes
    else:
        df["amt_category"] = 0.0
    
    df["merchant_city"] = df.apply(
        lambda r: f"{r.get('merchant','')}_{r.get('city','')}", axis=1
    )

    # Supprimer les colonnes non nécessaires
    drop_cols = ["Unnamed: 0", "unix_time", "dob", "cc_num", "first", "last", 
                 "street", "state", "current_time", "trans_num", "is_fraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Sauvegarder le fichier clean
    clean_file_path = os.path.join(DATA_PATH, f"clean_{execution_id}.csv")
    df.to_csv(clean_file_path, index=False)
    
    # 🔧 FIX : Sauvegarder les colonnes originales complètes pour reporting
    # On garde merchant, category, city, gender depuis le dataframe ORIGINAL
    reporting_columns = ['trans_num', 'merchant', 'category', 'city', 'gender', 'current_time', 'is_fraud']
    reporting_data = pd.DataFrame()
    
    for col in reporting_columns:
        if col in original_df.columns:
            reporting_data[col] = original_df[col]
        else:
            # Fallback sur les colonnes calculées
            if col == 'is_fraud':
                reporting_data['is_fraud_ground_truth'] = ground_truth
            elif col == 'trans_num':
                reporting_data[col] = trans_nums
            elif col == 'current_time':
                reporting_data[col] = current_times
            else:
                reporting_data[col] = None
    
    # Renommer is_fraud en is_fraud_ground_truth
    if 'is_fraud' in reporting_data.columns:
        reporting_data = reporting_data.rename(columns={'is_fraud': 'is_fraud_ground_truth'})
    
    reporting_file_path = os.path.join(DATA_PATH, f"reporting_{execution_id}.csv")
    reporting_data.to_csv(reporting_file_path, index=False)
    
    context['ti'].xcom_push(key='clean_file_path', value=clean_file_path)
    context['ti'].xcom_push(key='reporting_file_path', value=reporting_file_path)
    
    logger.info(f"✅ Prétraitement terminé → {clean_file_path} (shape={df.shape})")
    logger.info(f"✅ Reporting data sauvegardées → {reporting_file_path}")


# -------------------------
# 3) Predict and Save - VERSION FINALE CORRIGÉE
# -------------------------
def predict_and_save(**context):
    """Fait les prédictions, sauvegarde localement et insère dans Neon DB."""
    setup_credentials()

    clean_file_path = context['ti'].xcom_pull(key='clean_file_path', task_ids='preprocess_data')
    reporting_file_path = context['ti'].xcom_pull(key='reporting_file_path', task_ids='preprocess_data')
    execution_id = context['ti'].xcom_pull(key='execution_id', task_ids='fetch_transactions')

    df = pd.read_csv(clean_file_path)
    reporting_data = pd.read_csv(reporting_file_path)

    MODEL_URI = Variable.get("MLFLOW_MODEL_URI", default_var="models:/fraud_detector_xgboost_pipeline@champion")

    try:
        pipeline = mlflow.sklearn.load_model(MODEL_URI)
        logger.info(f"✅ Modèle chargé depuis {MODEL_URI}")
    except Exception as e:
        logger.warning(f"⚠️ Erreur de chargement modèle @champion : {e} — fallback /Production")
        pipeline = mlflow.sklearn.load_model("models:/fraud_detector_xgboost_pipeline/Production")

    expected_features = [
        "merchant", "category", "amt", "gender", "city", "zip",
        "city_pop", "job", "merch_lat", "merch_long",
        "hour", "day", "month", "year", "day_of_week", "is_weekend",
        "amt_log", "merchant_freq", "amt_category", "merchant_city"
    ]

    df_for_model = df[expected_features].copy()
    preds = pipeline.predict(df_for_model)

    df_result = df_for_model.copy()
    df_result["pred_is_fraud"] = preds.astype(int)

    # 🔧 FIX : Reset des index pour garantir l'alignement
    df_result = df_result.reset_index(drop=True)
    reporting_data = reporting_data.reset_index(drop=True)

    # 🔧 FIX : Nettoyer fraud_ dans reporting_data AVANT la fusion
    if 'merchant' in reporting_data.columns:
        reporting_data['merchant'] = reporting_data['merchant'].astype(str).str.replace(r"^fraud_", "", regex=True)

    # Joindre les métadonnées - SEULEMENT les colonnes qui ne sont PAS dans df_result
    # On garde merchant, category, city, gender du MODÈLE (déjà nettoyés)
    # On ajoute uniquement trans_num, current_time, is_fraud_ground_truth
    metadata_cols = ['trans_num', 'current_time', 'is_fraud_ground_truth']
    for col in metadata_cols:
        if col in reporting_data.columns:
            df_result[col] = reporting_data[col]

    # 🔍 DEBUG : Vérifier APRÈS la fusion
    problematic_rows = df_result[
        (df_result['merchant'].astype(str) == '0') | 
        (df_result['category'].astype(str) == '0') |
        (df_result['merchant'].astype(str).str.startswith('fraud_'))
    ]

    if len(problematic_rows) > 0:
        logger.error(f"🚨 {len(problematic_rows)} lignes avec des problèmes APRÈS fusion!")
        for idx, row in problematic_rows.head(3).iterrows():
            logger.error(f"   trans_num: {row.get('trans_num')}")
            logger.error(f"   merchant: '{row.get('merchant')}' (type: {type(row.get('merchant'))})")
            logger.error(f"   category: '{row.get('category')}' (type: {type(row.get('category'))})")

    pred_file_path = os.path.join(DATA_PATH, f"pred_{execution_id}.csv")
    df_result.to_csv(pred_file_path, index=False)
    logger.info(f"✅ Fichier de prédictions sauvegardé : {pred_file_path}")

    # ===== Insertion Neon DB =====
    logger.info("🔄 Insertion dans Neon DB...")
    try:
        init_neon_table()
        engine = get_neon_engine()
        inserted_count = 0

        text_columns = ['trans_num', 'merchant', 'category', 'gender', 'city', 'job', 'merchant_city']

        with engine.begin() as conn:
            for idx, row in df_result.iterrows():
                row_dict = row.to_dict()

                # Conversion numpy → Python
                for k, v in row_dict.items():
                    if isinstance(v, np.generic):
                        row_dict[k] = v.item()
                
                # FORCER les colonnes textuelles en str()
                for col in text_columns:
                    if col in row_dict:
                        val = row_dict[col]
                        if pd.isna(val) or val is None:
                            row_dict[col] = None
                        else:
                            row_dict[col] = str(val)
                
                # Forcer les numériques
                numeric_int_cols = ['zip', 'city_pop', 'hour', 'day', 'month', 'year', 'day_of_week', 'is_weekend', 'pred_is_fraud', 'is_fraud_ground_truth']
                for col in numeric_int_cols:
                    if col in row_dict:
                        val = row_dict[col]
                        if pd.isna(val) or val is None:
                            row_dict[col] = None
                        else:
                            try:
                                row_dict[col] = int(val)
                            except (ValueError, TypeError):
                                row_dict[col] = None
                
                numeric_float_cols = ['amt', 'merch_lat', 'merch_long', 'amt_log', 'merchant_freq', 'amt_category']
                for col in numeric_float_cols:
                    if col in row_dict:
                        val = row_dict[col]
                        if pd.isna(val) or val is None:
                            row_dict[col] = None
                        else:
                            try:
                                row_dict[col] = float(val)
                            except (ValueError, TypeError):
                                row_dict[col] = None

                # Conversion timestamp
                if "current_time" in row_dict:
                    ct_val = row_dict.pop("current_time")
                    if pd.isna(ct_val) or ct_val is None:
                        row_dict["transaction_time"] = None
                    else:
                        try:
                            row_dict["transaction_time"] = int(ct_val)
                        except (ValueError, TypeError):
                            row_dict["transaction_time"] = None
                elif "transaction_time" in row_dict:
                    tt_val = row_dict["transaction_time"]
                    if pd.isna(tt_val) or tt_val is None:
                        row_dict["transaction_time"] = None
                    else:
                        try:
                            row_dict["transaction_time"] = int(tt_val)
                        except (ValueError, TypeError):
                            row_dict["transaction_time"] = None

                # DEBUG les 3 premières lignes
                if idx < 3:
                    logger.info(f"🔍 DEBUG Ligne {idx} - AVANT insertion SQL:")
                    logger.info(f"   trans_num: {row_dict.get('trans_num')} (type: {type(row_dict.get('trans_num'))})")
                    logger.info(f"   merchant: {row_dict.get('merchant')} (type: {type(row_dict.get('merchant'))})")
                    logger.info(f"   category: {row_dict.get('category')} (type: {type(row_dict.get('category'))})")

                insert_sql = text("""
                    INSERT INTO fraud_predictions (
                        trans_num, merchant, category, amt, gender, city, zip, city_pop, job,
                        merch_lat, merch_long, hour, day, month, year, day_of_week, is_weekend,
                        amt_log, merchant_freq, amt_category, merchant_city,
                        pred_is_fraud, is_fraud_ground_truth, transaction_time
                    ) VALUES (
                        :trans_num, :merchant, :category, :amt, :gender, :city, :zip, :city_pop, :job,
                        :merch_lat, :merch_long, :hour, :day, :month, :year, :day_of_week, :is_weekend,
                        :amt_log, :merchant_freq, :amt_category, :merchant_city,
                        :pred_is_fraud, :is_fraud_ground_truth, :transaction_time
                    )
                    ON CONFLICT (trans_num) DO UPDATE SET
                        pred_is_fraud = EXCLUDED.pred_is_fraud,
                        merchant = EXCLUDED.merchant,
                        category = EXCLUDED.category,
                        city = EXCLUDED.city,
                        gender = EXCLUDED.gender,
                        amt = EXCLUDED.amt,
                        created_at = NOW();
                """)

                try:
                    conn.execute(insert_sql, row_dict)
                    inserted_count += 1
                except Exception as err:
                    logger.warning(f"⚠️ Erreur lors de l'insertion (ligne {idx}) : {err}")
                    logger.warning(f"   trans_num: {row_dict.get('trans_num')}")
                    logger.warning(f"   merchant: {row_dict.get('merchant')}")

        logger.info(f"✅ {inserted_count}/{len(df_result)} transactions insérées/actualisées dans Neon DB")

    except Exception as e:
        logger.error(f"❌ Échec de l'insertion Neon DB : {e}", exc_info=True)

    context['ti'].xcom_push(key='pred_file_path', value=pred_file_path)

# -------------------------
# 3) Predict and Save - VERSION FINALE CORRIGÉE
# -------------------------
def predict_and_save(**context):
    """Fait les prédictions, sauvegarde localement et insère dans Neon DB."""
    setup_credentials()

    clean_file_path = context['ti'].xcom_pull(key='clean_file_path', task_ids='preprocess_data')
    reporting_file_path = context['ti'].xcom_pull(key='reporting_file_path', task_ids='preprocess_data')
    execution_id = context['ti'].xcom_pull(key='execution_id', task_ids='fetch_transactions')

    df = pd.read_csv(clean_file_path)
    reporting_data = pd.read_csv(reporting_file_path)

    MODEL_URI = Variable.get("MLFLOW_MODEL_URI", default_var="models:/fraud_detector_xgboost_pipeline@champion")

    try:
        pipeline = mlflow.sklearn.load_model(MODEL_URI)
        logger.info(f"✅ Modèle chargé depuis {MODEL_URI}")
    except Exception as e:
        logger.warning(f"⚠️ Erreur de chargement modèle @champion : {e} — fallback /Production")
        pipeline = mlflow.sklearn.load_model("models:/fraud_detector_xgboost_pipeline/Production")

    expected_features = [
        "merchant", "category", "amt", "gender", "city", "zip",
        "city_pop", "job", "merch_lat", "merch_long",
        "hour", "day", "month", "year", "day_of_week", "is_weekend",
        "amt_log", "merchant_freq", "amt_category", "merchant_city"
    ]

    df_for_model = df[expected_features].copy()
    preds = pipeline.predict(df_for_model)

    df_result = df_for_model.copy()
    df_result["pred_is_fraud"] = preds.astype(int)

    # 🔧 FIX : Reset des index pour garantir l'alignement
    df_result = df_result.reset_index(drop=True)
    reporting_data = reporting_data.reset_index(drop=True)

    # 🔧 FIX : Nettoyer fraud_ dans reporting_data AVANT la fusion
    if 'merchant' in reporting_data.columns:
        reporting_data['merchant'] = reporting_data['merchant'].astype(str).str.replace(r"^fraud_", "", regex=True)

    # Joindre les métadonnées - SEULEMENT les colonnes qui ne sont PAS dans df_result
    # On garde merchant, category, city, gender du MODÈLE (déjà nettoyés)
    # On ajoute uniquement trans_num, current_time, is_fraud_ground_truth
    metadata_cols = ['trans_num', 'current_time', 'is_fraud_ground_truth']
    for col in metadata_cols:
        if col in reporting_data.columns:
            df_result[col] = reporting_data[col]

    # 🔍 DEBUG : Vérifier APRÈS la fusion
    problematic_rows = df_result[
        (df_result['merchant'].astype(str) == '0') | 
        (df_result['category'].astype(str) == '0') |
        (df_result['merchant'].astype(str).str.startswith('fraud_'))
    ]

    if len(problematic_rows) > 0:
        logger.error(f"🚨 {len(problematic_rows)} lignes avec des problèmes APRÈS fusion!")
        for idx, row in problematic_rows.head(3).iterrows():
            logger.error(f"   trans_num: {row.get('trans_num')}")
            logger.error(f"   merchant: '{row.get('merchant')}' (type: {type(row.get('merchant'))})")
            logger.error(f"   category: '{row.get('category')}' (type: {type(row.get('category'))})")

    pred_file_path = os.path.join(DATA_PATH, f"pred_{execution_id}.csv")
    df_result.to_csv(pred_file_path, index=False)
    logger.info(f"✅ Fichier de prédictions sauvegardé : {pred_file_path}")

    # ===== Insertion Neon DB =====
    logger.info("🔄 Insertion dans Neon DB...")
    try:
        init_neon_table()
        engine = get_neon_engine()
        inserted_count = 0

        text_columns = ['trans_num', 'merchant', 'category', 'gender', 'city', 'job', 'merchant_city']

        with engine.begin() as conn:
            for idx, row in df_result.iterrows():
                row_dict = row.to_dict()

                # Conversion numpy → Python
                for k, v in row_dict.items():
                    if isinstance(v, np.generic):
                        row_dict[k] = v.item()
                
                # FORCER les colonnes textuelles en str()
                for col in text_columns:
                    if col in row_dict:
                        val = row_dict[col]
                        if pd.isna(val) or val is None:
                            row_dict[col] = None
                        else:
                            row_dict[col] = str(val)
                
                # Forcer les numériques
                numeric_int_cols = ['zip', 'city_pop', 'hour', 'day', 'month', 'year', 'day_of_week', 'is_weekend', 'pred_is_fraud', 'is_fraud_ground_truth']
                for col in numeric_int_cols:
                    if col in row_dict:
                        val = row_dict[col]
                        if pd.isna(val) or val is None:
                            row_dict[col] = None
                        else:
                            try:
                                row_dict[col] = int(val)
                            except (ValueError, TypeError):
                                row_dict[col] = None
                
                numeric_float_cols = ['amt', 'merch_lat', 'merch_long', 'amt_log', 'merchant_freq', 'amt_category']
                for col in numeric_float_cols:
                    if col in row_dict:
                        val = row_dict[col]
                        if pd.isna(val) or val is None:
                            row_dict[col] = None
                        else:
                            try:
                                row_dict[col] = float(val)
                            except (ValueError, TypeError):
                                row_dict[col] = None

                # Conversion timestamp
                if "current_time" in row_dict:
                    ct_val = row_dict.pop("current_time")
                    if pd.isna(ct_val) or ct_val is None:
                        row_dict["transaction_time"] = None
                    else:
                        try:
                            row_dict["transaction_time"] = int(ct_val)
                        except (ValueError, TypeError):
                            row_dict["transaction_time"] = None
                elif "transaction_time" in row_dict:
                    tt_val = row_dict["transaction_time"]
                    if pd.isna(tt_val) or tt_val is None:
                        row_dict["transaction_time"] = None
                    else:
                        try:
                            row_dict["transaction_time"] = int(tt_val)
                        except (ValueError, TypeError):
                            row_dict["transaction_time"] = None

                # DEBUG les 3 premières lignes
                if idx < 3:
                    logger.info(f"🔍 DEBUG Ligne {idx} - AVANT insertion SQL:")
                    logger.info(f"   trans_num: {row_dict.get('trans_num')} (type: {type(row_dict.get('trans_num'))})")
                    logger.info(f"   merchant: {row_dict.get('merchant')} (type: {type(row_dict.get('merchant'))})")
                    logger.info(f"   category: {row_dict.get('category')} (type: {type(row_dict.get('category'))})")

                insert_sql = text("""
                    INSERT INTO fraud_predictions (
                        trans_num, merchant, category, amt, gender, city, zip, city_pop, job,
                        merch_lat, merch_long, hour, day, month, year, day_of_week, is_weekend,
                        amt_log, merchant_freq, amt_category, merchant_city,
                        pred_is_fraud, is_fraud_ground_truth, transaction_time
                    ) VALUES (
                        :trans_num, :merchant, :category, :amt, :gender, :city, :zip, :city_pop, :job,
                        :merch_lat, :merch_long, :hour, :day, :month, :year, :day_of_week, :is_weekend,
                        :amt_log, :merchant_freq, :amt_category, :merchant_city,
                        :pred_is_fraud, :is_fraud_ground_truth, :transaction_time
                    )
                    ON CONFLICT (trans_num) DO UPDATE SET
                        pred_is_fraud = EXCLUDED.pred_is_fraud,
                        merchant = EXCLUDED.merchant,
                        category = EXCLUDED.category,
                        city = EXCLUDED.city,
                        gender = EXCLUDED.gender,
                        amt = EXCLUDED.amt,
                        created_at = NOW();
                """)

                try:
                    conn.execute(insert_sql, row_dict)
                    inserted_count += 1
                except Exception as err:
                    logger.warning(f"⚠️ Erreur lors de l'insertion (ligne {idx}) : {err}")
                    logger.warning(f"   trans_num: {row_dict.get('trans_num')}")
                    logger.warning(f"   merchant: {row_dict.get('merchant')}")

        logger.info(f"✅ {inserted_count}/{len(df_result)} transactions insérées/actualisées dans Neon DB")

    except Exception as e:
        logger.error(f"❌ Échec de l'insertion Neon DB : {e}", exc_info=True)

    context['ti'].xcom_push(key='pred_file_path', value=pred_file_path)


# -------------------------
# 4) Upload S3 + send emails
# -------------------------
def upload_and_alert(**context):
    """Upload les prédictions sur S3 (fichier incrémental) et envoie un email d'alerte"""
    setup_credentials()
    
    pred_file_path = context['ti'].xcom_pull(key='pred_file_path', task_ids='predict_and_save')
    
    if not os.path.exists(pred_file_path):
        raise FileNotFoundError(f"Fichier de prédictions manquant : {pred_file_path}")

    bucket = Variable.get("BUCKET")
    s3 = boto3.client("s3")
    
    # ===== FICHIER INCRÉMENTAL SUR S3 =====
    s3_master_key = "predictions/all_predictions.csv"
    local_master_path = "/tmp/all_predictions.csv"
    
    # Charger les nouvelles prédictions
    df_new = pd.read_csv(pred_file_path)
    
    # Télécharger le fichier master existant (s'il existe)
    try:
        s3.download_file(bucket, s3_master_key, local_master_path)
        df_existing = pd.read_csv(local_master_path)
        logger.info(f"✅ Fichier master existant téléchargé : {len(df_existing)} lignes")
        
        # Concaténer avec les nouvelles prédictions
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Dédupliquer par trans_num (garder la plus récente)
        df_combined = df_combined.drop_duplicates(subset=['trans_num'], keep='last')
        
        logger.info(f"📊 Fusion : {len(df_existing)} anciennes + {len(df_new)} nouvelles = {len(df_combined)} total (après dédupli)")
    
    except s3.exceptions.NoSuchKey:
        logger.warning(f"⚠️ Fichier master non trouvé sur S3, création d'un nouveau fichier")
        df_combined = df_new
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du téléchargement du fichier master : {e}, création d'un nouveau fichier")
        df_combined = df_new
    
    # Sauvegarder le fichier combiné localement
    df_combined.to_csv(local_master_path, index=False)
    
    # Uploader sur S3 (écrase l'ancien fichier master)
    s3.upload_file(local_master_path, bucket, s3_master_key)
    s3_url = f"s3://{bucket}/{s3_master_key}"
    logger.info(f"✅ Fichier master mis à jour sur S3: {s3_url}")
    
    # ===== ARCHIVER AUSSI CETTE EXÉCUTION (OPTIONNEL) =====
    # Si tu veux garder une trace de chaque exécution
    #s3_archive_key = f"predictions/archive/predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    #s3.upload_file(pred_file_path, bucket, s3_archive_key)
    #logger.info(f"✅ Archive créée : s3://{bucket}/{s3_archive_key}")

    # ===== GÉNÉRER L'EMAIL (basé sur les nouvelles transactions uniquement) =====
    df = df_new  # On travaille sur les nouvelles transactions pour l'email
    df_fraud = df[df["pred_is_fraud"] == 1]

    # Configuration SMTP
    conn = BaseHook.get_connection("smtp_default")
    smtp_host = conn.extra_dejson.get("smtp_host", "smtp.gmail.com")
    smtp_port = conn.extra_dejson.get("smtp_port", 587)
    use_tls = conn.extra_dejson.get("smtp_starttls", True)

    # Générer l'email
    if not df_fraud.empty:
        html_rows = ""
        for _, r in df_fraud.iterrows():
            trans = r.get("trans_num", "N/A")
            merchant = r.get("merchant", "N/A")
            category = r.get("category", "N/A")
            amt = r.get("amt", 0.0)
            city = r.get("city", "N/A")
            html_rows += f"<tr><td><strong>{trans}</strong></td><td>{merchant}</td><td>{category}</td><td>{amt:.2f}</td><td>{city}</td><td>❌ FRAUDE</td></tr>"

        html_content = f"""
        <h2>⚠️ ALERTE FRAUDE DÉTECTÉE</h2>
        <p>{len(df_fraud)} transaction(s) suspecte(s) détectée(s) dans cette exécution.</p>
        <table border="1" style="border-collapse: collapse; width:100%;">
            <tr>
                <th>ID Transaction</th><th>Marchand</th><th>Catégorie</th>
                <th>Montant</th><th>Ville</th><th>Statut</th>
            </tr>
            {html_rows}
        </table>
        <p>📂 <strong>Fichier complet (toutes les prédictions) :</strong> <a href="https://s3.console.aws.amazon.com/s3/object/{bucket}?prefix={s3_master_key}">Télécharger all_predictions.csv</a></p>
        <p>📊 Dashboard de suivi : <a href="https://huggingface.co/spaces/ericjedha/fraud-detection-streamlit">Voir Dashboard</a></p>
        <p style="font-size: 0.9em; color: #666;">💡 Le fichier master contient {len(df_combined)} transactions au total.</p>
        """
        subject = "- 𖣘 - Airflow Fraud Detection : ⚠️ Alerte Fraude détectée !"
    else:
        first = df.iloc[0] if len(df) > 0 else {}
        trans = first.get("trans_num", "N/A")
        merchant = first.get("merchant", "N/A")
        category = first.get("category", "N/A")
        amt = first.get("amt", 0.0)
        city = first.get("city", "N/A")

        html_content = f"""
        <h2>✅ Transaction Normale</h2>
        <p>Aucune fraude détectée sur {len(df)} transaction(s) analysée(s) dans cette exécution.</p>
        <table border="1" style="border-collapse: collapse; width:100%;">
            <tr>
                <th>ID Transaction</th><th>Marchand</th><th>Catégorie</th><th>Montant</th><th>Ville</th><th>Statut</th>
            </tr>
            <tr>
                <td><strong>{trans}</strong></td><td>{merchant}</td><td>{category}</td><td>{amt:.2f}</td><td>{city}</td><td style="color:green;">✅ LÉGITIME</td>
            </tr>
        </table>
        <p>📂 <strong>Fichier complet (toutes les prédictions) :</strong> <a href="https://s3.console.aws.amazon.com/s3/object/{bucket}?prefix={s3_master_key}">Télécharger all_predictions.csv</a></p>
        <p>📊 Dashboard de suivi : <a href="https://huggingface.co/spaces/ericjedha/fraud-detection-streamlit">Voir Dashboard</a></p>
        <p style="font-size: 0.9em; color: #666;">💡 Le fichier master contient {len(df_combined)} transactions au total.</p>
        """
        subject = "- 𖣘 - Airflow Fraud Detection : 🧘🏻‍♀️ Rien à Signaler"

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = f"Fraud Detection <{conn.login}>"
    msg["To"] = Variable.get("ALERT_EMAIL", default_var="en9.eric@gmail.com")
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        if use_tls:
            server.starttls()
        server.login(conn.login, conn.password)
        server.send_message(msg)

    logger.info(f"✅ Email envoyé: {subject}")


# -------------------------
# Tasks
# -------------------------
fetch_api = PythonOperator(
    task_id="fetch_transactions", 
    python_callable=fetch_transactions, 
    dag=dag
)

clean_data = PythonOperator(
    task_id="preprocess_data", 
    python_callable=preprocess_data, 
    dag=dag
)

predict_save = PythonOperator(
    task_id="predict_and_save", 
    python_callable=predict_and_save, 
    dag=dag
)

upload_notify = PythonOperator(
    task_id="upload_and_alert", 
    python_callable=upload_and_alert, 
    dag=dag
)

fetch_api >> clean_data >> predict_save >> upload_notify