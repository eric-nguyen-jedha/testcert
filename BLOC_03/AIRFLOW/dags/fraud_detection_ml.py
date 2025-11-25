# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import mlflow
import mlflow.sklearn
import os
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '/opt/airflow/data'

# =====================================================
# 1. Définitions globales du DAG
# =====================================================
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

dag = DAG(
    "fraud_detection_02_xgboost_dag",
    default_args=default_args,
    description="Pipeline ML pour détection de fraude avec XGBoost et MLflow",
    schedule=None,
    start_date=datetime(2025, 10, 27),
    catchup=False,
    tags=["fraud", "mlflow", "xgboost"]
)

# =====================================================
# 2. Fonctions Python
# =====================================================
def setup_credentials():
    """Charge les credentials AWS et MLflow depuis Airflow Variables"""
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"] = Variable.get("AWS_DEFAULT_REGION", default_var="us-east-1")
        os.environ["ARTIFACT_STORE_URI"] = Variable.get("ARTIFACT_STORE_URI", default_var="")
        os.environ["BACKEND_STORE_URI_FP"] = Variable.get("BACKEND_STORE_URI_FP", default_var="")

        mlflow_uri = Variable.get("mlflow_uri", default_var="https://ericjedha-fraud-detection.hf.space/")
        mlflow.set_tracking_uri(mlflow_uri)

        print("✅ Credentials AWS et MLflow configurés")
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de la configuration des credentials : {e}")

def load_csv(**context):
    setup_credentials()
    csv_filename = Variable.get("LOCAL_FRAUD_CSV", default_var="fraudTest.csv")
    csv_path = os.path.join(DATA_PATH, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Le fichier {csv_path} n'existe pas.")
    
    df = pd.read_csv(csv_path)
    print(f"✅ CSV chargé depuis {csv_path}, shape = {df.shape}")
    context['ti'].xcom_push(key='raw_csv_path', value=csv_path)

def clean_data(**context):
    setup_credentials()
    csv_path = context['ti'].xcom_pull(key='raw_csv_path', task_ids='load_csv')
    df = pd.read_csv(csv_path)

    # Colonnes à supprimer (mais on garde merch_lat, merch_long, trans_num pour l'instant)
    cols_to_drop = [
        "Unnamed: 0", "unix_time", "dob",
        "state", "street", "first", "last", "cc_num"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ===== RENOMMAGE : lat/long -> merch_lat/merch_long =====
    if "lat" in df.columns:
        df = df.rename(columns={"lat": "merch_lat"})
    if "long" in df.columns:
        df = df.rename(columns={"long": "merch_long"})

    # ===== TEMPORAL FEATURES =====
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["year"] = df["trans_date_trans_time"].dt.year
    df["day_of_week"] = df["trans_date_trans_time"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df = df.drop(columns=["trans_date_trans_time"])

    # ===== MERCHANT / CATEGORY FEATURES =====
    df["merchant"] = df["merchant"].str.replace(r"^fraud_", "", regex=True)
    df["amt_log"] = np.log1p(df["amt"])
    
    # Calcul de merchant_freq sur l'ensemble d'entraînement
    freq = df["merchant"].value_counts() / len(df)
    df["merchant_freq"] = df["merchant"].map(freq)
    
    df["amt_category"] = df["amt_log"] * df["category"].astype("category").cat.codes
    df["merchant_city"] = df["merchant"].astype(str) + "_" + df["city"].astype(str)

    clean_csv_path = os.path.join(DATA_PATH, "fraudTest_clean.csv")
    df.to_csv(clean_csv_path, index=False)
    print(f"✅ Données nettoyées sauvegardées en {clean_csv_path}, shape = {df.shape}")
    context['ti'].xcom_push(key='clean_csv_path', value=clean_csv_path)

def train_mlflow(**context):
    setup_credentials()
    clean_csv_path = context['ti'].xcom_pull(key='clean_csv_path', task_ids='clean_data')
    df = pd.read_csv(clean_csv_path)

    # Séparer trans_num et is_fraud AVANT de créer X
    trans_nums = df["trans_num"].copy() if "trans_num" in df.columns else None
    y = df["is_fraud"].astype(int)
    
    # Supprimer trans_num et is_fraud de X
    cols_to_exclude = ["is_fraud", "trans_num"]
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])

    # Définir les features attendues (IMPORTANT: cet ordre sera utilisé en prédiction)
    expected_features = [
        "merchant", "category", "amt", "gender", "city", "zip", 
        "city_pop", "job", "merch_lat", "merch_long",
        "hour", "day", "month", "year", "day_of_week", "is_weekend",
        "amt_log", "merchant_freq", "amt_category", "merchant_city"
    ]
    
    # Vérifier que toutes les features attendues sont présentes
    missing_features = set(expected_features) - set(X.columns)
    if missing_features:
        raise ValueError(f"❌ Features manquantes dans X: {missing_features}")
    
    # Réorganiser X pour avoir le bon ordre
    X = X[expected_features]

    # Colonnes catégorielles et numériques
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    print(f"📊 Features totales: {len(expected_features)}")
    print(f"📊 Catégorielles: {categorical_cols}")
    print(f"📊 Numériques: {numerical_cols}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Calculer merchant_freq sur TRAIN uniquement pour éviter le data leakage
    merchant_freq_train = X_train["merchant"].value_counts() / len(X_train)
    merchant_freq_map = merchant_freq_train.to_dict()
    
    # Appliquer sur train et test
    X_train["merchant_freq"] = X_train["merchant"].map(merchant_freq_map).fillna(0.001)
    X_test["merchant_freq"] = X_test["merchant"].map(merchant_freq_map).fillna(0.001)

    # ===== Pipeline preprocessing + modèle =====
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=sum(y_train==0)/sum(y_train==1),
            random_state=42,
            n_estimators=300
        ))
    ])

    # ===== MLFLOW TRACKING =====
    mlflow.set_experiment("fraud_detection_xgboost")
    
    with mlflow.start_run(run_name="xgboost_fraud_detection") as run:
        # Entraînement
        pipeline.fit(X_train, y_train)

        # Prédictions & métriques
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log des métriques
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_recall_fraud", report["1"]["recall"])
        mlflow.log_metric("test_precision_fraud", report["1"]["precision"])
        mlflow.log_metric("test_f1_fraud", report["1"]["f1-score"])

        # Matrice de confusion
        cm_path = os.path.join(DATA_PATH, "confusion_matrix.png")
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Non-Fraude", "Fraude"],
                    yticklabels=["Non-Fraude", "Fraude"])
        plt.title("Matrice de confusion – XGBoost")
        plt.tight_layout()
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Log des paramètres et metadata
        mlflow.log_param("expected_features", expected_features)
        mlflow.log_param("categorical_cols", categorical_cols)
        mlflow.log_param("numerical_cols", numerical_cols)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        
        # Sauvegarder le mapping merchant_freq comme artifact JSON
        merchant_freq_path = os.path.join(DATA_PATH, "merchant_freq_map.json")
        with open(merchant_freq_path, 'w') as f:
            json.dump(merchant_freq_map, f)
        mlflow.log_artifact(merchant_freq_path)

        # Log du pipeline sklearn complet
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="xgboost_fraud_pipeline",
            registered_model_name="fraud_detector_xgboost"
        )

        run_id = run.info.run_id
        print(f"✅ MLflow Run ID: {run_id}")

        # Enregistrer et créer l'alias @champion
        try:
            from mlflow import MlflowClient
            client = MlflowClient()
            
            # Récupérer la dernière version du modèle
            model_name = "fraud_detector_xgboost_pipeline"
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            
            if latest_versions:
                model_version = latest_versions[0].version
                
                # Définir l'alias @champion (MLflow 2.0+)
                try:
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="champion",
                        version=model_version
                    )
                    print(f"✅ Alias @champion défini sur la version {model_version}")
                except AttributeError:
                    # Fallback pour anciennes versions de MLflow: utiliser les stages
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version,
                        stage="Production"
                    )
                    print(f"✅ Modèle version {model_version} mis en Production (fallback)")
        except Exception as e:
            print(f"⚠️ Impossible de définir l'alias: {e}")

        print(f"✅ Pipeline complet loggé dans MLflow, AUC={auc:.4f}")


# =====================================================
# 3. Définition des tâches du DAG
# =====================================================
load_s3_csv = PythonOperator(
    task_id="load_csv",
    python_callable=load_csv,
    dag=dag
)

data_clean = PythonOperator(
    task_id="clean_data",
    python_callable=clean_data,
    dag=dag
)

train_ml_mlflow = PythonOperator(
    task_id="train_mlflow",
    python_callable=train_mlflow,
    dag=dag
)

load_s3_csv >> data_clean >> train_ml_mlflow