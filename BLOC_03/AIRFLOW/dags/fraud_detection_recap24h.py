from airflow import DAG
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os

logger = logging.getLogger(__name__)

# ==============================
# 📅 DAG : Exécution à 9h chaque matin
# ==============================
default_args = {
    "owner": "fraud_detection",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10)
}

with DAG(
    dag_id="fraud_detection_04_recap_email",
    start_date=datetime(2024, 1, 1),
    schedule="0 9 * * *",  # Tous les jours à 09h00
    catchup=False,
    default_args=default_args,
    tags=["fraud", "email", "dashboard"]
) as dag:

    def send_fraud_recap_email(**context):
        """Se connecte à la DB, calcule les stats des 24 dernières heures et envoie un email résumé"""

        # ---------------------
        # Connexion base Neon
        # ---------------------
        neon_url = Variable.get("NEON_DATABASE_URL_SECRET")
        engine = create_engine(neon_url)

        query = text("""
            SELECT pred_is_fraud, COUNT(*) as n
            FROM fraud_predictions
            WHERE created_at >= NOW() - INTERVAL '24 HOURS'
            GROUP BY pred_is_fraud
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        frauds = int(df.loc[df["pred_is_fraud"] == 1, "n"].sum()) if 1 in df["pred_is_fraud"].values else 0
        no_frauds = int(df.loc[df["pred_is_fraud"] == 0, "n"].sum()) if 0 in df["pred_is_fraud"].values else 0

        logger.info(f"Fraudes détectées (24h) : {frauds}")
        logger.info(f"Transactions légitimes : {no_frauds}")

        # ---------------------
        # Préparation email
        # ---------------------
        conn = BaseHook.get_connection("smtp_default")
        smtp_host = conn.extra_dejson.get("smtp_host", "smtp.gmail.com")
        smtp_port = conn.extra_dejson.get("smtp_port", 587)
        use_tls = conn.extra_dejson.get("smtp_starttls", True)

        dashboard_home = "https://huggingface.co/spaces/ericjedha/fraud-detection-streamlit"
        link_fraud = f"{dashboard_home}?page=fraudes"
        link_no_fraud = f"{dashboard_home}?page=non-fraudes"

        html_content = f"""
        <h2>🕵🏻 Rapport Quotidien — Détection de Fraudes</h2>
        <p>Voici le résumé des 24 dernières heures :</p>

        <ul>
            <li>🚨 <strong>Fraudes détectées :</strong> {frauds}</li>
            <li>✅ <strong>Transactions légitimes :</strong> {no_frauds}</li>
        </ul>

        <p>📊 Dashboard complet : 
            <a href="{dashboard_home}">{dashboard_home}</a>
        </p>

        <hr>
        <p style="font-size:0.9em;color:#888;">Email automatique généré par Airflow — DAG : daily_fraud_recap_email</p>
        """

        subject = f"- 𖣘 - Rapport Fraud Detection — {datetime.now().strftime('%d %b %Y')}"
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"Fraud Detection <{conn.login}>"
        msg["To"] = Variable.get("ALERT_EMAIL", default_var="en9.eric@gmail.com")
        msg.attach(MIMEText(html_content, "html"))

        # ---------------------
        # Envoi email
        # ---------------------
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if use_tls:
                server.starttls()
            server.login(conn.login, conn.password)
            server.send_message(msg)

        logger.info(f"✅ Email envoyé : {subject}")

    send_email = PythonOperator(
    task_id="send_fraud_recap_email",
    python_callable=send_fraud_recap_email,
    dag=dag,
)

