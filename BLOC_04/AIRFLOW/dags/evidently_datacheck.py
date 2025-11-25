# -*- coding: utf-8 -*-
# DAG Evidently - Rapport Drift visuel + Test Suite en format textuel HTML

from datetime import datetime, timedelta
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import BaseOperator
from airflow.hooks.base import BaseHook
from airflow.models import Variable
import boto3
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

# ========================== IMPORTS EVIDENTLY ==========================
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import (TestNumberOfColumns, TestNumberOfRows, 
                            TestNumberOfMissingValues, TestShareOfMissingValues)

logger = logging.getLogger(__name__)

# ========================== CONFIGURATION ==========================
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

DATA_PATH = '/opt/airflow/data'
REPORTS_PATH = '/opt/airflow/reports'
WEATHER_CSV_FILE = 'weather_paris.csv'

# ========================== GÉNÉRATION RAPPORT TEXTUEL ==========================

def generate_test_suite_html(test_suite, output_path):
    """
    Génère un rapport HTML purement textuel de la Test Suite
    sans aucun graphique - juste des tableaux et du texte.
    """
    test_results = test_suite.as_dict()
    tests = test_results.get('tests', [])
    
    # Comptage pour le résumé
    tests_summary = {'passed': 0, 'failed': 0, 'warning': 0, 'error': 0}
    for test in tests:
        status = test.get('status', 'unknown').lower()
        if status == 'success':
            tests_summary['passed'] += 1
        elif status == 'fail':
            tests_summary['failed'] += 1
        elif status == 'warning':
            tests_summary['warning'] += 1
        elif status == 'error':
            tests_summary['error'] += 1

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Evidently Test Suite - Résultats Détaillés</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 15px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            .summary-box {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
            }}
            .summary-item {{
                text-align: center;
                padding: 15px;
                background: white;
                border-radius: 5px;
            }}
            .summary-item .label {{
                font-size: 0.9em;
                color: #7f8c8d;
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }}
            .summary-item .value {{
                font-size: 2em;
                font-weight: bold;
            }}
            .passed {{ color: #27ae60; }}
            .failed {{ color: #e74c3c; }}
            .warning {{ color: #f39c12; }}
            .error {{ color: #e74c3c; }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .status-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: bold;
            }}
            .badge-passed {{
                background: #27ae6020;
                color: #27ae60;
            }}
            .badge-failed {{
                background: #e74c3c20;
                color: #e74c3c;
            }}
            .badge-warning {{
                background: #f39c1220;
                color: #f39c12;
            }}
            .badge-error {{
                background: #e74c3c20;
                color: #e74c3c;
            }}
            .test-description {{
                color: #7f8c8d;
                font-size: 0.9em;
                font-style: italic;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 2px solid #ddd;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📋 Evidently Test Suite - Résultats Détaillés</h1>
            
            <h2>📊 Résumé Global</h2>
            <div class="summary-box">
                <div class="summary-item">
                    <span class="label">Tests Réussis</span>
                    <span class="value passed">✓ {passed}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Tests Échoués</span>
                    <span class="value failed">✗ {failed}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Warnings</span>
                    <span class="value warning">⚠ {warning}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Erreurs</span>
                    <span class="value error">⚡ {error}</span>
                </div>
            </div>
            
            <h2>🔬 Détails des Tests</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>Nom du Test</th>
                        <th style="width: 120px;">Statut</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
    """.format(
        passed=tests_summary['passed'],
        failed=tests_summary['failed'],
        warning=tests_summary['warning'],
        error=tests_summary['error']
    )
    
    for idx, test in enumerate(tests, 1):
        test_name = test.get('name', 'Unknown Test')
        raw_status = test.get('status', 'unknown').lower()
        description = test.get('description', 'Aucune description disponible')
        
        if raw_status == 'success':
            display_status = 'passed'
            status_icon = '✓'
        elif raw_status == 'fail':
            display_status = 'failed'
            status_icon = '✗'
        elif raw_status == 'warning':
            display_status = 'warning'
            status_icon = '⚠'
        elif raw_status == 'error':
            display_status = 'error'
            status_icon = '⚡'
        else:
            display_status = 'error'
            status_icon = '?'

        status_class = f"badge-{display_status}"
        
        html_content += f"""
                    <tr>
                        <td><strong>{idx}</strong></td>
                        <td><strong>{test_name}</strong></td>
                        <td><span class="status-badge {status_class}">{status_icon} {raw_status.upper()}</span></td>
                        <td class="test-description">{description}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
            
            <div class="footer">
                <p><strong>Rapport généré le {timestamp}</strong></p>
                <p>Powered by Evidently AI • Pipeline Airflow</p>
            </div>
        </div>
    </body>
    </html>
    """.format(timestamp=datetime.now().strftime('%d/%m/%Y à %H:%M:%S'))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ Test suite HTML textuel généré: {output_path}")

# ========================== TÂCHES DU DAG ==========================

def setup_aws_environment():
    """Configure l'environnement AWS."""
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"] = Variable.get("AWS_DEFAULT_REGION")
        return True
    except Exception as e:
        logger.error(f"AWS setup failed: {e}")
        raise

def download_weather_csv_from_s3(**context):
    """Télécharge le fichier CSV depuis S3."""
    logger.info("⬇️ Downloading weather data CSV from S3...")
    setup_aws_environment()
    s3 = boto3.client('s3')
    bucket_name = Variable.get("BUCKET")
    local_path = os.path.join(DATA_PATH, WEATHER_CSV_FILE)
    os.makedirs(DATA_PATH, exist_ok=True)
    try:
        s3.download_file(bucket_name, WEATHER_CSV_FILE, local_path)
        context["ti"].xcom_push(key="local_weather_csv", value=local_path)
        logger.info(f"✓ CSV downloaded: {local_path}")
    except Exception as e:
        raise RuntimeError(f"S3 download failed: {e}")

def evidently_data_quality_check(**context):
    """
    Génère :
    1. Un rapport de drift visuel (avec graphiques partiels)
    2. Une test suite en format HTML textuel (sans graphiques)
    """
    logger.info("🔍 Starting Evidently analysis...")
    
    csv_path = context["ti"].xcom_pull(task_ids="download_weather_csv", key="local_weather_csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    paris_df = pd.read_csv(csv_path)
    paris_df['datetime'] = pd.to_datetime(paris_df['datetime'])
    paris_df = paris_df.sort_values('datetime')
    
    split_idx = len(paris_df) // 2
    reference_data, current_data = paris_df.iloc[:split_idx], paris_df.iloc[split_idx:]
    
    numerical_columns = [c for c in ['temp', 'feels_like', 'pressure', 'humidity', 'dew_point', 
                                     'clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h'] 
                        if c in paris_df.columns]
    os.makedirs(REPORTS_PATH, exist_ok=True)
    
    # --- RAPPORT DE DRIFT ---
    logger.info("📊 Generating Data Drift report (visual)...")
    data_drift_report = Report(metrics=[DataDriftPreset(columns=numerical_columns)])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    report_path = os.path.join(REPORTS_PATH, "weather_paris_drift_report.html")
    data_drift_report.save_html(report_path)
    logger.info(f"✓ Drift report saved: {report_path}")
    
    # --- TEST SUITE ---
    logger.info("🧪 Running Test Suite and generating textual HTML...")
    data_drift_test_suite = TestSuite(tests=[
        DataDriftTestPreset(columns=numerical_columns),
        TestNumberOfColumns(),
        TestNumberOfRows(),
        TestNumberOfMissingValues(),
        TestShareOfMissingValues()
    ])
    data_drift_test_suite.run(reference_data=reference_data, current_data=current_data)
    
    test_suite_path = os.path.join(REPORTS_PATH, "weather_paris_test_suite_textual.html")
    generate_test_suite_html(data_drift_test_suite, test_suite_path)
    logger.info(f"✓ Test suite textual report saved: {test_suite_path}")

    # --- EXTRACTION DES RÉSULTATS (Evidently 0.4.x) ---
    test_results = data_drift_test_suite.as_dict()
    tests_summary = {'passed': 0, 'failed': 0, 'warning': 0, 'error': 0}

    for test in test_results.get('tests', []):
        status = test.get('status', 'unknown').lower()
        if status == 'success':
            tests_summary['passed'] += 1
        elif status == 'fail':
            tests_summary['failed'] += 1
        elif status == 'warning':
            tests_summary['warning'] += 1
        elif status == 'error':
            tests_summary['error'] += 1
        else:
            logger.warning(f"Statut inconnu : '{status}' pour le test '{test.get('name', 'N/A')}'")

    summary = {
        'total_rows': len(paris_df),
        'reference_rows': len(reference_data),
        'current_rows': len(current_data),
        'reference_period': f"{reference_data['datetime'].min()} to {reference_data['datetime'].max()}",
        'current_period': f"{current_data['datetime'].min()} to {current_data['datetime'].max()}",
        'tests_passed': tests_summary['passed'],
        'tests_failed': tests_summary['failed'],
        'tests_warning': tests_summary['warning'],
        'tests_error': tests_summary['error'],
        'columns_analyzed': numerical_columns,
        'report_path': report_path,
        'test_suite_path': test_suite_path
    }
    
    context["ti"].xcom_push(key="evidently_summary", value=summary)
    logger.info("✅ Evidently analysis complete.")
    return summary

def upload_reports_to_s3(**context):
    """Téléverse les deux rapports sur S3."""
    logger.info("⬆️ Uploading reports to S3...")
    setup_aws_environment()
    s3_client = boto3.client('s3')
    bucket_name = Variable.get("BUCKET")
    summary = context["ti"].xcom_pull(task_ids="evidently_check", key="evidently_summary")
    uploaded_urls = {}
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for key, path in [("report_url", summary.get('report_path')), 
                      ("test_suite_url", summary.get('test_suite_path'))]:
        if path and os.path.exists(path):
            s3_key = f"evidently-reports/{timestamp}/{os.path.basename(path)}"
            s3_client.upload_file(
                path, 
                bucket_name, 
                s3_key, 
                ExtraArgs={'ContentType': 'text/html', 'ContentDisposition': 'inline'}
            )
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=604800  # 7 jours
            )
            uploaded_urls[key] = url
            logger.info(f"✓ Uploaded {key}: {s3_key}")
    
    context["ti"].xcom_push(key="report_urls", value=uploaded_urls)
    logger.info("✅ All reports uploaded to S3")

# ========================== OPÉRATEUR EMAIL ==========================

class S3ReportEmailOperator(BaseOperator):
    """Envoie un email avec les liens vers les deux rapports."""
    
    def __init__(self, to: str, subject: str, conn_id: str = "smtp_default", **kwargs):
        super().__init__(**kwargs)
        self.to, self.subject, self.conn_id = to, subject, conn_id

    def execute(self, context):
        conn = BaseHook.get_connection(self.conn_id)
        smtp_host = conn.extra_dejson.get("smtp_host", "smtp.gmail.com")
        smtp_port = conn.extra_dejson.get("smtp_port", 587)
        use_tls = conn.extra_dejson.get("smtp_starttls", True)
        
        summary = context["ti"].xcom_pull(task_ids="evidently_check", key="evidently_summary")
        report_urls = context["ti"].xcom_pull(task_ids="upload_reports_to_s3", key="report_urls")
        
        html_content = self._build_html_content(summary, report_urls) if summary else "<p>Pas de résumé disponible</p>"
        
        msg = MIMEMultipart()
        msg["Subject"] = self.subject
        msg["From"] = f"- 𖣘 - Airflow 𓏺〢 Evidently : Meteo <{conn.login}>"
        msg["To"] = self.to
        msg.attach(MIMEText(html_content, "html", "utf-8"))
        
        server = smtplib.SMTP(smtp_host, smtp_port)
        if use_tls:
            server.starttls()
        server.login(conn.login, conn.password)
        server.send_message(msg)
        server.quit()
        self.log.info(f"✅ Email sent to {self.to}")

    def _build_html_content(self, summary, report_urls=None):
        tests_total = max(1, sum([summary.get('tests_passed', 0), summary.get('tests_failed', 0),
                                   summary.get('tests_warning', 0), summary.get('tests_error', 0)]))
        
        is_success = summary.get('tests_failed', 0) == 0 and summary.get('tests_error', 0) == 0
        status_icon = "✅" if is_success else "⚠️"
        status_text = "Aucun problème détecté" if is_success else "Des problèmes ont été détectés"
        status_color = "#27ae60" if is_success else "#f39c12"
        
        report_url = report_urls.get('report_url', '#') if report_urls else '#'
        test_suite_url = report_urls.get('test_suite_url', '#') if report_urls else '#'

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 30px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }}
                .summary-box {{
                    background: {status_color}15;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 5px solid {status_color};
                }}
                .summary-box h2 {{
                    color: {status_color};
                    margin-top: 0;
                    border: none;
                    padding: 0;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-label {{
                    font-weight: bold;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    display: block;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 1.8em;
                    color: #2c3e50;
                    font-weight: bold;
                }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .columns-list {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: monospace;
                    font-size: 0.9em;
                }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196f3;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 15px 0;
                }}
                .report-link {{
                    display: inline-block;
                    padding: 12px 20px;
                    background: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    margin: 10px 5px;
                    font-weight: bold;
                }}
                .report-link:hover {{
                    background: #2980b9;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 2px solid #ddd;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>𖣘 Airflow : Evidently Data Quality Check</h1>
                <div class="summary-box">
                    <h2>{status_icon} Statut Global : {status_text}</h2>
                    <p style="margin:10px 0 0 0"><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
                </div>
                
                <h2>📊 Résumé</h2>
                <div class="metrics-grid">
                    <div class="metric"><span class="metric-label">Total lignes</span><span class="metric-value">{summary['total_rows']:,}</span></div>
                    <div class="metric"><span class="metric-label">Référence</span><span class="metric-value">{summary['reference_rows']:,}</span></div>
                    <div class="metric"><span class="metric-label">Actuel</span><span class="metric-value">{summary['current_rows']:,}</span></div>
                </div>
                
                <h2>🔬 Résultats des Tests</h2>
                <table>
                    <tr><th>Statut</th><th>Nombre</th><th>%</th></tr>
                    <tr><td><span class="success">✓ Réussis</span></td><td><strong>{summary['tests_passed']}</strong></td><td>{(summary['tests_passed']/tests_total*100):.1f}%</td></tr>
                    <tr><td><span class="error">✗ Échoués</span></td><td><strong>{summary['tests_failed']}</strong></td><td>{(summary['tests_failed']/tests_total*100):.1f}%</td></tr>
                    <tr><td><span class="warning">⚠ Warnings</span></td><td><strong>{summary['tests_warning']}</strong></td><td>{(summary['tests_warning']/tests_total*100):.1f}%</td></tr>
                    <tr style="font-weight:bold;background:#f0f0f0"><td>Total</td><td>{tests_total}</td><td>100%</td></tr>
                </table>
                
                <h2>📈 Périodes</h2>
                <table>
                    <tr><td><strong>Référence</strong></td><td>{summary['reference_period']}</td></tr>
                    <tr><td><strong>Actuelle</strong></td><td>{summary['current_period']}</td></tr>
                </table>
                
                <h2>🔍 Colonnes Analysées</h2>
                <div class="columns-list">{', '.join(summary['columns_analyzed'])}</div>
                
                <div class="info-box">
                    <h2 style="margin-top:0;color:#1976d2">🔗 Rapports Détaillés</h2>
                    <p>Deux rapports sont disponibles en ligne :</p>
                    <div style="text-align:center;margin:20px 0">
                        <a href="{report_url}" target="_blank" class="report-link">📊 Rapport de Drift Visuel</a>
                        <a href="{test_suite_url}" target="_blank" class="report-link">📋 Test Suite Détaillée</a>
                    </div>
                    <ul style="font-size:0.9em;color:#666;margin-top:15px">
                        <li><strong>Rapport de Drift</strong> : visualisations et statistiques sur l'évolution des données</li>
                        <li><strong>Test Suite</strong> : liste détaillée de tous les tests avec leur statut</li>
                    </ul>
                    <p style="font-size:0.85em;color:#999;margin-top:15px">💡 Les rapports restent accessibles pendant 7 jours</p>
                </div>
                
                <div class="footer">
                    <p><strong>Rapport généré automatiquement par Airflow</strong></p>
                    <p>🌤️ Pipeline Météo Paris • Powered by Evidently AI</p>
                </div>
            </div>
        </body>
        </html>
        """

# ========================== DÉFINITION DU DAG ==========================

with DAG(
    dag_id='meteo_02_evidently',
    default_args=default_args,
    description='Rapport Drift visuel + Test Suite textuelle sans graphiques',
    schedule=None,
    catchup=False,
    tags=['evidently', 'data-quality', 'final'],
) as dag:

    task_download_csv = PythonOperator(
        task_id='download_weather_csv',
        python_callable=download_weather_csv_from_s3
    )
    
    task_evidently_check = PythonOperator(
        task_id='evidently_check',
        python_callable=evidently_data_quality_check
    )
    
    task_upload_reports = PythonOperator(
        task_id='upload_reports_to_s3',
        python_callable=upload_reports_to_s3
    )
    
    task_send_email = S3ReportEmailOperator(
        task_id='send_evidently_report_email',
        to="en9.eric@gmail.com",
        subject="- 𖣘 - Airflow 𓏺〢 Evidently : Data Quality Météo",
        conn_id="smtp_default"
    )
    trigger_ml_pipeline = TriggerDagRunOperator(
        task_id="meteo_03_ml_pipeline",
        trigger_dag_id="meteo_03_ml_pipeline",
        wait_for_completion=False,  # mettre True si tu veux que ce DAG attende sa fin
        reset_dag_run=True,
    )


    task_download_csv >> task_evidently_check >> task_upload_reports >> task_send_email >> trigger_ml_pipeline