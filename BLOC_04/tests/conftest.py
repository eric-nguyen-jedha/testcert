import sys
from unittest.mock import MagicMock
import pytest


# 🛡️ Mock des modules Airflow pour permettre l'import hors environnement Airflow
# (ex: dans Jenkins, où apache-airflow n'est pas installé)
_airflow_modules = [
    "airflow",
    "airflow.models",
    "airflow.models.Variable",
    "airflow.exceptions",
    "airflow.providers",
    "airflow.providers.amazon",
    "airflow.providers.amazon.aws",
    "airflow.providers.amazon.aws.hooks",
    "airflow.providers.amazon.aws.hooks.s3",
]

for mod_name in _airflow_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# 🧪 Fixture utilitaire pour les tests
@pytest.fixture
def mock_ti():
    """Fixture pour mocker le TaskInstance (XCom)."""
    ti = MagicMock()
    yield ti
# -*- coding: utf-8 -*-
"""
Fixtures pytest partagées pour tous les tests
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, Mock
from datetime import datetime, timedelta

# Ajouter le répertoire dags au path pour pouvoir importer les modules
dags_dir = os.path.join(os.path.dirname(__file__), '..', 'dags')
if dags_dir not in sys.path:
    sys.path.insert(0, dags_dir)

# ============================================================================
# FIXTURES - Données de test
# ============================================================================

@pytest.fixture
def sample_weather_api_response():
    """
    Mock d'une réponse complète de l'API OpenWeather
    Simule une journée ensoleillée à Paris
    """
    return {
        "coord": {"lon": 2.3522, "lat": 48.8566},
        "weather": [
            {
                "id": 800,
                "main": "Clear",
                "description": "clear sky",
                "icon": "01d"
            }
        ],
        "base": "stations",
        "main": {
            "temp": 18.5,
            "feels_like": 17.8,
            "temp_min": 16.2,
            "temp_max": 20.1,
            "pressure": 1015,
            "humidity": 62
        },
        "visibility": 10000,
        "wind": {
            "speed": 3.5,
            "deg": 220
        },
        "clouds": {"all": 20},
        "dt": 1697500000,
        "sys": {
            "type": 2,
            "id": 2041230,
            "country": "FR",
            "sunrise": 1697433600,
            "sunset": 1697474400
        },
        "timezone": 3600,
        "id": 2988507,
        "name": "Paris",
        "cod": 200
    }


@pytest.fixture
def sample_weather_api_response_with_rain():
    """Mock d'une réponse API avec de la pluie"""
    return {
        "dt": 1697500000,
        "main": {
            "temp": 12.5,
            "feels_like": 11.2,
            "pressure": 1008,
            "humidity": 85
        },
        "clouds": {"all": 90},
        "visibility": 5000,
        "wind": {"speed": 6.5, "deg": 180},
        "rain": {"1h": 2.5},
        "weather": [{"main": "Rain"}]
    }


@pytest.fixture
def sample_weather_csv_data():
    """
    Données CSV d'historique météo pour tests d'entraînement
    200 échantillons avec distribution réaliste
    """
    np.random.seed(42)
    n_samples = 200
    
    # Générer des données météo réalistes
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Températures avec saisonnalité
    temps_base = 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365))
    temps = temps_base + np.random.normal(0, 3, n_samples)
    
    return pd.DataFrame({
        'datetime': dates,
        'temp': temps,
        'feels_like': temps - np.random.uniform(0, 2, n_samples),
        'pressure': np.random.normal(1013, 10, n_samples),
        'humidity': np.random.randint(40, 90, n_samples),
        'clouds': np.random.randint(0, 100, n_samples),
        'visibility': np.random.choice([5000, 8000, 10000], n_samples),
        'wind_speed': np.abs(np.random.normal(3, 2, n_samples)),
        'wind_deg': np.random.randint(0, 360, n_samples),
        'rain_1h': np.random.exponential(0.5, n_samples),
        'weather_main': np.random.choice(
            ['Clear', 'Clouds', 'Rain', 'Fog', 'Drizzle'],
            n_samples,
            p=[0.4, 0.3, 0.15, 0.1, 0.05]
        ),
        'weather_description': ['test'] * n_samples
    })


@pytest.fixture
def sample_preprocessed_features():
    """
    Features prétraitées prêtes pour la prédiction
    Format attendu par les modèles (17 colonnes)
    """
    return pd.DataFrame({
        'temp': [18.5],
        'feels_like': [17.8],
        'pressure': [1015],
        'humidity': [62],
        'clouds': [20],
        'visibility': [10000],
        'wind_speed': [3.5],
        'wind_deg': [220],
        'rain_1h': [0.0],
        'hour': [14],
        'month': [10],
        'weekday': [3],
        'is_weekend': [0],
        'hour_sin': [0.7071],
        'hour_cos': [0.7071],
        'month_sin': [0.5],
        'month_cos': [0.866]
    })


# ============================================================================
# FIXTURES - Mocks AWS/MLflow
# ============================================================================

@pytest.fixture
def mock_s3_client():
    """Mock du client boto3 S3"""
    mock_client = MagicMock()
    mock_client.upload_file = Mock()
    mock_client.download_file = Mock()
    mock_client.list_objects_v2 = Mock(return_value={
        'Contents': [
            {'Key': 'historical.csv', 'LastModified': datetime.now()},
            {'Key': 'forecast_6h.csv', 'LastModified': datetime.now()}
        ]
    })
    return mock_client


@pytest.fixture
def mock_mlflow_model():
    """Mock d'un modèle MLflow"""
    mock_model = MagicMock()
    # Par défaut, prédit la classe 1 (Clouds)
    mock_model.predict = Mock(return_value=np.array([1]))
    mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.7, 0.1, 0.1]]))
    return mock_model


@pytest.fixture
def mock_mlflow_client():
    """Mock du client MLflow"""
    mock_client = MagicMock()
    mock_client.download_artifacts = Mock()
    mock_client.list_experiments = Mock(return_value=[
        MagicMock(name='Meteo', experiment_id='1')
    ])
    return mock_client


@pytest.fixture
def mock_airflow_variable():
    """Mock des Variables Airflow"""
    def get_variable(key, default_var=None):
        variables = {
            'AWS_ACCESS_KEY_ID': 'test_access_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret_key',
            'AWS_DEFAULT_REGION': 'eu-west-3',
            'BUCKET': 'test-weather-bucket',
            'OPEN_WEATHER_API_KEY': 'test_api_key_123456',
            'mlflow_uri': 'http://localhost:8081'
        }
        return variables.get(key, default_var)
    
    return get_variable


# ============================================================================
# FIXTURES - Label Encoder
# ============================================================================

@pytest.fixture
def sample_label_encoder():
    """LabelEncoder avec les classes météo standards"""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['Clear', 'Clouds', 'Fog', 'Rain', 'Snow'])
    return le


# ============================================================================
# FIXTURES - Configuration
# ============================================================================

@pytest.fixture
def cities_config():
    """Configuration des villes pour les prédictions"""
    return {
        'paris': {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
        'lyon': {'lat': 45.7640, 'lon': 4.8357, 'name': 'Lyon'},
        'marseille': {'lat': 43.2965, 'lon': 5.3698, 'name': 'Marseille'}
    }


@pytest.fixture
def weather_code_mapping():
    """Mapping des codes météo vers les labels"""
    return {
        0: 'Clear',
        1: 'Clouds',
        2: 'Fog',
        3: 'Rain',
        4: 'Snow'
    }


# ============================================================================
# FIXTURES - Chemins temporaires
# ============================================================================

@pytest.fixture
def temp_csv_file(tmp_path, sample_weather_csv_data):
    """Créer un fichier CSV temporaire pour les tests"""
    csv_file = tmp_path / "weather_test.csv"
    sample_weather_csv_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Créer un répertoire temporaire pour les modèles"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


# ============================================================================
# FIXTURES - Contexte Airflow
# ============================================================================

@pytest.fixture
def mock_airflow_context():
    """Mock du contexte d'exécution Airflow"""
    return {
        'dag': MagicMock(),
        'task': MagicMock(),
        'ti': MagicMock(),
        'execution_date': datetime(2023, 10, 15),
        'dag_run': MagicMock(conf={}),
        'params': {}
    }


# ============================================================================
# HOOKS - Configuration pytest
# ============================================================================

def pytest_configure(config):
    """Configuration pytest au démarrage"""
    config.addinivalue_line(
        "markers", "unit: Tests unitaires rapides"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration"
    )
    config.addinivalue_line(
        "markers", "slow: Tests lents (>1s)"
    )
    config.addinivalue_line(
        "markers", "api: Tests nécessitant API externe"
    )
    config.addinivalue_line(
        "markers", "mlflow: Tests liés à MLflow"
    )
    config.addinivalue_line(
        "markers", "s3: Tests liés à S3/AWS"
    )


def pytest_collection_modifyitems(config, items):
    """Modifier la collection des tests"""
    # Ajouter automatiquement le marker 'unit' aux tests non marqués
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# FIXTURES - Nettoyage
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Nettoyer l'environnement avant et après chaque test"""
    # Setup
    original_env = os.environ.copy()
    
    yield
    
    # Teardown - restaurer les variables d'environnement
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Réinitialiser le seed random pour chaque test"""
    np.random.seed(42)
    yield
    np.random.seed()
