# -*- coding: utf-8 -*-
"""
Tests unitaires et d'intégration pour les DAGs météo
À exécuter sur Jenkins indépendamment d'Airflow
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

# Import des fonctions à tester (à adapter selon votre structure)
# Hypothèse : les fonctions sont extraites dans des modules séparés
import sys
sys.path.insert(0, '/opt/airflow/dags')

# =============================================================================
# 🛡️ MOCK D'AIRFLOW POUR FONCTIONNER HORS ENVIRONNEMENT AIRFLOW (ex: Jenkins)
# =============================================================================
import sys
from unittest.mock import MagicMock

_airflow_modules = [
    "airflow",
    "airflow.models",
    "airflow.models.Variable",
    "airflow.exceptions",
    "airflow.operators",
    "airflow.operators.python",
    "airflow.providers",
    "airflow.providers.amazon",
    "airflow.providers.amazon.aws",
    "airflow.providers.amazon.aws.hooks",
    "airflow.providers.amazon.aws.hooks.s3",
    "airflow.providers.postgres",
    "airflow.providers.postgres.hooks",
    "airflow.providers.postgres.hooks.postgres",
]

for mod in _airflow_modules:
    sys.modules[mod] = MagicMock()

# Mock plugin custom si utilisé
sys.modules["s3_to_postgres"] = MagicMock()

# Mock Variable.get pour éviter les erreurs de base de données
class MockVariable:
    @staticmethod
    def get(key, default_var=None):
        return {
            "BUCKET": "test-bucket",
            "AWS_ACCESS_KEY_ID": "fake",
            "AWS_SECRET_ACCESS_KEY": "fake",
            "AWS_DEFAULT_REGION": "eu-west-3",
            "OPEN_WEATHER_API_KEY": "fake_key",
            "mlflow_uri": "http://localhost:8081",
        }.get(key, default_var or f"mock_{key}")

sys.modules["airflow.models.Variable"].Variable = MockVariable
# =============================================================================

from realtime_prediction_forecast import (
    preprocess_weather_json,
    WEATHER_CODE_MAPPING
)

# ============================================================================
# TESTS UNITAIRES - Prétraitement des données
# ============================================================================

class TestPreprocessWeatherJson:
    """Tests unitaires pour la fonction de prétraitement"""
    
    @pytest.fixture
    def sample_api_response(self):
        """Mock d'une réponse de l'API OpenWeather"""
        return {
            "dt": 1697500000,
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "pressure": 1013,
                "humidity": 65
            },
            "clouds": {"all": 75},
            "visibility": 10000,
            "wind": {
                "speed": 3.5,
                "deg": 180
            },
            "rain": {"1h": 0.5}
        }
    
    def test_preprocess_returns_dataframe(self, sample_api_response):
        """Test que la fonction retourne un DataFrame"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_preprocess_has_correct_columns(self, sample_api_response):
        """Test que le DataFrame contient les 17 colonnes attendues"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        expected_columns = [
            'temp', 'feels_like', 'pressure', 'humidity', 'clouds',
            'visibility', 'wind_speed', 'wind_deg', 'rain_1h',
            'hour', 'month', 'weekday', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        assert list(result.columns) == expected_columns
        assert len(result.columns) == 17
    
    def test_preprocess_no_timestamp_for_historical(self, sample_api_response):
        """Test que timestamp n'est pas présent pour le modèle historical"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert 'timestamp' not in result.columns
    
    def test_preprocess_no_timestamp_for_forecast(self, sample_api_response):
        """Test que timestamp n'est pas présent pour le modèle forecast_6h"""
        result = preprocess_weather_json(sample_api_response, model_type='forecast_6h')
        assert 'timestamp' not in result.columns
    
    def test_preprocess_extracts_basic_features(self, sample_api_response):
        """Test l'extraction correcte des features de base"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert result['temp'].iloc[0] == 15.5
        assert result['feels_like'].iloc[0] == 14.2
        assert result['pressure'].iloc[0] == 1013
        assert result['humidity'].iloc[0] == 65
        assert result['wind_speed'].iloc[0] == 3.5
    
    def test_preprocess_handles_missing_rain(self):
        """Test la gestion des données de pluie manquantes"""
        api_response = {
            "dt": 1697500000,
            "main": {"temp": 15.5, "feels_like": 14.2, "pressure": 1013, "humidity": 65},
            "clouds": {"all": 75},
            "visibility": 10000,
            "wind": {"speed": 3.5, "deg": 180}
            # Pas de "rain"
        }
        result = preprocess_weather_json(api_response, model_type='historical')
        assert result['rain_1h'].iloc[0] == 0.0
    
    def test_preprocess_creates_temporal_features(self, sample_api_response):
        """Test la création des features temporelles"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert 'hour' in result.columns
        assert 'month' in result.columns
        assert 'weekday' in result.columns
        assert 'is_weekend' in result.columns
        assert result['hour'].iloc[0] >= 0 and result['hour'].iloc[0] <= 23
        assert result['month'].iloc[0] >= 1 and result['month'].iloc[0] <= 12
    
    def test_preprocess_creates_cyclical_features(self, sample_api_response):
        """Test la création des features cycliques"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        # Vérifier que les valeurs sont dans [-1, 1]
        assert -1 <= result['hour_sin'].iloc[0] <= 1
        assert -1 <= result['hour_cos'].iloc[0] <= 1
    
    def test_preprocess_all_numeric_types(self, sample_api_response):
        """Test que toutes les colonnes sont numériques"""
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_preprocess_weekend_detection(self, sample_api_response):
        """Test la détection correcte du weekend"""
        # Modifier le timestamp pour un samedi (weekday=5)
        saturday_timestamp = 1697270400  # 14 oct 2023 (samedi)
        sample_api_response['dt'] = saturday_timestamp
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert result['is_weekend'].iloc[0] == 1
        
        # Tester un jour de semaine
        monday_timestamp = 1697011200  # 11 oct 2023 (lundi)
        sample_api_response['dt'] = monday_timestamp
        result = preprocess_weather_json(sample_api_response, model_type='historical')
        assert result['is_weekend'].iloc[0] == 0


# ============================================================================
# TESTS UNITAIRES - Mapping des codes météo
# ============================================================================

class TestWeatherCodeMapping:
    """Tests pour le mapping des codes vers les labels"""
    
    def test_mapping_contains_all_classes(self):
        """Test que le mapping contient toutes les classes attendues"""
        expected_classes = ['Clear', 'Clouds', 'Fog', 'Rain', 'Snow']
        assert len(WEATHER_CODE_MAPPING) == 5
        for weather_class in expected_classes:
            assert weather_class in WEATHER_CODE_MAPPING.values()
    
    def test_mapping_correct_order(self):
        """Test que le mapping respecte l'ordre alphabétique (LabelEncoder)"""
        expected_order = {
            0: 'Clear',
            1: 'Clouds',
            2: 'Fog',
            3: 'Rain',
            4: 'Snow'
        }
        assert WEATHER_CODE_MAPPING == expected_order
    
    def test_mapping_codes_are_sequential(self):
        """Test que les codes sont séquentiels de 0 à n-1"""
        codes = sorted(WEATHER_CODE_MAPPING.keys())
        assert codes == list(range(len(WEATHER_CODE_MAPPING)))


# ============================================================================
# TESTS UNITAIRES - Fonctions d'entraînement (DAG paris_meteo_ml_pipeline)
# ============================================================================

class TestDataPreparation:
    """Tests pour la préparation des données d'entraînement"""
    
    @pytest.fixture
    def sample_weather_csv(self):
        """Créer un CSV de test"""
        data = {
            'datetime': pd.date_range('2023-01-01', periods=100, freq='H'),
            'temp': np.random.uniform(0, 30, 100),
            'feels_like': np.random.uniform(0, 30, 100),
            'pressure': np.random.uniform(990, 1030, 100),
            'humidity': np.random.randint(30, 90, 100),
            'clouds': np.random.randint(0, 100, 100),
            'visibility': np.random.randint(5000, 10000, 100),
            'wind_speed': np.random.uniform(0, 10, 100),
            'wind_deg': np.random.randint(0, 360, 100),
            'rain_1h': np.random.uniform(0, 5, 100),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Fog'], 100),
            'weather_description': ['test'] * 100
        }
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            # Utiliser 'yield' pour s'assurer que le fichier est supprimé après le test
            yield f.name
        os.unlink(f.name) # Nettoyage explicite
    
    @pytest.mark.integration
    def test_csv_has_required_columns(self, sample_weather_csv):
        """Test que le CSV contient les colonnes nécessaires"""
        df = pd.read_csv(sample_weather_csv)
        required_columns = [
            'datetime', 'temp', 'feels_like', 'pressure', 'humidity',
            'clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h',
            'weather_main'
        ]
        for col in required_columns:
            assert col in df.columns
    
    def test_weather_classes_cleaning(self):
        """Test le nettoyage des classes météo (Drizzle -> Rain, Mist -> Fog)"""
        df = pd.DataFrame({
            'weather_main': ['Clear', 'Drizzle', 'Mist', 'Rain', 'Clouds']
        })
        df['weather_main'] = df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
        assert 'Drizzle' not in df['weather_main'].values
        assert 'Mist' not in df['weather_main'].values
        assert 'Rain' in df['weather_main'].values
        assert 'Fog' in df['weather_main'].values
    
    def test_temporal_features_creation(self):
        """Test la création des features temporelles"""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-10-15 14:30:00'])
        })
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        
        assert df['hour'].iloc[0] == 14
        assert df['month'].iloc[0] == 10
        assert df['weekday'].iloc[0] == 6  # Dimanche


# ============================================================================
# TESTS D'INTÉGRATION - Workflow complet
# ============================================================================

class TestFullPredictionWorkflow:
    """Tests d'intégration du workflow complet de prédiction"""
    
    @patch('realtime_prediction_forecast.fetch_weather')
    @patch('realtime_prediction_forecast.mlflow.pyfunc.load_model')
    @patch('realtime_prediction_forecast.boto3.client')
    def test_predict_weather_end_to_end(self, mock_s3, mock_mlflow, mock_fetch):
        """Test du workflow complet de prédiction pour une ville"""
        
        # Mock de la réponse API
        mock_fetch.return_value = {
            "dt": 1697500000,
            "main": {"temp": 15.5, "feels_like": 14.2, "pressure": 1013, "humidity": 65},
            "clouds": {"all": 75},
            "visibility": 10000,
            "wind": {"speed": 3.5, "deg": 180},
            "rain": {"1h": 0.5}
        }
        
        # Mock du modèle MLflow
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])  # Clouds
        mock_mlflow.return_value = mock_model
        
        # Mock S3
        mock_s3_client = MagicMock()
        mock_s3.return_value = mock_s3_client
        
        # Tester le prétraitement
        result = preprocess_weather_json(mock_fetch.return_value, model_type='historical')
        
        # Vérifications
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.shape[1] == 17
        
        # Vérifier que la prédiction peut être faite
        prediction = mock_model.predict(result)
        assert prediction[0] == 1
        
        # Vérifier le mapping
        weather_label = WEATHER_CODE_MAPPING.get(int(prediction[0]))
        assert weather_label == 'Clouds'
    
    @patch('realtime_prediction_forecast.Variable.get')
    def test_environment_setup(self, mock_variable_get):
        """Test de la configuration de l'environnement"""
        mock_variable_get.side_effect = lambda x, **kwargs: {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_DEFAULT_REGION': 'eu-west-3',
            'BUCKET': 'test-bucket'
        }.get(x, kwargs.get('default_var', ''))
        
        # Vérifier que les variables peuvent être récupérées
        assert mock_variable_get('AWS_ACCESS_KEY_ID') == 'test_key'
        assert mock_variable_get('BUCKET') == 'test-bucket'


# ============================================================================
# TESTS DE ROBUSTESSE - Cas limites
# ============================================================================

class TestEdgeCases:
    """Tests des cas limites et erreurs"""
    
    def test_preprocess_with_none_visibility(self):
        """Test avec visibility manquante"""
        api_response = {
            "dt": 1697500000,
            "main": {"temp": 15.5, "feels_like": 14.2, "pressure": 1013, "humidity": 65},
            "clouds": {"all": 75},
            "wind": {"speed": 3.5, "deg": 180}
            # Pas de visibility
        }
        result = preprocess_weather_json(api_response, model_type='historical')
        assert pd.isna(result['visibility'].iloc[0]) or result['visibility'].iloc[0] is not None
    
    def test_preprocess_extreme_values(self):
        """Test avec des valeurs extrêmes"""
        api_response = {
            "dt": 1697500000,
            "main": {"temp": -40, "feels_like": -45, "pressure": 950, "humidity": 100},
            "clouds": {"all": 100},
            "visibility": 100,
            "wind": {"speed": 50, "deg": 359},
            "rain": {"1h": 100}
        }
        result = preprocess_weather_json(api_response, model_type='historical')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_weather_code_mapping_unknown_code(self):
        """Test du comportement avec un code inconnu"""
        unknown_code = 99
        weather_label = WEATHER_CODE_MAPPING.get(unknown_code, f"Unknown_{unknown_code}")
        assert weather_label == "Unknown_99"
    
    def test_cyclical_features_bounds(self):
        """Test que les features cycliques sont dans les bonnes bornes"""
        api_response = {
            "dt": 1697500000,
            "main": {"temp": 15, "feels_like": 14, "pressure": 1013, "humidity": 65},
            "clouds": {"all": 50},
            "visibility": 10000,
            "wind": {"speed": 3, "deg": 180}
        }
        result = preprocess_weather_json(api_response, model_type='historical')
        
        # Les valeurs sin/cos doivent être entre -1 et 1
        assert -1 <= result['hour_sin'].iloc[0] <= 1
        assert -1 <= result['hour_cos'].iloc[0] <= 1
        assert -1 <= result['month_sin'].iloc[0] <= 1
        assert -1 <= result['month_cos'].iloc[0] <= 1


# ============================================================================
# TESTS DE VALIDATION DES DONNÉES
# ============================================================================

class TestDataValidation:
    """Tests de validation des données"""
    
    def test_no_null_values_in_critical_features(self):
        """Test qu'il n'y a pas de valeurs nulles dans les features critiques"""
        api_response = {
            "dt": 1697500000,
            "main": {"temp": 15.5, "feels_like": 14.2, "pressure": 1013, "humidity": 65},
            "clouds": {"all": 75},
            "visibility": 10000,
            "wind": {"speed": 3.5, "deg": 180},
            "rain": {"1h": 0.5}
        }
        result = preprocess_weather_json(api_response, model_type='historical')
        
        critical_features = ['temp', 'pressure', 'humidity', 'hour', 'month']
        for feature in critical_features:
            assert not pd.isna(result[feature].iloc[0])
    
    def test_feature_types_consistency(self):
        """Test la cohérence des types de données"""
        api_response = {
            "dt": 1697500000,
            "main": {"temp": 15.5, "feels_like": 14.2, "pressure": 1013, "humidity": 65},
            "clouds": {"all": 75},
            "visibility": 10000,
            "wind": {"speed": 3.5, "deg": 180}
        }
        result = preprocess_weather_json(api_response, model_type='historical')
        
        # Vérifier les types numériques
        float_features = ['temp', 'feels_like', 'wind_speed', 'hour_sin', 'hour_cos']
        int_features = ['pressure', 'humidity', 'hour', 'month', 'weekday']
        
        for feature in float_features + int_features:
            assert pd.api.types.is_numeric_dtype(result[feature])


# ============================================================================
# CONFIGURATION PYTEST
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
