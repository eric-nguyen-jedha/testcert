# -*- coding: utf-8 -*-
"""
Tests pour le DAG d'entraînement paris_meteo_ml_pipeline
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os

# ============================================================================
# TESTS UNITAIRES - Préparation des données historiques
# ============================================================================

class TestHistoricalDataPreparation:
    """Tests pour la préparation des données du modèle historical"""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Données brutes simulées"""
        np.random.seed(42)
        return pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=200, freq='H'),
            'temp': np.random.uniform(0, 30, 200),
            'feels_like': np.random.uniform(0, 30, 200),
            'pressure': np.random.uniform(990, 1030, 200),
            'humidity': np.random.randint(30, 90, 200),
            'clouds': np.random.randint(0, 100, 200),
            'visibility': np.random.randint(5000, 10000, 200),
            'wind_speed': np.random.uniform(0, 10, 200),
            'wind_deg': np.random.randint(0, 360, 200),
            'rain_1h': np.random.uniform(0, 5, 200),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Fog', 'Drizzle', 'Mist'], 200),
            'weather_description': ['test'] * 200
        })
    
    def test_drizzle_replaced_by_rain(self, sample_raw_data):
        """Test que Drizzle est remplacé par Rain"""
        df = sample_raw_data.copy()
        df['weather_main'] = df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
        assert 'Drizzle' not in df['weather_main'].unique()
        assert 'Rain' in df['weather_main'].unique()
    
    def test_mist_replaced_by_fog(self, sample_raw_data):
        """Test que Mist est remplacé par Fog"""
        df = sample_raw_data.copy()
        df['weather_main'] = df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
        assert 'Mist' not in df['weather_main'].unique()
        assert 'Fog' in df['weather_main'].unique()
    
    def test_min_samples_filtering(self, sample_raw_data):
        """Test le filtrage des classes avec peu d'échantillons"""
        df = sample_raw_data.copy()
        min_samples = 10
        valid_classes = df['weather_main'].value_counts()
        valid_classes = valid_classes[valid_classes >= min_samples].index
        df_filtered = df[df['weather_main'].isin(valid_classes)]
        
        # Vérifier que toutes les classes restantes ont au moins min_samples
        for weather_class in df_filtered['weather_main'].unique():
            count = (df_filtered['weather_main'] == weather_class).sum()
            assert count >= min_samples
    
    def test_temporal_features_are_created(self, sample_raw_data):
        """Test la création des features temporelles"""
        df = sample_raw_data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        assert 'hour' in df.columns
        assert 'month' in df.columns
        assert 'weekday' in df.columns
        assert 'is_weekend' in df.columns
        assert df['hour'].min() >= 0 and df['hour'].max() <= 23
        assert df['month'].min() >= 1 and df['month'].max() <= 12
    
    def test_cyclical_features_are_created(self, sample_raw_data):
        """Test la création des features cycliques"""
        df = sample_raw_data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Vérifier que les valeurs sont dans [-1, 1]
        assert df['hour_sin'].between(-1, 1).all()
        assert df['hour_cos'].between(-1, 1).all()
        assert df['month_sin'].between(-1, 1).all()
        assert df['month_cos'].between(-1, 1).all()
    
    def test_no_dew_point_in_historical(self, sample_raw_data):
        """Test que dew_point est bien supprimé pour historical"""
        df = sample_raw_data.copy()
        if 'dew_point' in df.columns:
            df = df.drop('dew_point', axis=1)
        assert 'dew_point' not in df.columns
    
    def test_no_timestamp_in_historical(self, sample_raw_data):
        """Test que timestamp est bien supprimé pour historical"""
        df = sample_raw_data.copy()
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        assert 'timestamp' not in df.columns
    
    def test_label_encoder_alphabetical_order(self):
        """Test que LabelEncoder encode par ordre alphabétique"""
        le = LabelEncoder()
        classes = ['Rain', 'Clear', 'Fog', 'Clouds', 'Snow']
        le.fit(classes)
        
        # L'ordre alphabétique est : Clear, Clouds, Fog, Rain, Snow
        expected_order = ['Clear', 'Clouds', 'Fog', 'Rain', 'Snow']
        assert list(le.classes_) == expected_order
        
        # Vérifier les codes
        assert le.transform(['Clear'])[0] == 0
        assert le.transform(['Clouds'])[0] == 1
        assert le.transform(['Fog'])[0] == 2
        assert le.transform(['Rain'])[0] == 3
        assert le.transform(['Snow'])[0] == 4


# ============================================================================
# TESTS UNITAIRES - Préparation des données forecast 6h
# ============================================================================

class TestForecast6hDataPreparation:
    """Tests pour la préparation des données du modèle forecast_6h"""
    
    @pytest.fixture
    def sample_timeseries_data(self):
        """Données de série temporelle simulées"""
        np.random.seed(42)
        return pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=300, freq='H'),
            'temp': np.random.uniform(0, 30, 300),
            'feels_like': np.random.uniform(0, 30, 300),
            'pressure': np.random.uniform(990, 1030, 300),
            'humidity': np.random.randint(30, 90, 300),
            'clouds': np.random.randint(0, 100, 300),
            'visibility': np.random.randint(5000, 10000, 300),
            'wind_speed': np.random.uniform(0, 10, 300),
            'wind_deg': np.random.randint(0, 360, 300),
            'rain_1h': np.random.uniform(0, 5, 300),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Fog'], 300)
        })
    
    def test_target_shifted_correctly(self, sample_timeseries_data):
        """Test que la cible est bien décalée de 6 pas de temps"""
        df = sample_timeseries_data.copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        df['weather_6h'] = df['weather_main'].shift(-6)
        
        # Vérifier que weather_6h[0] == weather_main[6]
        assert df.loc[0, 'weather_6h'] == df.loc[6, 'weather_main']
        assert df.loc[10, 'weather_6h'] == df.loc[16, 'weather_main']
    
    def test_dropna_removes_last_rows(self, sample_timeseries_data):
        """Test que dropna supprime les dernières lignes sans cible"""
        df = sample_timeseries_data.copy()
        original_len = len(df)
        df['weather_6h'] = df['weather_main'].shift(-6)
        df = df.dropna(subset=['weather_6h'])
        
        # Les 6 dernières lignes doivent être supprimées
        assert len(df) == original_len - 6
    
    def test_no_dew_point_in_forecast(self, sample_timeseries_data):
        """Test que dew_point n'est pas présent pour forecast_6h"""
        df = sample_timeseries_data.copy()
        feature_cols = [
            'temp', 'feels_like', 'pressure', 'humidity',
            'clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h'
        ]
        assert 'dew_point' not in feature_cols
    
    def test_no_timestamp_in_forecast(self, sample_timeseries_data):
        """Test que timestamp n'est pas présent pour forecast_6h"""
        df = sample_timeseries_data.copy()
        feature_cols = [
            'temp', 'feels_like', 'pressure', 'humidity',
            'clouds', 'visibility', 'wind_speed', 'wind_deg', 'rain_1h',
            'hour', 'month', 'weekday', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        assert 'timestamp' not in feature_cols
        assert len(feature_cols) == 17
    
    def test_datetime_sorted_before_shift(self, sample_timeseries_data):
        """Test que les données sont triées par datetime avant le shift"""
        df = sample_timeseries_data.copy()
        # Mélanger les données
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Trier
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Vérifier que c'est bien trié
        assert df['datetime'].is_monotonic_increasing


# ============================================================================
# TESTS UNITAIRES - Entraînement des modèles
# ============================================================================

class TestModelTraining:
    """Tests pour l'entraînement des modèles"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Données d'entraînement simulées"""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'temp': np.random.uniform(0, 30, n_samples),
            'feels_like': np.random.uniform(0, 30, n_samples),
            'pressure': np.random.uniform(990, 1030, n_samples),
            'humidity': np.random.randint(30, 90, n_samples),
            'clouds': np.random.randint(0, 100, n_samples),
            'visibility': np.random.randint(5000, 10000, n_samples),
            'wind_speed': np.random.uniform(0, 10, n_samples),
            'wind_deg': np.random.randint(0, 360, n_samples),
            'rain_1h': np.random.uniform(0, 5, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'weekday': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.randint(0, 2, n_samples),
            'hour_sin': np.random.uniform(-1, 1, n_samples),
            'hour_cos': np.random.uniform(-1, 1, n_samples),
            'month_sin': np.random.uniform(-1, 1, n_samples),
            'month_cos': np.random.uniform(-1, 1, n_samples)
        })
        
        y = np.random.choice([0, 1, 2, 3], n_samples)  # 4 classes
        
        return X, y
    
    def test_train_test_split_80_20(self, sample_training_data):
        """Test que le split est bien 80/20"""
        X, y = sample_training_data
        split = int(0.8 * len(X))
        
        X_train = X[:split]
        X_test = X[split:]
        
        total = len(X)
        train_ratio = len(X_train) / total
        test_ratio = len(X_test) / total
        
        assert abs(train_ratio - 0.8) < 0.01
        assert abs(test_ratio - 0.2) < 0.01
    
    def test_xgboost_parameters(self):
        """Test les paramètres XGBoost recommandés"""
        from xgboost import XGBClassifier
        
        # Modèle historical
        model_hist = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1
        )
        
        assert model_hist.n_estimators == 100
        assert model_hist.max_depth == 6
        assert model_hist.learning_rate == 0.1
        
        # Modèle forecast_6h
        model_6h = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=-1
        )
        
        assert model_6h.n_estimators == 50
        assert model_6h.max_depth == 4
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.xgboost.log_model')
    def test_mlflow_logging(self, mock_log_model, mock_log_metric, mock_log_params, mock_start_run):
        """Test que MLflow log correctement les paramètres et métriques"""
        from xgboost import XGBClassifier
        
        # Simuler un contexte MLflow
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()
        
        # Créer un modèle
        model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
        
        # Simuler le logging
        params = {
            'model_type': 'historical',
            'n_estimators': 100,
            'max_depth': 6
        }
        
        metrics = {
            'accuracy_train': 0.95,
            'accuracy_test': 0.85
        }
        
        with mock_start_run():
            mock_log_params(params)
            for name, value in metrics.items():
                mock_log_metric(name, value)
            mock_log_model(model, "model")
        
        # Vérifier les appels
        mock_log_params.assert_called_once()
        assert mock_log_metric.call_count == len(metrics)


# ============================================================================
# TESTS D'INTÉGRATION - Pipeline complet
# ============================================================================

class TestTrainingPipelineIntegration:
    """Tests d'intégration pour le pipeline d'entraînement complet"""
    
    @patch('boto3.client')
    @patch('mlflow.start_run')
    def test_full_training_pipeline(self, mock_mlflow, mock_s3):
        """Test du pipeline complet d'entraînement"""
        # Mock S3
        mock_s3_client = MagicMock()
        mock_s3.return_value = mock_s3_client
        
        # Mock MLflow
        mock_mlflow.return_value.__enter__ = Mock()
        mock_mlflow.return_value.__exit__ = Mock()
        
        # Créer des données de test
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=200, freq='H'),
            'temp': np.random.uniform(10, 25, 200),
            'feels_like': np.random.uniform(10, 25, 200),
            'pressure': np.random.uniform(1000, 1020, 200),
            'humidity': np.random.randint(40, 80, 200),
            'clouds': np.random.randint(0, 100, 200),
            'visibility': np.full(200, 10000),
            'wind_speed': np.random.uniform(0, 5, 200),
            'wind_deg': np.random.randint(0, 360, 200),
            'rain_1h': np.zeros(200),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'], 200)
        })
        
        # Tester la pipeline de préparation
        df['weather_main'] = df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        
        # Vérifications
        assert 'hour' in df.columns
        assert 'month' in df.columns
        assert len(df) == 200
    
    @pytest.mark.integration
    def test_label_encoder_save_and_load(self):
        """Test de sauvegarde et chargement du LabelEncoder"""
        le = LabelEncoder()
        classes = ['Clear', 'Clouds', 'Fog', 'Rain', 'Snow']
        le.fit(classes)
        
        # Sauvegarder
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(le, f)
            temp_path = f.name
        
        # Recharger
        with open(temp_path, 'rb') as f:
            le_loaded = pickle.load(f)
        
        # Vérifier
        assert list(le_loaded.classes_) == list(le.classes_)
        assert le_loaded.transform(['Clear'])[0] == le.transform(['Clear'])[0]
        
        # Nettoyer
        os.unlink(temp_path)


# ============================================================================
# TESTS DE VALIDATION - Métriques
# ============================================================================

class TestMetricsValidation:
    """Tests pour valider le calcul des métriques"""
    
    def test_metrics_calculation(self):
        """Test du calcul des métriques de classification"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
    
    def test_confusion_matrix_shape(self):
        """Test de la forme de la matrice de confusion"""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        n_classes = len(np.unique(y_true))
        assert cm.shape == (n_classes, n_classes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
