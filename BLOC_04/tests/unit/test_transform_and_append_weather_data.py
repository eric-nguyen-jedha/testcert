# tests/test_transform_and_append_weather_data.py

#✅ Test 3 — transform_and_append_weather_data

import json
import os
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import pytest
from dags.weather_utils import transform_and_append_weather_data


@patch("dags.weather_utils.S3Hook")
@patch("dags.weather_utils.Variable.get")
@patch("dags.weather_utils.setup_aws_environment")
@patch("dags.weather_utils.os.path.exists")
@patch("builtins.open", new_callable=mock_open)
def test_transform_and_append_weather_data_new_csv(
    mock_file, mock_exists, mock_setup, mock_var, mock_s3_class
):
    """Test de transformation avec création d'un nouveau CSV"""
    
    # Configuration des mocks
    mock_var.return_value = "FAKE_BUCKET"
    
    # Mock du S3Hook
    mock_s3_instance = MagicMock()
    mock_s3_class.return_value = mock_s3_instance
    
    # Simuler que le CSV n'existe pas sur S3 (premier run)
    mock_s3_instance.download_file.side_effect = Exception("CSV not found")
    
    # Mock du context avec XCom pull
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = "/tmp/test_weather.json"
    context = {"ti": mock_ti}
    
    # Préparer le JSON fake
    fake_json = {
        "dt": 1700000000,
        "main": {"temp": 20, "feels_like": 19, "pressure": 1000, "humidity": 50},
        "clouds": {"all": 10},
        "wind": {"speed": 3.5, "deg": 180},
        "weather": [{"main": "Clear", "description": "sunny"}],
    }
    
    # Mock de os.path.exists pour le JSON local
    mock_exists.return_value = True
    
    # Mock de la lecture du JSON
    mock_file.return_value.read.return_value = json.dumps(fake_json)
    
    # Exécution
    transform_and_append_weather_data(**context)
    
    # Vérifications
    mock_setup.assert_called_once()
    mock_var.assert_called_with("BUCKET")
    mock_ti.xcom_pull.assert_called_once_with(
        task_ids="fetch_weather_data",
        key="local_json_path"
    )
    mock_s3_class.assert_called_with(aws_conn_id="aws_default")
    mock_s3_instance.load_file.assert_called_once()
    load_call = mock_s3_instance.load_file.call_args
    assert load_call[1]["key"] == "weather_paris_fect.csv"
    assert load_call[1]["bucket_name"] == "FAKE_BUCKET"
    assert load_call[1]["replace"] is True
    mock_ti.xcom_push.assert_called_once_with(
        key="weather_csv_key",
        value="weather_paris_fect.csv"
    )


@patch("dags.weather_utils.S3Hook")
@patch("dags.weather_utils.Variable.get")
@patch("dags.weather_utils.setup_aws_environment")
@patch("dags.weather_utils.os.path.exists")
@patch("builtins.open", new_callable=mock_open)
@patch("dags.weather_utils.pd.read_csv")
def test_transform_and_append_weather_data_existing_csv(
    mock_read_csv, mock_file, mock_exists, mock_setup, mock_var, mock_s3_class
):
    """Test de transformation avec CSV existant (append)"""
    
    mock_var.return_value = "FAKE_BUCKET"
    mock_s3_instance = MagicMock()
    mock_s3_class.return_value = mock_s3_instance
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = "/tmp/test_weather.json"
    context = {"ti": mock_ti}
    
    fake_json = {
        "dt": 1700000000,
        "main": {"temp": 20, "feels_like": 19, "pressure": 1000, "humidity": 50},
        "clouds": {"all": 10},
        "wind": {"speed": 3.5, "deg": 180},
        "weather": [{"main": "Clear", "description": "sunny"}],
    }
    
    mock_exists.return_value = True
    mock_file.return_value.read.return_value = json.dumps(fake_json)
    
    existing_df = pd.DataFrame([{
        "datetime": "2023-01-01 00:00:00",
        "temp": 15.0,
        "feels_like": 14.0,
        "pressure": 1010.0,
        "humidity": 60.0,
        "clouds": 20.0,
        "visibility": 10000.0,
        "wind_speed": 2.5,
        "wind_deg": 90.0,
        "rain_1h": 0.0,
        "weather_main": "Rain",
        "weather_description": "light rain"
    }])
    mock_read_csv.return_value = existing_df
    mock_s3_instance.download_file.return_value = None
    
    transform_and_append_weather_data(**context)
    
    mock_s3_instance.download_file.assert_called_once()
    mock_read_csv.assert_called_once()
    mock_s3_instance.load_file.assert_called_once()
    mock_ti.xcom_push.assert_called_once_with(
        key="weather_csv_key",
        value="weather_paris_fect.csv"
    )


@patch("dags.weather_utils.setup_aws_environment")
@patch("dags.weather_utils.os.path.exists")
def test_transform_and_append_weather_data_missing_json(mock_exists, mock_setup):
    """Test avec JSON manquant (doit lever une erreur)"""
    
    mock_exists.return_value = False
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = "/tmp/missing.json"
    context = {"ti": mock_ti}
    
    with pytest.raises(ValueError, match="Impossible de récupérer le JSON local"):
        transform_and_append_weather_data(**context)


@patch("dags.weather_utils.S3Hook")
@patch("dags.weather_utils.Variable.get")
@patch("dags.weather_utils.setup_aws_environment")
@patch("dags.weather_utils.os.path.exists")
@patch("builtins.open", new_callable=mock_open)
@patch("dags.weather_utils.pd.read_csv")
def test_transform_and_append_weather_data_duplicate_prevention(
    mock_read_csv, mock_file, mock_exists, mock_setup, mock_var, mock_s3_class
):
    """Test de prévention des doublons"""
    
    mock_var.return_value = "FAKE_BUCKET"
    mock_s3_instance = MagicMock()
    mock_s3_class.return_value = mock_s3_instance
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = "/tmp/test_weather.json"
    context = {"ti": mock_ti}
    
    fake_json = {
        "dt": 1672531200,  # 2023-01-01 00:00:00 UTC
        "main": {"temp": 20, "feels_like": 19, "pressure": 1000, "humidity": 50},
        "clouds": {"all": 10},
        "wind": {"speed": 3.5, "deg": 180},
        "weather": [{"main": "Clear", "description": "sunny"}],
    }
    
    mock_exists.return_value = True
    mock_file.return_value.read.return_value = json.dumps(fake_json)
    
    existing_df = pd.DataFrame([{
        "datetime": "2023-01-01 00:00:00",
        "temp": 15.0,
        "feels_like": 14.0,
        "pressure": 1010.0,
        "humidity": 60.0,
        "clouds": 20.0,
        "visibility": 10000.0,
        "wind_speed": 2.5,
        "wind_deg": 90.0,
        "rain_1h": 0.0,
        "weather_main": "Rain",
        "weather_description": "light rain"
    }])
    mock_read_csv.return_value = existing_df
    mock_s3_instance.download_file.return_value = None
    
    transform_and_append_weather_data(**context)
    
    mock_s3_instance.load_file.assert_called_once()
    mock_ti.xcom_push.assert_called_once()
