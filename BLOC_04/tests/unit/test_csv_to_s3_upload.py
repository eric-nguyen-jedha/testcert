# tests/unit/test_csv_to_s3_upload.py
import json
import os
from unittest.mock import patch, MagicMock, mock_open
import pytest
from dags.weather_utils import transform_and_append_weather_data


@patch("dags.weather_utils.S3Hook")
@patch("dags.weather_utils.Variable.get")
@patch("dags.weather_utils.setup_aws_environment")
@patch("dags.weather_utils.os.path.exists")
@patch("builtins.open", new_callable=mock_open)
@patch("dags.weather_utils.pd.read_csv")
@patch("dags.weather_utils.pd.DataFrame.to_csv")
def test_csv_uploaded_to_s3(
    mock_to_csv, mock_read_csv, mock_file, mock_exists, mock_setup, mock_var, mock_s3_class
):
    """Test que le CSV est bien uploadé sur S3 après transformation."""
    
    # Configuration
    mock_var.return_value = "TEST_BUCKET"
    mock_s3_instance = MagicMock()
    mock_s3_class.return_value = mock_s3_instance
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = "/tmp/test_weather.json"
    context = {"ti": mock_ti}

    mock_exists.return_value = True
    mock_file.return_value.read.return_value = json.dumps({
    "dt": 1700000000,
    "main": {
        "temp": 20,
        "feels_like": 19,
        "pressure": 1000,
        "humidity": 50
    },
    "clouds": {"all": 10},
    "wind": {"speed": 3.5, "deg": 180},
    "weather": [{"main": "Clear", "description": "sunny"}],
    "visibility": 10000,
    "rain": {"1h": 0.5}
})
    
    # Simuler CSV existant (pour éviter le premier run)
    existing_df = MagicMock()
    mock_read_csv.return_value = existing_df
    
    # Exécution
    transform_and_append_weather_data(**context)
    
    # Vérifications
    mock_s3_instance.load_file.assert_called_once()
    call_kwargs = mock_s3_instance.load_file.call_args[1]
    assert call_kwargs["bucket_name"] == "TEST_BUCKET"
    assert call_kwargs["key"] == "weather_paris_fect.csv"
    assert call_kwargs["replace"] is True
    assert call_kwargs["filename"].startswith("/tmp/")
    assert call_kwargs["filename"].endswith("weather_paris_fect.csv")
    
    # Vérifie que XCom push a été appelé
    mock_ti.xcom_push.assert_called_once_with(
        key="weather_csv_key",
        value="weather_paris_fect.csv"
    )
