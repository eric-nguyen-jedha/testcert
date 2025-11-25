#✅ Test 2 — fetch_weather_data

import json
import os
from unittest.mock import patch, MagicMock, mock_open
import pytest
from dags.weather_utils import fetch_weather_data  # ✅ Import propre depuis le module utilitaire


@patch("dags.weather_utils.requests.get")
@patch("dags.weather_utils.Variable.get")
@patch("builtins.open", new_callable=mock_open)
def test_fetch_weather_data(mock_file, mock_var, mock_get):
    """Test de la fonction fetch_weather_data"""
    
    # Mock de la Variable Airflow
    mock_var.return_value = "FAKE_API_KEY"
    
    # Mock de la réponse API
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "weather": [{"main": "Clouds", "description": "few clouds"}],
        "main": {"temp": 10, "feels_like": 9, "pressure": 1013, "humidity": 75},
        "dt": 1700000000,
        "clouds": {"all": 20},
        "wind": {"speed": 3.0, "deg": 200},
    }
    mock_get.return_value = fake_resp
    
    # Mock du context Airflow avec task instance
    mock_ti = MagicMock()
    context = {"ti": mock_ti}
    
    # Exécution de la fonction
    fetch_weather_data(**context)
    
    # Vérifications
    # 1. L'API a été appelée avec les bons paramètres
    mock_get.assert_called_once()
    call_url = mock_get.call_args[0][0]
    assert "api.openweathermap.org" in call_url
    assert "FAKE_API_KEY" in call_url
    
    # 2. Le fichier a été ouvert en écriture
    mock_file.assert_called_once()
    file_path = mock_file.call_args[0][0]
    assert file_path.startswith("/tmp/")
    assert file_path.endswith("_weather.json")
    assert mock_file.call_args[0][1] == "w"
    
    # 3. XCom push a été appelé avec le bon chemin
    mock_ti.xcom_push.assert_called_once()
    args, kwargs = mock_ti.xcom_push.call_args
    assert kwargs["key"] == "local_json_path"
    assert kwargs["value"].startswith("/tmp/")
    assert kwargs["value"].endswith("_weather.json")
    
    # 4. json.dump a été appelé (via le mock de open)
    handle = mock_file()
    handle.write.assert_called()


# Test avec erreur API
@patch("dags.weather_utils.requests.get")
@patch("dags.weather_utils.Variable.get")
def test_fetch_weather_data_api_error(mock_var, mock_get):
    """Test de gestion d'erreur API"""
    
    mock_var.return_value = "FAKE_API_KEY"
    
    # Mock d'une réponse d'erreur
    fake_resp = MagicMock(status_code=401)
    fake_resp.text = "Unauthorized"
    mock_get.return_value = fake_resp
    
    mock_ti = MagicMock()
    context = {"ti": mock_ti}
    
    # Doit lever une ValueError
    with pytest.raises(ValueError, match="Erreur API : 401"):
        fetch_weather_data(**context)
