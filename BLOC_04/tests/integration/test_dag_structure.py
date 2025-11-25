# tests/integration/test_dag_structure.py

import os
import sys
from unittest.mock import patch, MagicMock

# 🔑 Mock du plugin custom pour éviter toute dépendance
sys.modules["s3_to_postgres"] = MagicMock()

def test_dag_loads_correctly():
    from airflow.models import DagBag

    # ✅ Charge le DOSSIER dags/, pas un fichier
    dags_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dags")
    dags_dir = os.path.abspath(dags_dir)

    with patch.dict(os.environ, {"AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS": "False"}):
        dag_bag = DagBag(dag_folder=dags_dir, include_examples=False)

    # 🔍 Affiche les erreurs d'import (débogage)
    assert len(dag_bag.import_errors) == 0, f"Import errors: {dag_bag.import_errors}"

    dag = dag_bag.get_dag("etl_weather_dag")
    assert dag is not None, "DAG etl_weather_dag not found"

    expected_tasks = {
        "fetch_weather_data",
        "transform_and_append_weather_data",
        "create_weather_table",
        "transfer_weather_data_to_postgres",
    }
    actual_tasks = set(dag.task_dict.keys())
    assert actual_tasks == expected_tasks


def test_dag_dependencies():
    from airflow.models import DagBag

    dags_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dags")
    dags_dir = os.path.abspath(dags_dir)

    with patch.dict(os.environ, {"AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS": "False"}):
        dag_bag = DagBag(dag_folder=dags_dir, include_examples=False)

    assert len(dag_bag.import_errors) == 0, f"Import errors: {dag_bag.import_errors}"

    dag = dag_bag.get_dag("etl_weather_dag")
    assert dag is not None

    expected_deps = {
        "fetch_weather_data": ["transform_and_append_weather_data"],
        "transform_and_append_weather_data": ["create_weather_table"],
        "create_weather_table": ["transfer_weather_data_to_postgres"],
    }

    for upstream, downstreams in expected_deps.items():
        task = dag.get_task(upstream)
        assert set(task.downstream_task_ids) == set(downstreams)
