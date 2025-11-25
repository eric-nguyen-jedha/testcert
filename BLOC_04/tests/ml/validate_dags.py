#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de validation des DAGs Airflow — version stable pour CI/CD
Vérifie que les DAGs ML peuvent être importés sans erreur
"""
import sys
import os
from pathlib import Path
from unittest.mock import patch

# =============================================================================
# 🛡️ DÉSACTIVER LA BASE DE DONNÉES AIRFLOW
# =============================================================================
# Forcer Airflow à ne pas utiliser de base de données
os.environ['AIRFLOW__CORE__UNIT_TEST_MODE'] = 'True'
os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'] = 'sqlite:///:memory:'
os.environ['AIRFLOW__SECRETS__BACKEND'] = ''

# =============================================================================
# Imports nécessaires
# =============================================================================
import importlib.util


def validate_dag_file(dag_path):
    """Valider un fichier DAG individuel"""
    print(f"📄 Validation de {dag_path.name}...")
    
    try:
        # Patcher Variable.get AVANT d'importer le module
        with patch('airflow.models.Variable.get') as mock_var_get:
            # Configurer le mock pour retourner des valeurs appropriées
            def fake_get(key, default_var=None):
                fake_vars = {
                    "BUCKET": "test-bucket",
                    "AWS_ACCESS_KEY_ID": "fake_key",
                    "AWS_SECRET_ACCESS_KEY": "fake_secret",
                    "AWS_DEFAULT_REGION": "eu-west-3",
                    "OPEN_WEATHER_API_KEY": "fake_api_key",
                    "mlflow_uri": "http://localhost:8081",
                    "ARTIFACT_STORE_URI": "s3://test-bucket/mlflow"
                }
                result = fake_vars.get(key)
                if result is not None:
                    return str(result)
                if default_var is not None:
                    return str(default_var)
                return f"mock_{key}"
            
            mock_var_get.side_effect = fake_get
            
            # Charger le module
            module_name = dag_path.stem
            spec = importlib.util.spec_from_file_location(module_name, str(dag_path))
            
            if spec is None:
                print("  ❌ Impossible de créer la spec du module")
                return False
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # Exécuter le module avec le mock actif
            spec.loader.exec_module(module)
            
            # Chercher les DAGs dans le module
            dags_found = []
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    # Vérifier si c'est un DAG (vrai ou mock)
                    if hasattr(attr, 'dag_id') and hasattr(attr, 'default_args'):
                        # Extraire dag_id et forcer en string
                        try:
                            dag_id_value = attr.dag_id
                            # Si c'est un MagicMock, convertir en string
                            dag_id_str = str(dag_id_value)
                            # Éviter les strings comme "<MagicMock...>"
                            if not dag_id_str.startswith('<') and not dag_id_str.startswith('Mock'):
                                dags_found.append(dag_id_str)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            if not dags_found:
                print("  ⚠️  Aucun DAG trouvé dans le module")
                return False
            
            # S'assurer que tous les éléments sont des strings
            dags_found_str = [str(d) for d in dags_found]
            print(f"  ✅ DAGs trouvés: {', '.join(dags_found_str)}")
            return True
        
    except SyntaxError as e:
        print(f"  ❌ Erreur de syntaxe: {e}")
        if hasattr(e, 'lineno') and hasattr(e, 'text'):
            print(f"     Ligne {e.lineno}: {e.text}")
        return False
    except ImportError as e:
        print(f"  ❌ Erreur d'import: {e}")
        import traceback
        print("     Détails:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ❌ Erreur lors de la validation: {e}")
        print("     Détails de l'erreur:")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Valider tous les DAGs ML"""
    
    # Déterminer le chemin des DAGs
    script_dir = Path(__file__).parent
    
    # Essayer plusieurs chemins possibles
    possible_paths = [
        script_dir.parent.parent / "dags_ml",
        script_dir.parent.parent / "dags",
        Path.cwd() / "dags_ml",
    ]
    
    dags_dir = None
    for path in possible_paths:
        if path.exists():
            dags_dir = path
            break
    
    if dags_dir is None:
        print("❌ Impossible de trouver le répertoire dags_ml/")
        print(f"   Chemins testés: {[str(p) for p in possible_paths]}")
        sys.exit(1)
    
    print("=" * 60)
    print("🔍 VALIDATION DES DAGS AIRFLOW ML")
    print("=" * 60)
    print(f"📂 Répertoire: {dags_dir}")
    print()
    
    # Liste des DAGs à valider
    dags_to_validate = [
        "realtime_prediction_forecast.py",
        "paris_meteo_ml_pipeline.py"
    ]
    
    results = {}
    for filename in dags_to_validate:
        dag_path = dags_dir / filename
        
        if not dag_path.exists():
            print(f"❌ Fichier non trouvé: {filename}")
            print(f"   Chemin complet: {dag_path}")
            results[filename] = False
            continue
        
        results[filename] = validate_dag_file(dag_path)
        print()
    
    # Résumé
    print("=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total: {total}")
    print(f"✅ Réussis: {passed}")
    print(f"❌ Échoués: {failed}")
    print()
    
    if failed > 0:
        print("❌ La validation a échoué!")
        sys.exit(1)
    else:
        print("✅ Tous les DAGs sont valides!")
        sys.exit(0)


if __name__ == "__main__":
    main()
