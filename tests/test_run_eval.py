# tests/test_run_eval.py

import mlflow
import pytest

def test_relevancia_minima():
    client = mlflow.tracking.MlflowClient()
    experiments = [e for e in client.search_experiments() if e.name.startswith("eval_")]
    
    assert experiments, "No hay experimentos con nombre 'eval_'"

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert runs, f"No hay ejecuciones en el experimento {exp.name}"

        scores = [r.data.metrics.get("lc_is_correct", 0) for r in runs if "lc_is_correct" in r.data.metrics]
        if scores:
            promedio = sum(scores) / len(scores)
            print(f"Precisión promedio en {exp.name}: {promedio:.2f}")
            assert promedio >= 0.8, f"Precisión insuficiente en {exp.name}: {promedio:.2f}"
        else:
            pytest.fail(f"No se encontraron métricas 'lc_is_correct' en {exp.name}")
