"""
Unit tests for MLflow experiment tracking integration (Phase 10).
"""

import os
import mlflow
import pytest
from train import flatten_config, load_config


def test_flatten_config():
    """Test that nested config dictionary is properly flattened for MLflow params."""
    nested_cfg = {
        "model": {"hidden_dim": 64, "num_layers": 1},
        "training": {"epochs": 20, "learning_rate": 0.001},
        "simple_key": "simple_val",
    }
    flat = flatten_config(nested_cfg)
    assert flat["model.hidden_dim"] == 64
    assert flat["model.num_layers"] == 1
    assert flat["training.epochs"] == 20
    assert flat["training.learning_rate"] == 0.001
    assert flat["simple_key"] == "simple_val"


def test_mlflow_tracking_isolated_run(tmp_path):
    """Test isolated MLflow run logging parameters, metrics, and artifacts."""
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:///{str(tracking_dir).replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = "test-anomaly-experiment"
    mlflow.set_experiment(exp_name)

    artifact_file = tmp_path / "dummy_artifact.txt"
    artifact_file.write_text("dummy artifact content", encoding="utf-8")

    with mlflow.start_run(run_name="test-run") as run:
        run_id = run.info.run_id

        # Log params
        mlflow.log_params({"model.hidden_dim": 64, "training.lr": 0.001})

        # Log metrics across steps
        mlflow.log_metrics({"train_loss": 0.05, "val_loss": 0.04}, step=1)
        mlflow.log_metrics({"train_loss": 0.02, "val_loss": 0.01}, step=2)

        # Log final metrics
        mlflow.log_metrics({"best_val_loss": 0.01, "anomaly_threshold": 0.015})

        # Log artifact
        mlflow.log_artifact(str(artifact_file))

    # Retrieve and verify run details
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    retrieved_run = client.get_run(run_id)

    assert retrieved_run.data.params["model.hidden_dim"] == "64"
    assert retrieved_run.data.params["training.lr"] == "0.001"

    assert pytest.approx(retrieved_run.data.metrics["best_val_loss"]) == 0.01
    assert pytest.approx(retrieved_run.data.metrics["anomaly_threshold"]) == 0.015

    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]
    assert "dummy_artifact.txt" in artifact_names
