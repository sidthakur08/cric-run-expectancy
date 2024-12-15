import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import joblib

# We'll import the train_model function directly from model.py
from model import train_model

@pytest.fixture
def synthetic_delivery_csv(tmp_path):
    """
    Creates a synthetic version of the 'delivery.csv' file in a temporary directory.
    The columns replicate the expected schema from the real data:
      matchid, team, innings, remaining_overs, runs_on_ball
    """
    csv_path = tmp_path / "delivery.csv"
    
    # Create some synthetic data
    # We'll have multiple matchid values, multiple teams, innings, overs, etc.
    # so that the sum of runs per over is meaningful.
    data = {
        "matchid": [1, 1, 1, 2, 2, 3],
        "team": ["Ireland", "Ireland", "England", "Ireland", "England", "England"],
        "innings": [1, 1, 1, 1, 2, 1],
        "remaining_overs": [10, 9, 10, 10, 10, 9],
        "runs_on_ball": [4, 6, 3, 2, 1, 4]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

def test_train_model_runs_successfully(synthetic_delivery_csv, tmp_path):
    """
    Test that the train_model function runs without error on synthetic data
    and produces a model artifact with the correct filename.
    """
    # We expect the model to be saved at 'models/model.pkl' by default.
    # But let's patch the path so we don't write to the real location.
    model_path = tmp_path / "test_model.pkl"
    
    # We'll patch 'pd.read_csv' to read from our synthetic_delivery_csv fixture
    # and patch joblib.dump so it writes to our tmp_path.
    df = pd.read_csv(synthetic_delivery_csv)
    
    with patch("model.pd.read_csv", return_value=df): #patch("model.joblib.dump") as mock_dump, \
        
        # Call the training function
        train_model(input_path=synthetic_delivery_csv, output_path=model_path)
        
        assert os.path.exists(model_path), "Model artifact not saved properly in test environment."


def test_train_model_produces_reasonable_metrics(synthetic_delivery_csv, capsys):
    """
    Test that the printed MAE and RMSE are computed and printed.
    We won't assert a strict threshold (since it's random data), but we check that
    the function prints MAE/RMSE.
    """
    df = pd.read_csv(synthetic_delivery_csv)

    with patch("model.pd.read_csv", return_value=df):
        
        train_model(input_path=synthetic_delivery_csv)
        captured = capsys.readouterr()  # capture the stdout
        
        # We expect something like "MAE: X, RMSE: Y"
        assert "MAE:" in captured.out, "Expected MAE output not found."
        assert "RMSE:" in captured.out, "Expected RMSE output not found."


def test_train_model_handles_missing_columns(tmp_path):
    """
    If the CSV is missing expected columns, the code might raise an error.
    """
    # Create a CSV missing 'runs_on_ball' column
    csv_path = tmp_path / "delivery.csv"
    df = pd.DataFrame({
        "matchid": [1, 2],
        "team": ["Ireland", "England"],
        "innings": [1, 1],
        "remaining_overs": [10, 9],
        # Missing 'runs_on_ball'
    })
    df.to_csv(csv_path, index=False)
    
    df = pd.read_csv(csv_path)
    with patch("model.pd.read_csv", return_value = df):
        
        with pytest.raises(KeyError) as exc_info:
            train_model(input_path=csv_path)
        assert "runs_on_ball" in str(exc_info.value), "Expected KeyError for missing runs_on_ball column"


def test_train_model_empty_data(tmp_path):
    """
    If the CSV is empty or has no rows, we expect an error or a poor result.
    """
    csv_path = tmp_path / "delivery.csv"
    pd.DataFrame(columns=["matchid", "team", "innings", "remaining_overs", "runs_on_ball"]).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    
    with patch("model.pd.read_csv", return_value=df):
        
        with pytest.raises(ValueError):
            train_model(input_path=csv_path)