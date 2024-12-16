import pytest
import pandas as pd
from unittest.mock import patch
import joblib
import os

from pred import get_preds

@pytest.fixture
def synthetic_model(tmp_path):
    """
    Create a synthetic scikit-learn pipeline or model artifact and save it
    so we can load it in the test. 
    """
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LinearRegression

    model_path = tmp_path / "test_model.pkl"

    # Example pipeline
    cat_feats = ["team"]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(), cat_feats),
    ], remainder='passthrough')

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Fit a trivial model on dummy data
    X = pd.DataFrame({
        "team": ["Ireland", "England"],
        "innings": [1, 1],
        "remaining_overs": [10, 9]
    })
    y = [6, 4]  # runs_this_over

    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)

    return model_path


@pytest.fixture
def synthetic_csv(tmp_path):
    """
    Create a synthetic CSV resembling 'delivery.csv' with columns:
    'team', 'innings', 'remaining_overs', etc.
    We don't strictly need match_id or other columns if get_preds won't use them.
    """
    csv_path = tmp_path / "inference_data.csv"
    df = pd.DataFrame({
        "team": ["Ireland", "Ireland", "England", "Ireland"],
        "innings": [1, 1, 1, 2],
        "remaining_overs": [10, 10, 9, 15]  # just some random values
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_get_preds_basic(synthetic_model, synthetic_csv, capsys):
    """
    Test a basic scenario where we load a model pipeline,
    filter the CSV by team='Ireland' and overs range,
    and check the printed output.
    """
    get_preds(
        model_path=str(synthetic_model),
        csv_path=str(synthetic_csv),
        team="Ireland",
        start_over=40,
        end_over=45
    )
    captured = capsys.readouterr()
    print(captured.out)
    # Confirm the function prints a dataframe with 'pred_runs_in_over'
    assert "pred_runs_in_over" in captured.out, "Expected predictions in the printed dataframe."


def test_get_preds_different_range(synthetic_model, synthetic_csv, capsys):
    """
    Test a different overs range. 
    Possibly filters out or includes different rows.
    """
    get_preds(
        model_path=str(synthetic_model),
        csv_path=str(synthetic_csv),
        team="Ireland",
        start_over=30,
        end_over=40
    )
    captured = capsys.readouterr()
    # Because Ireland has some 'remaining_overs'=15 in the CSV, the overs=35
    # should appear if 35 is in range(11..20). 
    # But let's just check that something was printed.
    assert "pred_runs_in_over" in captured.out, "Expected predictions in the printed dataframe for overs 11..20."


def test_get_preds_unknown_team(synthetic_model, synthetic_csv, capsys):
    """
    If the user passes a team that doesn't exist in the CSV, 
    we might get an empty DataFrame in the output. 
    Check if that scenario is handled or prints an empty DF.
    """
    get_preds(
        model_path=str(synthetic_model),
        csv_path=str(synthetic_csv),
        team="MarsXI",
        start_over=1,
        end_over=10
    )
    captured = capsys.readouterr()
    # The printed dataframe might have 0 rows if the team doesn't exist.
    assert "No data available for predictions. Check your team name and overs range." in captured.out or "NaN" in captured.out, "Expected empty or no predictions for unknown team."


def test_get_preds_missing_model(synthetic_csv):
    """
    If the model file doesn't exist or path is wrong, joblib.load will raise an error.
    """
    with pytest.raises(FileNotFoundError):
        get_preds(
            model_path="non_existent_model.pkl",
            csv_path=str(synthetic_csv),
            team="Ireland"
        )


def test_get_preds_missing_columns(synthetic_model, tmp_path, capsys):
    """
    If the CSV file is missing expected columns (e.g. 'remaining_overs'),
    we should see an error from the code or from the pipeline.
    """
    broken_csv = tmp_path / "broken_inference.csv"
    df = pd.DataFrame({
        "team": ["Ireland", "England"],
        "innings": [1, 1]
        # missing 'remaining_overs'
    })
    df.to_csv(broken_csv, index=False)

    with pytest.raises(KeyError):
        get_preds(
            model_path=str(synthetic_model),
            csv_path=str(broken_csv),
            team="Ireland",
            start_over=1,
            end_over=10
        )