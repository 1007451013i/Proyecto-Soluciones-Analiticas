import pandas as pd
from pathlib import Path
from model.predict import make_prediction


def test_make_prediction_runs_on_test_csv():
    csv_path = Path(__file__).resolve().parents[1] / "model" / "datasets" / "heart_test.csv"
    df = pd.read_csv(csv_path)
    result = make_prediction(input_data=df)
    assert "predictions" in result
    assert len(result["predictions"]) == len(df)