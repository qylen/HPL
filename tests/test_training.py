from hpl1.synthetic import generate_synthetic_dataset
from hpl1.training import train_and_evaluate


def test_training_pipeline_smoke(tmp_path):
    data = tmp_path / "features.csv"
    artifact = tmp_path / "model.joblib"
    predictions = tmp_path / "preds.csv"
    generate_synthetic_dataset(data, rows=1000, seed=4)
    report = train_and_evaluate(data, "ridge", artifact, predictions)
    assert artifact.exists()
    assert predictions.exists()
    assert report["test_metrics"]["rows"] > 0
