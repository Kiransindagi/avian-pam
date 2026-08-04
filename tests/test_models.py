import numpy as np

from src.models import RandomForestModel, list_registered_models


def test_registered_models_exist():
    models = list_registered_models()
    assert "dummy_mean" in models
    assert "linear_regression" in models
    assert "random_forest" in models
    assert "svr" in models
    assert "voting_ensemble" in models


def test_fit_predict_save_load(tmp_path):
    X = np.random.randn(20, 5)
    y = np.random.randint(1, 10, size=20).astype(np.float32)

    model = RandomForestModel(n_estimators=10)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 20
    assert np.all(preds >= 0)

    # Save and Load
    ckpt = tmp_path / "rf_model.joblib"
    model.save(ckpt)
    assert ckpt.exists()

    loaded_model = RandomForestModel.load(ckpt)
    loaded_preds = loaded_model.predict(X)
    assert np.allclose(preds, loaded_preds)
