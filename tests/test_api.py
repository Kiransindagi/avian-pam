import pytest
from fastapi.testclient import TestClient
from src.inference.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_and_version(client):
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "HEALTHY"

    res_version = client.get("/version")
    assert res_version.status_code == 200
    assert "version" in res_version.json()


def test_api_metadata_endpoints(client):
    res_model = client.get("/model_info")
    assert res_model.status_code == 200
    assert "registered_models" in res_model.json()

    res_feat = client.get("/feature_info")
    assert res_feat.status_code == 200
    assert "active_extractors" in res_feat.json()
