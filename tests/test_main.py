from fastapi.testclient import TestClient
from app.main import app

client= TestClient(app)

def test_replica_lag_routes_to_database():
    response = client.post(
        "/predict",
        json={"text":"replica lag is increasing"},
    )

    assert response.status_code == 200
    assert response.json() == {"category":"database"}

def test_failed_release_routes_to_deployment():
    response = client.post(
        "/predict",
        json={"text": "deployment failed during release"},
    )

    assert response.status_code == 200
    assert response.json() == {"category": "deployment"}