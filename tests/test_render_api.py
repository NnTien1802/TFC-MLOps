"""Quick checks for the Render status API; no trained predictor is implied."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_info_makes_no_inference_claim():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['inference_available'] is False
