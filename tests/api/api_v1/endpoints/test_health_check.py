from fastapi.testclient import TestClient

from service.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/docs")
    assert response.status_code == 200
