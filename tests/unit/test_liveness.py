from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_liveness():
    response = client.get("/liveness")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
