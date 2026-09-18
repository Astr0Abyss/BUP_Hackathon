from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_exact_contract() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.content == b'{"status":"ok"}'

