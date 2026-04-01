import pytest
from httpx import ASGITransport, AsyncClient

from app.api import traces
from app.database import init_db
from app.main import app


@pytest.mark.asyncio
async def test_demo_endpoint_returns_full_result():
    await init_db()
    traces._request_log.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/trace/demo", json={"scenario": "hallucination"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario"] == "hallucination"
    assert payload["summary"]["failed"] > 0
    assert payload["eval_results"]
    assert "trace_id" in payload


@pytest.mark.asyncio
async def test_demo_endpoint_rate_limits_by_ip(monkeypatch):
    await init_db()
    traces._request_log.clear()
    monkeypatch.setattr(traces, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(traces, "RATE_LIMIT_WINDOW_SECONDS", 60)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"x-forwarded-for": "203.0.113.10"},
    ) as client:
        first = await client.post("/trace/demo", json={"scenario": "hallucination"})
        second = await client.post("/trace/demo", json={"scenario": "hallucination"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 59
