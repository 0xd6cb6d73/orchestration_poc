from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

import poc.evaluation.suite.phoenix as phoenix


def test_publish_http_client_uses_long_upload_timeout() -> None:
    with phoenix.publish_http_client() as client:
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 600.0
        assert client.timeout.write == 600.0
        assert client.timeout.pool == 10.0


def test_publish_builds_default_client_through_publish_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[httpx.Client] = []
    factory = phoenix.publish_http_client

    def spy() -> httpx.Client:
        client = factory()
        built.append(client)
        return client

    monkeypatch.setattr(phoenix, "publish_http_client", spy)

    captured: dict[str, object] = {}

    class StubClient:
        def __init__(self, *, http_client: httpx.Client | None = None) -> None:
            captured["http_client"] = http_client

    monkeypatch.setattr("phoenix.client.Client", StubClient)

    # A report without a config fails validation, but only after the client is built.
    with pytest.raises(ValidationError):
        phoenix.publish({"config": {}})

    assert len(built) == 1
    assert captured["http_client"] is built[0]
    assert built[0].timeout.read == 600.0
    built[0].close()
