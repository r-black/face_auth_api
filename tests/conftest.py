import pytest
from starlette.testclient import TestClient

from service.main import app


@pytest.fixture(scope="module")
@pytest.mark.asyncio
def test_app():
    client = TestClient(app)
    yield client  # testing happens here
