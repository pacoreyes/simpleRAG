import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marca el test como que requiere claves de API en vivo")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration", default=False):
        return
    skip_integration = pytest.mark.skip(reason="Pass --integration to run live API tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Ejecuta los tests de integración que requieren claves de API de Pinecone y Gemini en vivo",
    )
