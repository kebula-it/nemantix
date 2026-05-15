import pytest

CREDENTIALS_PATH = "credentials.json"


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that call real providers",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as calling real external LLMs"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_it = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_it)
