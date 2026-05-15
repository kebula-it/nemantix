import pytest


def pytest_addoption(parser):
    # Add a custom command line option
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run tests that call real external resources (Email, SFTP, Messaging)",
    )


def pytest_configure(config):
    # Register the custom marker
    config.addinivalue_line(
        "markers", "external: mark test as calling real external resources"
    )


def pytest_collection_modifyitems(config, items):
    # Skip tests marked with 'external' unless the flag is provided
    if config.getoption("--run-external"):
        return
    skip_external = pytest.mark.skip(reason="need --run-external option to run")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_external)
