import logging
from unittest.mock import patch

import pytest

from nemantix.common.logger import (
    disable_console_logs,
    get_package_logger,
    update_logger_levels,
)


@pytest.fixture(autouse=True)
def clean_loggers():
    """Fixture to reset the logging manager state before and after each test."""
    logging.root.manager.loggerDict.clear()
    yield
    logging.root.manager.loggerDict.clear()


@patch("nemantix.hub.observer.ObserverLogHandler", autospec=True)
def test_get_package_logger_defaults(MockObserverHandler):
    """Tests logger creation with default arguments (Console + Observer, no File)."""
    logger = get_package_logger("nemantix.test_default")

    assert logger.name == "nemantix.test_default"
    assert logger.level == logging.INFO
    assert logger.propagate is False

    # Should have a StreamHandler and the specific MockObserverHandler instance
    handlers = logger.handlers
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    assert MockObserverHandler.return_value in handlers
    assert not any(isinstance(h, logging.FileHandler) for h in handlers)


@patch("nemantix.common.logger.logging.FileHandler")
@patch("nemantix.hub.observer.ObserverLogHandler")
def test_get_package_logger_with_file(MockObserver, MockFileHandler):
    """Tests logger creation when a log_file is specified."""
    logger = get_package_logger("nemantix.test_file", log_file="test.log")

    assert MockFileHandler.called
    handlers = logger.handlers

    # The return values of our mocks should be registered as handlers
    assert MockFileHandler.return_value in handlers
    assert MockObserver.return_value in handlers


def test_update_logger_levels():
    """Tests dynamic level updating for loggers sharing a specific prefix."""
    log1 = get_package_logger("nemantix.module1", level=logging.INFO)
    log2 = get_package_logger("nemantix.module2", level=logging.INFO)
    other_log = logging.getLogger("other.module")
    other_log.setLevel(logging.INFO)

    update_logger_levels(level=logging.DEBUG, prefix="nemantix")

    # Only nemantix loggers should be updated
    assert log1.level == logging.DEBUG
    assert log2.level == logging.DEBUG
    assert other_log.level == logging.INFO


def test_disable_console_logs():
    """Tests the removal of StreamHandlers from namespaced loggers."""
    logger = get_package_logger("nemantix.no_console", console_logs=True)

    # Ensure it starts with a StreamHandler
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    disable_console_logs(prefix="nemantix")

    # Ensure StreamHandler was removed
    assert not any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
