import logging
from unittest.mock import MagicMock, patch

import pytest

from nemantix.common.connectors import DBConnector
from nemantix.hub import EventType
from nemantix.hub.observer import (
    AgentMetrics,
    Observer,
    ObserverLogHandler,
    SystemMetrics,
)


class MockEvent:
    def __init__(self, type_: EventType, payload=None, lines=None, timestamp=1600000000.0):
        self.type = type_
        self.payload = payload or {}
        self.lines = lines or (1, 1)
        self.timestamp = timestamp


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_connector():
    """Provides a mocked database connector."""
    connector = MagicMock(spec=DBConnector)
    # Mock the context manager for get_session(): `with connector.get_session() as session:`
    mock_session = MagicMock()
    connector.get_session.return_value.__enter__.return_value = mock_session
    return connector


@pytest.fixture
def observer(mock_connector):
    """Provides a fresh Observer instance with mocked psutil to prevent system noise."""
    with patch('psutil.Process') as mock_process, \
            patch('psutil.net_io_counters') as mock_net:
        # Setup fake initial hardware states
        mock_proc_instance = mock_process.return_value
        mock_proc_instance.io_counters.return_value.read_count = 100
        mock_proc_instance.io_counters.return_value.write_count = 50
        mock_net.return_value.bytes_recv = 1024
        mock_net.return_value.bytes_sent = 2048

        obs = Observer(connector=mock_connector)
        yield obs


# ==========================================
# OBSERVER TESTS
# ==========================================

def test_observer_initialization(observer, mock_connector):
    """Ensures metrics are initialized and DB tables are created if a connector is passed."""
    assert isinstance(observer.hardware, SystemMetrics)
    assert isinstance(observer.agent, AgentMetrics)
    mock_connector.create_tables.assert_called_once()


def test_hardware_tracking_calculates_deltas(observer):
    """Tests if the start/stop cycle accurately calculates hardware deltas."""
    # 1. Start Tracking
    observer.start_hardware_tracking(MockEvent(EventType.MONITOR_START))
    assert observer._is_tracking is True

    # 2. Simulate hardware changes over time
    observer.process.cpu_percent.return_value = 45.5
    observer.process.memory_info.return_value.rss = 104857600  # 100 MB

    # Simulate I/O advancing
    new_io_mock = MagicMock()
    new_io_mock.read_count = 150
    new_io_mock.write_count = 70
    observer.process.io_counters.return_value = new_io_mock

    # Simulate Network advancing
    with patch('psutil.net_io_counters') as mock_net:
        mock_net.return_value.bytes_recv = 2048  # Delta: 1024 bytes (1 KB)
        mock_net.return_value.bytes_sent = 4096  # Delta: 2048 bytes (2 KB)

        # 3. Stop Tracking
        observer.stop_hardware_tracking(MockEvent(EventType.MONITOR_STOP))

    # Assertions
    assert observer._is_tracking is False
    assert observer.hardware.cpu_percent == 45.5
    assert observer.hardware.ram_mb == 100.0
    assert observer.hardware.io_read_count == 50
    assert observer.hardware.io_write_count == 20
    assert observer.hardware.network_kb_recv == 1.0
    assert observer.hardware.network_kb_sent == 2.0


def test_agent_metrics_accumulation(observer):
    """Tests if the observer correctly accumulates cognitive metrics."""
    # Simulate LLM calls (1 internal, 2 external)
    observer.on_llm(MockEvent(EventType.LLM, {'name': 'gpt-4', 'internal_usage': True}))
    observer.on_llm(MockEvent(EventType.LLM, {'name': 'gpt-4', 'internal_usage': False}))
    observer.on_llm(MockEvent(EventType.LLM, {'name': 'gpt-4', 'internal_usage': False}))

    # Simulate Tool Calls
    observer.on_tool_call(MockEvent(EventType.CALL_ENTER, {'type': 'tool', 'name': 'search'}))
    observer.on_tool_call(MockEvent(EventType.CALL_ENTER, {'type': 'tool', 'name': 'search'}))
    observer.on_tool_call(
        MockEvent(EventType.CALL_ENTER, {'type': 'action', 'name': 'ignore_me'}))  # Should be ignored

    # Simulate User Request & Runtime Coding
    observer.on_user_request(MockEvent(EventType.USER_REQUEST))
    observer.on_runtime_coding(MockEvent(EventType.EXECUTOR_PHASE_START, {'phase': 'code_deliberate'}))

    # Simulate Errors
    observer.on_error(MockEvent(EventType.ERROR, "Syntax Error", lines=(5, 5)))

    # Assertions
    assert observer.agent.llm_calls['gpt-4']['internal'] == 1
    assert observer.agent.llm_calls['gpt-4']['external'] == 2
    assert observer.agent.tool_frequencies['search'] == 2
    assert 'ignore_me' not in observer.agent.tool_frequencies
    assert observer.agent.user_requests == 1
    assert observer.agent.runtime_codings == 1
    assert observer.agent.errors == 1
    assert "[ERROR] Line (5, 5): Syntax Error" in observer.agent.logs


def test_on_log_saves_to_db(observer, mock_connector):
    """Tests if logs are appended to memory and written to the DB."""

    # We must patch the inline import inside `_save_to_db`
    mock_log_model = MagicMock()
    modules_patch = {'nemantix.hub.storage': MagicMock(ObserverLogModel=mock_log_model)}

    with patch.dict('sys.modules', modules_patch):
        event = MockEvent(EventType.LOG_EVENT, payload="System starting up", timestamp=1600000000.0)
        observer.on_log(event)

    # Check in-memory list
    assert "[LOG] System starting up" in observer.agent.logs

    # Check DB transaction
    mock_session = mock_connector.get_session.return_value.__enter__.return_value
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


# ==========================================
# LOG HANDLER TESTS
# ==========================================

@patch('nemantix.hub.observer.context')
def test_observer_log_handler_emits_event(mock_context):
    """Tests if the custom logging handler correctly pipes Python logs into the EventHub."""

    # Setup mock event hub
    mock_hub = MagicMock()
    mock_hub.has_subscribers.return_value = True
    mock_context.event_hub.get.return_value = mock_hub

    # Setup Log Handler
    handler = ObserverLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Create a fake log record
    record = logging.LogRecord(
        name="test_logger", level=logging.INFO, pathname="test.py", lineno=42,
        msg="Hello from logger", args=(), exc_info=None
    )

    # Execute
    handler.emit(record)

    # Assert
    mock_hub.emit.assert_called_once()
    emitted_event = mock_hub.emit.call_args[0][0]

    assert emitted_event.type == EventType.LOG_EVENT
    assert emitted_event.payload['msg'] == "Hello from logger"
    assert emitted_event.payload['lineno'] == 42


@patch('nemantix.hub.observer.context')
def test_observer_log_handler_ignores_if_no_hub_or_subscribers(mock_context):
    """Ensures the handler gracefully drops logs if observability is turned off."""
    handler = ObserverLogHandler()
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=1, msg="test", args=(),
                               exc_info=None)

    # Scenario 1: No hub found in context
    mock_context.event_hub.get.return_value = None
    handler.emit(record)  # Should not crash

    # Scenario 2: Hub found, but no one is subscribed to LOG_EVENT
    mock_hub = MagicMock()
    mock_hub.has_subscribers.return_value = False
    mock_context.event_hub.get.return_value = mock_hub
    handler.emit(record)

    mock_hub.emit.assert_not_called()
