import logging
import os
import platform
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nemantix.common import context
from nemantix.common.logger import get_package_logger
from nemantix.hub.base import Storable
from nemantix.hub.event_hub import EventHub, Observable
from nemantix.hub.events import Event, EventType

if TYPE_CHECKING:
    from nemantix.common.connectors import DBConnector

logger = get_package_logger(__name__)


class EnvironmentDetector:
    @staticmethod
    def is_docker() -> bool:
        """Checks for Docker environments."""
        # 1. Check for the standard .dockerenv file
        if os.path.exists("/.dockerenv"):
            return True

        # 2. Fallback check for cgroups (works for some older/custom containers)
        try:
            with open("/proc/1/cgroup", "rt") as f:
                return "docker" in f.read()
        except Exception:
            return False

    @staticmethod
    def is_kubernetes() -> bool:
        """Checks if running inside a K8s Pod."""
        return "KUBERNETES_SERVICE_HOST" in os.environ

    @staticmethod
    def is_aws() -> bool:
        """Basic check for AWS environments (Lambda, ECS, etc.)."""
        return any(key.startswith("AWS_") for key in os.environ)

    @staticmethod
    def is_azure() -> bool:
        """Checks for Microsoft Azure environments (App Services, Functions, or VMs)."""

        # 1. Fast Check: Azure PaaS/Serverless Environment Variables
        if "WEBSITE_SITE_NAME" in os.environ or "WEBSITE_INSTANCE_ID" in os.environ:
            return True

        # 2. Network Check: Azure Virtual Machine IMDS Endpoint
        try:
            # Azure specifically requires the 'Metadata: true' header
            req = urllib.request.Request(
                "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                headers={"Metadata": "true"},
            )
            # Use a strict 0.5s timeout so local execution doesn't hang
            with urllib.request.urlopen(req, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except Exception:
            # If the network request fails, times out, or is refused, we are not in Azure
            pass

        return False

    @staticmethod
    def get_linux_distro() -> str:
        """Attempts to get the specific Linux distribution name and version."""
        if hasattr(platform, "freedesktop_os_release"):
            try:
                os_info = platform.freedesktop_os_release()
                # Returns something like "Ubuntu 22.04.3 LTS" or "Debian GNU/Linux 12 (bookworm)"
                return os_info.get("PRETTY_NAME", "Linux")
            except OSError:
                pass

        return "Linux"

    @classmethod
    def get_environment_name(cls) -> str:
        """Returns a human-readable string of the current execution environment."""
        if cls.is_kubernetes():
            return "Kubernetes Pod"

        if cls.is_docker():
            return "Docker Container"

        if cls.is_aws():
            return "AWS Cloud"

        if cls.is_azure():
            return "Microsoft Azure"

        # OS-level resolution for bare-metal / local execution
        os_name = platform.system()
        if os_name == "Darwin":
            return "Local Machine (macOS)"
        elif os_name == "Windows":
            return "Local Machine (Windows)"
        elif os_name == "Linux":
            distro = cls.get_linux_distro()
            return f"Local Machine ({distro})"

        return f"Unknown Environment ({os_name})"


def _llm_dict_factory():
    return defaultdict(lambda: dict(internal=0, external=0))


def _json_parse_dict_factory():
    return defaultdict(lambda: dict(success=0, total=0))


@dataclass
class SystemMetrics:
    """Snapshot of hardware utilization."""

    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    io_read_count: int = 0
    io_write_count: int = 0
    network_kb_sent: float = 0.0
    network_kb_recv: float = 0.0
    execution_env: str = field(default_factory=EnvironmentDetector.get_environment_name)


@dataclass
class AgentMetrics:
    """Snapshot of cognitive and behavioral utilization."""

    llm_calls: dict[str, dict[str, int]] = field(default_factory=_llm_dict_factory)
    json_parses: dict[str, dict[str, int]] = field(
        default_factory=_json_parse_dict_factory
    )
    user_requests: int = 0
    runtime_codings: int = 0
    errors: int = 0
    tool_frequencies: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    kb_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    logs: list[str] = field(default_factory=lambda: deque(maxlen=65_536))


class ObserverLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        event_hub = context.event_hub.get()
        if event_hub is None:
            return

        if not event_hub.has_subscribers(EventType.LOG_EVENT):
            return

        payload = dict(
            level=record.levelname,
            name=record.name,
            line=record.lineno,
            message=record.getMessage(),
            thread=record.threadName,
            function=record.funcName,
        )

        event = Event(
            type=EventType.LOG_EVENT,
            lines=(-1, -1),
            scope="",
            script=None,
            statement="",
            payload=payload,
        )
        event_hub.emit(event)


class Observer(Observable, Storable):
    def __init__(self, connector: "DBConnector | None" = None):
        super().__init__(connector)

        import psutil

        self.process = psutil.Process(os.getpid())
        self.hardware = SystemMetrics()
        self.agent = AgentMetrics()
        self.network = psutil.net_io_counters()

        self._is_tracking = False
        self._baseline_io = None
        self._baseline_net = None

    def subscribe(self, event_hub: EventHub):
        event_hub.subscribe(EventType.LLM, self.on_llm)
        event_hub.subscribe(EventType.JSON_PARSE, self.on_json_parse)
        event_hub.subscribe(EventType.CALL_ENTER, self.on_tool_call)
        event_hub.subscribe(EventType.USER_REQUEST, self.on_user_request)
        event_hub.subscribe(EventType.PHASE_START, self.on_runtime_coding)
        event_hub.subscribe(EventType.ERROR, self.on_error)
        event_hub.subscribe(EventType.CODING_ERROR, self.on_error)
        event_hub.subscribe(EventType.LOG_EVENT, self.on_log)
        event_hub.subscribe(EventType.MONITOR_START, self.start_hardware_tracking)
        event_hub.subscribe(EventType.MONITOR_STOP, self.stop_hardware_tracking)
        event_hub.subscribe(EventType.RETRIEVE, self.on_knowledge_base)
        event_hub.subscribe(EventType.EXPAND, self.on_knowledge_base)
        event_hub.subscribe(EventType.EXTEND, self.on_knowledge_base)
        event_hub.subscribe(EventType.GENERALIZE, self.on_knowledge_base)

    def start_hardware_tracking(self, _: Event | None = None):
        """Marks the baseline for all hardware counters."""
        if self._is_tracking:
            return

        import psutil

        self._is_tracking = True
        self.process.cpu_percent(interval=None)

        # record I/O
        if hasattr(self.process, "io_counters"):
            self._baseline_io = self.process.io_counters()

        # record Network
        self._baseline_net = psutil.net_io_counters()

    def stop_hardware_tracking(self, _: Event | None = None):
        """Calculates the deltas since start_hardware_tracking was called."""
        if not self._is_tracking:
            return

        import psutil

        self._is_tracking = False

        self.hardware.cpu_percent = self.process.cpu_percent(interval=None)
        self.hardware.ram_mb = self.process.memory_info().rss / (1024 * 1024)

        # I/O Delta
        if hasattr(self.process, "io_counters") and self._baseline_io:
            current_io = self.process.io_counters()
            self.hardware.io_read_count = (
                current_io.read_count - self._baseline_io.read_count
            )
            self.hardware.io_write_count = (
                current_io.write_count - self._baseline_io.write_count
            )

        # Network Delta
        current_net = psutil.net_io_counters()
        if current_net and self._baseline_net:
            bytes_sent = current_net.bytes_sent - self._baseline_net.bytes_sent
            bytes_recv = current_net.bytes_recv - self._baseline_net.bytes_recv

            self.hardware.network_kb_sent = bytes_sent / 1024.0
            self.hardware.network_kb_recv = bytes_recv / 1024.0

    def on_knowledge_base(self, event: Event):
        self.agent.kb_calls[event.type.name] += 1

    def on_llm(self, event: Event):
        name = event.payload["name"]

        if event.payload.get("internal_usage", False):
            self.agent.llm_calls[name]["internal"] += 1
        else:
            self.agent.llm_calls[name]["external"] += 1

    def on_json_parse(self, event: Event):
        # bucket by the responsible LLM; parses with no LLM go under "unknown"
        name = event.payload.get("name") or "unknown"
        self.agent.json_parses[name]["total"] += 1
        if event.payload.get("success"):
            self.agent.json_parses[name]["success"] += 1

    def on_tool_call(self, event: Event) -> bool:
        if event.payload.get("type", None) == "tool":
            self.agent.tool_frequencies[event.payload["name"]] += 1
            return True

        return False

    def on_user_request(self, _: Event):
        self.agent.user_requests += 1

    def on_runtime_coding(self, event: Event) -> bool:
        if event.payload.get("phase", None) == "code_deliberate":
            self.agent.runtime_codings += 1
            return True

        return False

    def on_error(self, event: Event):
        if event.type == EventType.CODING_ERROR:
            payload = event.payload
            data = dict(
                type="CODING_ERROR",
                error=payload["error"],
                code=payload["code"],
                scope=payload["scope"],
            )
            err_msg = f"[CODING-ERROR] {payload['error']}: {payload['code']} [{payload['scope']}]"
        else:
            data = dict(type="ERROR", lines=event.lines, payload=event.payload)
            err_msg = f"[ERROR] Line {event.lines}: {event.payload}"

        self.agent.errors += 1
        self.agent.logs.append(err_msg)
        self.save(
            timestamp=event.timestamp,
            payload=data,
            event=event.type.name,
            script=self.get_script_location(event),
        )

    def on_log(self, event: Event):
        self.agent.logs.append(f"[LOG] {event.payload}")
        self.save(
            timestamp=event.timestamp,
            payload=event.payload,
            event=event.type.name,
            script=self.get_script_location(event),
        )

    def print(self, line_size=50):
        """Prints a unified view of the system and agent states."""
        if self._is_tracking:
            self.stop_hardware_tracking()

        print("\n" + "=" * line_size)
        print("📊 OBSERVER REPORT")
        print("=" * line_size)

        print("💻 HARDWARE METRICS (Session Window)")
        print(f"  ├─ Environment:   {self.hardware.execution_env}")
        print(f"  ├─ CPU Avg Usage: {self.hardware.cpu_percent}%")
        print(f"  ├─ RAM End State: {self.hardware.ram_mb:.2f} MB")
        print(
            f"  ├─ Disk I/O:      {self.hardware.io_read_count} Reads | "
            f"{self.hardware.io_write_count} Writes"
        )
        print(
            f"  └─ Network:       {self.hardware.network_kb_recv:.2f} KB Down | "
            f"{self.hardware.network_kb_sent:.2f} KB Up"
        )

        print("\n🤖 AGENT METRICS")
        print(f"  ├─ User Requests: {self.agent.user_requests}")

        llm_calls = sum(
            v["internal"] + v["external"] for v in self.agent.llm_calls.values()
        )
        print(f"  ├─ LLM Invocations: {llm_calls}")
        for llm, calls in sorted(
            self.agent.llm_calls.items(), key=lambda x: x[1], reverse=True
        ):
            count = calls["internal"] + calls["external"]
            print(f"    └─ {llm}: {count} calls (internal: {calls['internal']})")

        if self.agent.json_parses:
            print("  ├─ JSON Parsing (success rate):")
            for name, stats in sorted(self.agent.json_parses.items()):
                total = stats["total"]
                success = stats["success"]
                pct = (success / total * 100) if total else 0.0
                print(f"    └─ {name}: {pct:.1f}% ({success}/{total})")
        else:
            print("  ├─ JSON Parsing: none")

        print(f"  ├─ Knowledge Base Usages: {sum(self.agent.kb_calls.values())}")
        for operation, calls in self.agent.kb_calls.items():
            print(f"    └─ {operation}: {calls} calls")

        print(f"  ├─ Runtime codings: {self.agent.runtime_codings}")
        print(f"  └─ Errors Encountered: {self.agent.errors}")

        print("\n🛠️  TOOL UTILIZATION")
        if not self.agent.tool_frequencies:
            print("  └─ No tools utilized.")

        for tool, count in sorted(
            self.agent.tool_frequencies.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  ├─ {tool}: {count} calls")

        print("=" * line_size + "\n")
