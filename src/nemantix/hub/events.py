import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemantix.core.script import Script


class BaseEventType(Enum):
    """Abstract base for all event-type enumerations.

    Has no members so it can be subclassed. Define domain-specific event types
    by subclassing this class — no members may be added to ``BaseEventType``
    itself, and ``EventType`` (core) may not be further subclassed.
    """


class EventType(BaseEventType):
    LINE = auto()
    CALL_ENTER = auto()
    CALL_EXIT = auto()
    CODING_START = auto()
    CODING_END = auto()
    CODING_ERROR = auto()
    SCRIPT_UPDATE = auto()
    EXPERTISE_BUILD = auto()
    ERROR = auto()
    BREAKPOINT = auto()
    LLM = auto()
    PHASE_START = auto()
    PHASE_END = auto()
    PROFILE_MARK = auto()
    LOG_EVENT = auto()
    USER_REQUEST = auto()
    MONITOR_START = auto()
    MONITOR_STOP = auto()
    RETRIEVE = auto()
    EXPAND = auto()
    EXTEND = auto()
    GENERALIZE = auto()
    OUTPUT = auto()


@dataclass
class Event:
    type: BaseEventType
    lines: tuple[int, int]
    scope: str
    script: "Script | None"
    statement: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
