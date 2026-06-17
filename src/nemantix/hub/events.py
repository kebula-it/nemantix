import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemantix.core.script import Script


class EventType(Enum):
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
    # TODO: rename as "phase_event"?
    EXECUTOR_PHASE_START = auto()
    EXECUTOR_PHASE_END = auto()
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
    type: EventType
    lines: tuple[int, int]
    scope: str
    script: "Script | None"
    statement: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
