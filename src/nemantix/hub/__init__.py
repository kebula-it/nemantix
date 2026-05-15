from typing import TYPE_CHECKING

from nemantix.hub.event_hub import EventHub
from nemantix.hub.events import Event, EventType
from nemantix.hub.debugger import Debugger
from nemantix.hub.profiler import Profiler
from nemantix.hub.tracer import Tracer

if TYPE_CHECKING:
    from nemantix.hub.observer import Observer
    from nemantix.hub.storage import ObserverLogModel

__all__ = ["EventHub", "Event", "EventType", "Debugger", "Profiler", "Tracer", "Observer", "ObserverLogModel"]


def __getattr__(name):
    if name == "Observer":
        from nemantix.hub.observer import Observer
        globals()["Observer"] = Observer
        return Observer
    if name == "ObserverLogModel":
        from nemantix.hub.storage import ObserverLogModel
        globals()["ObserverLogModel"] = ObserverLogModel
        return ObserverLogModel
    raise AttributeError(f"module 'nemantix.hub' has no attribute {name!r}")
