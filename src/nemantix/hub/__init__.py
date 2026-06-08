from typing import TYPE_CHECKING

from nemantix.hub.event_hub import EventHub
from nemantix.hub.events import Event, EventType
from nemantix.hub.debugger import Debugger
from nemantix.hub.profiler import Profiler
from nemantix.hub.tracer import Tracer
from nemantix.hub.base import Storable

if TYPE_CHECKING:
    from nemantix.hub.observer import Observer
    from nemantix.hub.storage import EventLogModel

__all__ = ["EventHub", "Event", "EventType", "Debugger", "Profiler", "Tracer",
           "Observer", "EventLogModel", "Storable"]


def __getattr__(name):
    if name == "Observer":
        from nemantix.hub.observer import Observer
        globals()["Observer"] = Observer
        return Observer

    if name == "ObserverLogModel":
        from nemantix.hub.storage import EventLogModel
        globals()["ObserverLogModel"] = EventLogModel
        return EventLogModel

    raise AttributeError(f"module 'nemantix.hub' has no attribute {name!r}")
