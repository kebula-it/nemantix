from contextvars import ContextVar

event_hub: ContextVar = ContextVar("event_hub", default=None)
