from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from nemantix.core.formatting._edit import NXFEdit


@dataclass
class NXFViolation:
    rule: str
    line: int
    message: str
    fix: NXFEdit | None = field(default=None, compare=False)


class NXFRule(ABC):
    """Base class for all NXF formatting rules.

    Each subclass must declare a ``code`` class attribute and implement
    ``detect()``.  If the violation is auto-fixable, ``detect()`` should
    populate ``NXFViolation.fix`` with the corresponding ``NXFEdit``.
    """

    code: str

    @abstractmethod
    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]: ...
