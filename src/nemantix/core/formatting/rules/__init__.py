from nemantix.core.formatting._rule import NXFRule
from nemantix.core.formatting.rules.nxf001 import NXF001Rule
from nemantix.core.formatting.rules.nxf002 import NXF002Rule
from nemantix.core.formatting.rules.nxf003 import NXF003Rule
from nemantix.core.formatting.rules.nxf004 import NXF004Rule
from nemantix.core.formatting.rules.nxf005 import NXF005Rule
from nemantix.core.formatting.rules.nxf101 import NXF101Rule
from nemantix.core.formatting.rules.nxf201 import NXF201Rule
from nemantix.core.formatting.rules.nxf202 import NXF202Rule
from nemantix.core.formatting.rules.nxf401 import NXF401Rule
from nemantix.core.formatting.rules.nxf402 import NXF402Rule
from nemantix.core.formatting.rules.nxf501 import NXF501Rule
from nemantix.core.formatting.rules.nxf502 import NXF502Rule

ALL_RULES: list[NXFRule] = [
    NXF001Rule(),
    NXF002Rule(),
    NXF003Rule(),
    NXF004Rule(),
    NXF005Rule(),
    NXF101Rule(),
    NXF201Rule(),
    NXF401Rule(),
    NXF402Rule(),
    NXF202Rule(),
    NXF501Rule(),
    NXF502Rule(),
]

__all__ = [
    "ALL_RULES",
    "NXF001Rule",
    "NXF002Rule",
    "NXF003Rule",
    "NXF004Rule",
    "NXF005Rule",
    "NXF101Rule",
    "NXF201Rule",
    "NXF401Rule",
    "NXF402Rule",
    "NXF202Rule",
    "NXF501Rule",
    "NXF502Rule",
]
