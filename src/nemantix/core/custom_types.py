from pathlib import Path

from nemantix.core import node as nmx_nodes


PathLike = str | Path

PlanOrActionBlock = nmx_nodes.PlanBlock | nmx_nodes.ActionBlock

# Interpreter types
ImportToolsetOrDeliberateStatement = nmx_nodes.ImportToolsetStatement | nmx_nodes.ImportDeliberateStatement
SimilarityQualifier = tuple[nmx_nodes.SimilarityQualifierEnum, str] | nmx_nodes.SimilarityQualifierEnum
