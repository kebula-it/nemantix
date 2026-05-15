from typing import Dict, Any
from pydantic import BaseModel


class Coordinates(BaseModel):
    """
    Represents the spatial boundaries of a document segment.

    Attributes:
        scheme (str): The coordinate format identifier (e.g., 'text.line_range', 'pdf.bbox').
        value (Dict[str, Any]): The actual coordinate values corresponding to the scheme.
    """
    scheme: str
    value: Dict[str, Any]

    def to_dict(self):
        return self.model_dump()
