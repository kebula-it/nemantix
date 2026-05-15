from pydantic import BaseModel, Field
from typing import List, Optional

from nemantix.knowledge_base.document_structure.coordinates import Coordinates


class NodeModel(BaseModel):
    """
    Pydantic schema representing a single node inferred by the AI Planner.
    Used for strict structured output generation.
    """
    kind: str = Field(description="The structural type of the node (e.g., chapter, section).")
    label: str = Field(description="The human-readable title of the node.")

    coordinates: Coordinates

    # Recursive reference allowing the LLM to output a deeply nested tree.
    children: List['NodeModel'] = Field(default_factory=list)

    node_id: Optional[str] = None


class DocumentHierarchyModel(BaseModel):
    """
    Pydantic schema representing the complete document tree inferred by the AI Planner.
    """
    title: str = Field(default="Unknown", description="The title of the document.")
    document_type: str = Field(default="document", description="The semantic category of the document.")
    nodes: List[NodeModel] = Field(default_factory=list)


class MetadataItem(BaseModel):
    """
    Pydantic schema used by the AI Enricher to extract specific metadata entities.
    """

    key: str = Field(description="The category of the metadata (e.g., 'Location', 'Date', 'Entity', 'Topic')")
    value: List[str] = Field(
        description="The extracted values. If there is only one value, put it in a list of length 1.")

    # This configuration is fundamental for OpenAI's Strict Mode (Structured Outputs).
    # It ensures `additionalProperties: False` in the generated JSON Schema.
    model_config = {"extra": "forbid"}


class SegmentEnrichment(BaseModel):
    """
    Pydantic schema defining the final output structure of the AI Enricher for a single chunk.
    """
    summary: str = Field(description="A concise 1-2 sentence summary of the segment content.")
    metadata: List[MetadataItem] = Field(description="List of extracted metadata items.")

    # Required for OpenAI Strict Mode
    model_config = {"extra": "forbid"}
