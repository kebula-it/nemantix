# llm/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices, ConfigDict


class ActionForLLM(BaseModel):
    name: str = Field(..., description="Action name")
    semantics: Optional[str] = Field(  # <- accept None
        default=None,
        description="Short description of what the action does (can be null/omitted).",
    )
    params: List[str] = Field(
        default_factory=list, description="Allowed parameter names"
    )


class Argument(BaseModel):
    """Represents a single argument as name/value pair."""

    name: str = Field(..., description="Argument name")
    value: str = Field(..., description="Argument value")

    model_config = ConfigDict(extra="forbid")


class ActionChoice(BaseModel):
    """What the LLM must return."""

    name: str = Field(..., description="Selected action name")
    args: List[Argument] = Field(
        default_factory=list,
        validation_alias=AliasChoices("args", "parameters", "params", "arguments"),
        description="Arguments for the selected action",
    )

    model_config = ConfigDict(extra="forbid")


class ActionSelectionPrompt(BaseModel):
    """Optional: structured payload we send to the LLM (nice for logging)."""

    system: str
    request: str
    actions: List[ActionForLLM]
