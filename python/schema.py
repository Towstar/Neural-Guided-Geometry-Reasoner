from __future__ import annotations
from typing import Any

# ---------------------------------------------------------

from pydantic import BaseModel, ConfigDict, field_validator

#----------------------------------------------------------
# Parses the JSON problem files into structured data, and performs validation.
#----------------------------------------------------------

# Represents the entities section of a problem, currently just the list of point names.
class EntitiesModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[str]

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("points must be a list")

        cleaned: list[str] = []
        for p in value:
            if not isinstance(p, str) or not p.strip():
                raise ValueError("each point must be a non-empty string")
            cleaned.append(p)

        return cleaned

class FactModel(BaseModel):
    """ Represents a single geometry fact with a predicate name and its argument list. """
    model_config = ConfigDict(extra="forbid")

    pred: str
    args: list[Any]

    @field_validator("pred")
    @classmethod
    def validate_pred(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("pred must be a non-empty string")
        return value

class ProofStepModel(BaseModel):
    """Represents one optional step in a proof trace, including used facts, derived fact, and rule name."""
    model_config = ConfigDict(extra="forbid")

    facts_used: list[FactModel] | None = None
    derived: FactModel | None = None
    rule: str | None = None


class ProblemModel(BaseModel):
    """Represents one complete geometry problem, including id, entities, givens, goal, and optional proof trace."""
    model_config = ConfigDict(extra="forbid")

    id: str
    entities: EntitiesModel
    givens: list[FactModel]
    goal: FactModel
    proof_trace: list[ProofStepModel] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("id must be a non-empty string")
        return value

class ProblemFileModel(BaseModel):
    """
    Represents a JSON file shaped as an object containing a top-level "problems" list.
    \nSupports JSON of the form:
      {"problems": [ ... ]}
    """
    model_config = ConfigDict(extra="forbid")

    problems: list[ProblemModel]