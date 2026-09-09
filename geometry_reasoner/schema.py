from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#----------------------------------------------------------
# Defines legal data for geometry facts and problems
#----------------------------------------------------------

# Allowed Predicates
PredicateName = Literal[
    "triangle",
    "collinear",
    "between",
    "midpoint",
    "equal_length",
    "parallel",
    "perpendicular",
    "equal_angle",
    "angle_bisector",
    "on_perp_bisector",
    "is_isosceles",
]

# Allowed argument types for facts. A fact argument can be a point name (string) or a compound term (list of strings).
FactArgument = str | list[str]

# Group predicates that require exactly three point arguments
THREE_POINT_PREDICATES = {
    "triangle",
    "collinear",
    "between",
    "midpoint",
    "on_perp_bisector",
    "is_isosceles",
}

# Requires that point names start with a letter and contain only letters, numbers, and underscores like A, B, P1, mid_point, etc.
POINT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Validation for POINT_NAME_PATTERN
def _validate_point(value: Any) -> str:
    if not isinstance(value, str) or not POINT_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "point names must start with a letter and contain only letters, numbers, "
            "and underscores"
        )
    return value

def _validate_term(value: Any, functor: str, arity: int) -> list[str]:
    """
    validates compound terms such as the following valid terms,
      ["segment", "A", "B"]
      ["line", "A", "B"]
      ["angle", "A", "B", "C"]
     """
    
    if not isinstance(value, list) or len(value) != arity + 1:
        raise ValueError(f"expected {functor} term with {arity} point arguments")
    # Required that the first item of the list is the functor (segment, line, or angle)
    if value[0] != functor:
        raise ValueError(f"expected a {functor} term")
    for point in value[1:]:
        _validate_point(point)
    return value

class EntitiesModel(BaseModel):
    """Entities of a problem, including points, lines, and circles. All names must be valid point names."""
    
    model_config = ConfigDict(extra="forbid")

    points: list[str]
    lines: list[str] = Field(default_factory=list)
    circles: list[str] = Field(default_factory=list)

    @field_validator("points", "lines", "circles")
    @classmethod
    def validate_name_list(cls, value: list[str]) -> list[str]:
        return [_validate_point(item) for item in value]

# Represents a single geometry fact, 
class FactModel(BaseModel):
    """A geometry fact constrained to the predicates understood by Prolog."""

    model_config = ConfigDict(extra="forbid")

    pred: PredicateName
    args: list[FactArgument]

    @model_validator(mode="after")
    def validate_signature(self) -> "FactModel":
        args = self.args

        if self.pred in THREE_POINT_PREDICATES:
            if len(args) != 3:
                raise ValueError(f"{self.pred} expects exactly three points")
            for value in args:
                _validate_point(value)
        elif self.pred == "angle_bisector":
            if len(args) != 4:
                raise ValueError("angle_bisector expects exactly four points")
            for value in args:
                _validate_point(value)
        elif self.pred == "equal_length":
            if len(args) != 2:
                raise ValueError("equal_length expects exactly two segment terms")
            for value in args:
                _validate_term(value, "segment", 2)
        elif self.pred in {"parallel", "perpendicular"}:
            if len(args) != 2:
                raise ValueError(f"{self.pred} expects exactly two line terms")
            for value in args:
                _validate_term(value, "line", 2)
        elif self.pred == "equal_angle":
            if len(args) != 2:
                raise ValueError("equal_angle expects exactly two angle terms")
            for value in args:
                _validate_term(value, "angle", 3)

        return self


class ProofStepModel(BaseModel):
    """One verified step in a proof trace."""

    model_config = ConfigDict(extra="forbid")

    # facts used is null for now since we do not return premise provenance yet. It will be a list of FactModel objects when we do.
    facts_used: list[FactModel] | None = None 
    derived: FactModel | None = None
    rule: str | None = None


class ProblemModel(BaseModel):
    """A complete canonical geometry problem."""

    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str | None = None
    entities: EntitiesModel
    givens: list[FactModel]
    goal: FactModel
    metadata: dict[str, Any] | None = None
    proof_trace: list[ProofStepModel] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("id must be a non-empty string")
        return value


class ProblemFileModel(BaseModel):
    """A wrapper for files shaped as ``{"problems": [...]}``."""

    model_config = ConfigDict(extra="forbid")

    problems: list[ProblemModel]

# What we parse the model response into.
class FormalizedGeometryModel(BaseModel):
    """The minimal geometry payload generated by the language model."""

    # metadata, IDs, entities, proof traces are excluded and are managed solely by Python

    model_config = ConfigDict(extra="forbid")

    statement: str
    givens: list[FactModel]
    goal: FactModel

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("statement must be non-empty")
        return value.strip()


class FormalizationResult(BaseModel):
    """
    A schema-constrained success or explicit unsupported result.
    
    i.e. a wrapper for 
    {"status": "ok", "problem": {...}} 
    or 
    {"status": "unsupported", "problem": null, "unsupported_reason": "..."}
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unsupported"]
    problem: FormalizedGeometryModel | None
    unsupported_reason: str | None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "FormalizationResult":
        if self.status == "ok":
            if self.problem is None:
                raise ValueError("an ok formalization must include a problem")
            if self.unsupported_reason is not None:
                raise ValueError("an ok formalization cannot include an unsupported reason")
        else:
            if self.problem is not None:
                raise ValueError("an unsupported formalization cannot include a problem")
            if self.unsupported_reason is None or not self.unsupported_reason.strip():
                raise ValueError("an unsupported formalization must include a reason")
        return self
