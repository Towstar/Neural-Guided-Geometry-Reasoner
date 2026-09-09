from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .canonicalize import canonicalize_problem
from .schema import FactModel, FormalizationResult, ProblemModel


DEFAULT_MODEL = "gpt-5.6-luna"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

SYSTEM_PROMPT = """You convert Euclidean geometry questions into a small symbolic DSL.
Extract only facts explicitly stated in the question; do not solve the problem or add
facts that merely follow from the givens. Use the exact point labels shown in the input.
Return status 'unsupported' when the question is illegible, ambiguous, or cannot be
represented using the supported signatures below.

Supported signatures:
- triangle(A, B, C), collinear(A, B, C), between(A, B, C)
- midpoint(M, A, B), on_perp_bisector(P, A, B), is_isosceles(A, B, C)
- angle_bisector(P, A, B, C)
- equal_length(segment(A, B), segment(C, D))
- parallel(line(A, B), line(C, D)), perpendicular(line(A, B), line(C, D))
- equal_angle(angle(A, B, C), angle(D, E, F))

Represent compound terms as JSON arrays whose first item is segment, line, or angle.
The goal is the proposition the question asks to prove. For image input, transcribe the
question into the statement field before formalizing it.
"""


class FormalizationError(RuntimeError):
    """Base class for safe, user-facing formalization failures."""


class MissingAPIKeyError(FormalizationError):
    pass


class InvalidImageError(FormalizationError):
    pass


class FormalizationAPIError(FormalizationError):
    pass


class FormalizationRefusalError(FormalizationError):
    pass


class IncompleteFormalizationError(FormalizationError):
    pass


@dataclass(frozen=True)
class CallMetadata:
    model: str
    response_id: str | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


def openai_api_key_from_environment() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OpenAI API key is missing; set OPENAI_API_KEY in .env"
        )
    return api_key

# For Example, returns {A,B,C} for 
# { "pred": "equal_length", "args": [["segment", "A", "D"], ["segment", "B", "D"]]}
def _point_names(fact: FactModel) -> set[str]:
    """Extract point names from a geometry fact. Ignores non points"""
    points: set[str] = set()
    for argument in fact.args:
        if isinstance(argument, str):
            points.add(argument)
        else:
            points.update(argument[1:])
    return points


def materialize_problem(
    result: FormalizationResult,
    source_fingerprint: bytes,
) -> dict[str, Any]:
    """Create an internal problem dict from a formalization result, including a unique id based on the source fingerprint & canonicalizes it"""
    if result.status != "ok" or result.problem is None:
        raise ValueError("cannot materialize an unsupported formalization")

    generated = result.problem
    points: set[str] = set()
    for fact in [*generated.givens, generated.goal]:
        points.update(_point_names(fact))

    # create a unique 12 digit id in the docker container style
    digest = hashlib.sha256(source_fingerprint).hexdigest()[:12]
    problem = ProblemModel(
        id=f"input_{digest}",
        statement=generated.statement,
        entities={"points": sorted(points)},
        givens=generated.givens,
        goal=generated.goal,
    )
    return canonicalize_problem(problem.model_dump())


def _find_refusal(response: Any) -> str | None:
    """see if the model refused to formalize the input, and if so return a refusal message"""
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "refusal", None):
                return "The model refused to formalize this input."
    return None

def _usage_value(usage: Any, name: str) -> int | None:
    """Returns the token count for a given usage field, or None if not present."""
    value = getattr(usage, name, None) if usage is not None else None
    return value if isinstance(value, int) else None


class OpenAIFormalizer:
    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.client = client or OpenAI(api_key=openai_api_key_from_environment(), timeout=60.0)
        self.last_call_metadata: CallMetadata | None = None

    def formalize_text(self, text: str) -> FormalizationResult:
        clean_text = text.strip()
        if not clean_text:
            raise FormalizationError("geometry text must be non-empty")
        return self._request(clean_text)

    def formalize_image(self, path: str | Path) -> FormalizationResult:
        image_bytes, mime_type = _read_image(Path(path))
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content = [
            {
                "type": "input_text",
                "text": "Read and formalize the geometry question in this image.",
            },
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "original",
            },
        ]
        return self._request(content)

    def _request(self, user_content: Any) -> FormalizationResult:
        started = time.perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                text_format=FormalizationResult,
                reasoning={"effort": "low"},
                max_output_tokens=2000,
                store=False,
            )
        except AuthenticationError as error:
            raise FormalizationAPIError(
                "OpenAI authentication failed; check OPENAI_API_KEY"
            ) from error
        except APITimeoutError as error:
            raise FormalizationAPIError("OpenAI request timed out") from error
        except RateLimitError as error:
            raise FormalizationAPIError("OpenAI rate limit was reached") from error
        except APIConnectionError as error:
            raise FormalizationAPIError("Could not connect to the OpenAI API") from error
        except APIStatusError as error:
            raise FormalizationAPIError(
                f"OpenAI request failed with status {error.status_code}"
            ) from error
        except ValidationError as error:
            raise FormalizationAPIError(
                "OpenAI returned data that failed geometry validation"
            ) from error

        latency = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        self.last_call_metadata = CallMetadata(
            model=getattr(response, "model", None) or self.model,
            response_id=getattr(response, "id", None),
            latency_seconds=latency,
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
        )

        refusal = _find_refusal(response)
        if refusal:
            raise FormalizationRefusalError(refusal)

        status = getattr(response, "status", "completed")
        if status != "completed":
            raise IncompleteFormalizationError(
                f"OpenAI response was not completed (status: {status})"
            )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise IncompleteFormalizationError(
                "OpenAI response did not contain a parsed formalization"
            )
        if not isinstance(parsed, FormalizationResult):
            try:
                parsed = FormalizationResult.model_validate(parsed)
            except ValidationError as error:
                raise FormalizationAPIError(
                    "OpenAI returned data that failed geometry validation"
                ) from error
        return parsed


def _read_image(path: Path) -> tuple[bytes, str]:
    if not path.exists() or not path.is_file():
        raise InvalidImageError(f"image file does not exist: {path}")
    mime_type = IMAGE_TYPES.get(path.suffix.lower())
    if mime_type is None:
        supported = ", ".join(sorted(IMAGE_TYPES))
        raise InvalidImageError(f"unsupported image type; use one of: {supported}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise InvalidImageError("image exceeds the 20 MiB application limit")

    try:
        with Image.open(path) as image:
            image.verify()
        if path.suffix.lower() == ".gif":
            with Image.open(path) as image:
                if getattr(image, "is_animated", False):
                    raise InvalidImageError("animated GIF input is not supported")
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageError("image could not be decoded") from error

    return path.read_bytes(), mime_type


def formalize_text(text: str) -> FormalizationResult:
    return OpenAIFormalizer().formalize_text(text)


def formalize_image(path: str | Path) -> FormalizationResult:
    return OpenAIFormalizer().formalize_image(path)
