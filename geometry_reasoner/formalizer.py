from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from dotenv import load_dotenv
import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from ollama import Client as OllamaClient
from ollama import ResponseError as OllamaResponseError
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .canonicalize import canonicalize_problem
from .schema import FactModel, FormalizationResult, ProblemModel


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_LLAMA_CPP_STARTUP_TIMEOUT_SECONDS = 180.0
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


class MissingModelError(FormalizationError):
    pass


class UnsupportedProviderError(FormalizationError):
    pass


class LlamaCppStartupError(FormalizationError):
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
    provider: str
    model: str
    response_id: str | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class FormalizerConfig:
    """Runtime configuration for one provider-backed formalizer."""

    provider: Literal["openai", "ollama", "llama-cpp"] = "openai"
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 120.0
    llama_cpp_server_path: Path | None = None
    llama_cpp_model_path: Path | None = None
    auto_start_llama_cpp: bool = True
    llama_cpp_startup_timeout_seconds: float = (
        DEFAULT_LLAMA_CPP_STARTUP_TIMEOUT_SECONDS
    )

class Formalizer(Protocol):
    """Provider-neutral boundary used by the CLI and evaluator."""

    provider: str
    model: str
    last_call_metadata: CallMetadata | None

    def formalize_text(self, text: str) -> FormalizationResult: ...

    def formalize_image(self, path: str | Path) -> FormalizationResult: ...


def _inline_local_refs(value: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_inline_local_refs(item, definitions) for item in value]
    if not isinstance(value, dict):
        return value

    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        definition_name = reference.removeprefix("#/$defs/")
        resolved = deepcopy(definitions[definition_name])
        overrides = {
            key: item for key, item in value.items() if key != "$ref"
        }
        return _inline_local_refs({**resolved, **overrides}, definitions)

    return {
        key: _inline_local_refs(item, definitions)
        for key, item in value.items()
    }

def openai_api_key_from_environment() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OpenAI API key is missing; set OPENAI_API_KEY in .env"
        )
    return api_key

def llama_cpp_formalization_schema() -> dict[str, Any]:
    pydantic_schema = FormalizationResult.model_json_schema()
    definitions = pydantic_schema["$defs"]

    problem_schema = _inline_local_refs(
        definitions["FormalizedGeometryModel"],
        definitions,
    )

    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["ok"]},
                    "problem": problem_schema,
                    "unsupported_reason": {"type": "null"},
                },
                "required": [
                    "status",
                    "problem",
                    "unsupported_reason",
                ],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["unsupported"]},
                    "problem": {"type": "null"},
                    "unsupported_reason": {"type": "string"},
                },
                "required": [
                    "status",
                    "problem",
                    "unsupported_reason",
                ],
            },
        ]
    }

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
    provider = "openai"

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = (
            _nonempty(model)
            or _nonempty(os.getenv("OPENAI_MODEL"))
            or DEFAULT_OPENAI_MODEL
        )
        if client is None:
            client_options: dict[str, Any] = {
                "api_key": openai_api_key_from_environment(),
                "timeout": timeout_seconds,
            }
            if resolved_base_url := _nonempty(base_url):
                client_options["base_url"] = resolved_base_url
            self.client = OpenAI(**client_options)
        else:
            self.client = client
        self.last_call_metadata: CallMetadata | None = None

    def formalize_text(self, text: str) -> FormalizationResult:
        return self._request(_validate_text(text))

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
            provider=self.provider,
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

_STARTED_LLAMA_CPP_SERVERS: dict[str, subprocess.Popen] = {}


def _llama_cpp_root_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _llama_cpp_health_status(base_url: str) -> int | None:
    try:
        response = httpx.get(f"{_llama_cpp_root_url(base_url)}/health", timeout=2.0)
    except httpx.HTTPError:
        return None
    return response.status_code


def _llama_cpp_launch_address(base_url: str) -> tuple[str, int]:
    parsed = urlparse(_llama_cpp_root_url(base_url))
    hostname = parsed.hostname
    if parsed.scheme != "http" or hostname is None:
        raise LlamaCppStartupError(
            "LLAMA_CPP_BASE_URL must be a local http URL to start llama.cpp automatically"
        )
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise LlamaCppStartupError(
            "automatic llama.cpp startup is limited to a loopback URL; start remote servers manually"
        )
    if parsed.path not in {"", "/"}:
        raise LlamaCppStartupError(
            "LLAMA_CPP_BASE_URL must not include a path when automatic startup is enabled"
        )
    return hostname, parsed.port or 8080


def _llama_cpp_server_path(configured_path: Path | None) -> Path:
    configured = configured_path or _nonempty(os.getenv("LLAMA_CPP_SERVER_PATH"))
    candidate = Path(configured) if configured else None
    if candidate is None:
        discovered = shutil.which("llama-server")
        candidate = Path(discovered) if discovered else None
    if candidate is None:
        raise LlamaCppStartupError(
            "llama-server was not found; pass --llama-server-path or set LLAMA_CPP_SERVER_PATH"
        )
    if not candidate.is_file():
        raise LlamaCppStartupError(
            f"llama-server does not exist at {candidate}"
        )
    return candidate


def _llama_cpp_model_path(configured_path: Path | None) -> Path:
    configured = configured_path or _nonempty(os.getenv("LLAMA_CPP_MODEL_PATH"))
    if configured is None:
        raise MissingModelError(
            "llama.cpp model path is missing; pass --llama-model-path or set LLAMA_CPP_MODEL_PATH"
        )
    candidate = Path(configured)
    if not candidate.is_file():
        raise MissingModelError(f"llama.cpp model does not exist at {candidate}")
    if candidate.suffix.lower() != ".gguf":
        raise MissingModelError("llama.cpp model path must point to a .gguf file")
    return candidate


def _ensure_llama_cpp_server(
    *,
    base_url: str,
    server_path: Path | None,
    model_path: Path | None,
    auto_start: bool,
    startup_timeout_seconds: float,
) -> None:
    status = _llama_cpp_health_status(base_url)
    if status == 200:
        return
    if status not in {None, 503}:
        raise LlamaCppStartupError(
            f"the service at {_llama_cpp_root_url(base_url)} returned health status {status}"
        )

    process = _STARTED_LLAMA_CPP_SERVERS.get(base_url)
    if status is None and (process is None or process.poll() is not None):
        if not auto_start:
            raise LlamaCppStartupError(
                f"llama.cpp is not reachable at {_llama_cpp_root_url(base_url)}"
            )

        hostname, port = _llama_cpp_launch_address(base_url)
        binary = _llama_cpp_server_path(server_path)
        model = _llama_cpp_model_path(model_path)
        log_path = Path.cwd() / "artifacts" / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [
                    str(binary),
                    "--model",
                    str(model),
                    "--host",
                    hostname,
                    "--port",
                    str(port),
                    "--jinja",
                    "--reasoning",
                    "off",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        _STARTED_LLAMA_CPP_SERVERS[base_url] = process

    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if _llama_cpp_health_status(base_url) == 200:
            return
        if process is not None and process.poll() is not None:
            raise LlamaCppStartupError(
                "llama-server exited before it became ready; see artifacts/llama-server.log"
            )
        time.sleep(0.25)
    raise LlamaCppStartupError(
        "llama-server did not become ready before the startup timeout; see artifacts/llama-server.log"
    )


class LlamaCppFormalizer:
    """Formalizes text through a local llama.cpp server with constrained JSON output."""

    provider = "llama-cpp"

    def __init__(
        self,
        client: httpx.Client | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        server_path: Path | None = None,
        model_path: Path | None = None,
        auto_start: bool = True,
        startup_timeout_seconds: float = DEFAULT_LLAMA_CPP_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        self.base_url = (
            _nonempty(base_url)
            or _nonempty(os.getenv("LLAMA_CPP_BASE_URL"))
            or DEFAULT_LLAMA_CPP_BASE_URL
        )
        self.model = _nonempty(model) or _nonempty(os.getenv("LLAMA_CPP_MODEL")) or "local-model"
        if client is None:
            _ensure_llama_cpp_server(
                base_url=self.base_url,
                server_path=server_path,
                model_path=model_path,
                auto_start=auto_start,
                startup_timeout_seconds=startup_timeout_seconds,
            )
            self.client = httpx.Client(
                base_url=f"{_llama_cpp_root_url(self.base_url)}/v1",
                timeout=timeout_seconds,
            )
        else:
            self.client = client
        self.last_call_metadata: CallMetadata | None = None

    def formalize_text(self, text: str) -> FormalizationResult:
        clean_text = _validate_text(text)
        started = time.perf_counter()
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": clean_text},
                    ],
                    "temperature": 0,
                    "stream": False,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "geometry_formalization",
                            "strict": True,
                            "schema": llama_cpp_formalization_schema(),
                        },
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as error:
            raise FormalizationAPIError("llama.cpp request timed out") from error
        except httpx.HTTPStatusError as error:
            raise FormalizationAPIError(
                f"llama.cpp request failed with status {error.response.status_code}"
            ) from error
        except httpx.HTTPError as error:
            raise FormalizationAPIError("Could not connect to llama.cpp") from error
        except ValueError as error:
            raise FormalizationAPIError("llama.cpp returned invalid JSON") from error

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise IncompleteFormalizationError(
                "llama.cpp response did not contain a formalization"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise IncompleteFormalizationError(
                "llama.cpp response did not contain a formalization"
            )
        try:
            result = FormalizationResult.model_validate_json(content)
        except ValidationError as error:
            raise FormalizationAPIError(
                "llama.cpp returned data that failed geometry validation"
            ) from error

        usage = payload.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        self.last_call_metadata = CallMetadata(
            provider=self.provider,
            model=payload.get("model") or self.model,
            response_id=payload.get("id"),
            latency_seconds=time.perf_counter() - started,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=(
                input_tokens + output_tokens
                if isinstance(input_tokens, int) and isinstance(output_tokens, int)
                else None
            ),
        )
        return result

    def formalize_image(self, path: str | Path) -> FormalizationResult:
        raise FormalizationError(
            "The llama.cpp provider currently supports text input only"
        )

class OllamaFormalizer:
    """Formalizes geometry with an Ollama model selected by its local model tag."""

    provider = "ollama"

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        host: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        load_dotenv()
        self.model = _nonempty(model) or _nonempty(os.getenv("OLLAMA_MODEL"))
        if self.model is None:
            raise MissingModelError(
                "Ollama model is missing; pass --model or set OLLAMA_MODEL"
            )
        self.host = (
            _nonempty(host)
            or _nonempty(os.getenv("OLLAMA_BASE_URL"))
            or _nonempty(os.getenv("OLLAMA_HOST"))
            or DEFAULT_OLLAMA_HOST
        )
        self.client = client or OllamaClient(host=self.host, timeout=timeout_seconds)
        self.last_call_metadata: CallMetadata | None = None

    def formalize_text(self, text: str) -> FormalizationResult:
        return self._request(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _validate_text(text)},
            ]
        )

    def formalize_image(self, path: str | Path) -> FormalizationResult:
        image_bytes, _ = _read_image(Path(path))
        return self._request(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Read and formalize the geometry question in this image.",
                    "images": [image_bytes],
                },
            ]
        )

    def _request(self, messages: list[dict[str, Any]]) -> FormalizationResult:
        started = time.perf_counter()
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=FormalizationResult.model_json_schema(),
                options={"temperature": 0},
                stream=False,
            )
        except OllamaResponseError as error:
            message = (
                f"Ollama model '{self.model}' was not found"
                if error.status_code == 404
                else "Ollama request failed"
            )
            raise FormalizationAPIError(message) from error
        except httpx.TimeoutException as error:
            raise FormalizationAPIError("Ollama request timed out") from error
        except httpx.HTTPError as error:
            raise FormalizationAPIError("Could not connect to Ollama") from error

        if not getattr(response, "done", True):
            raise IncompleteFormalizationError("Ollama response was not completed")

        content = getattr(getattr(response, "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise IncompleteFormalizationError(
                "Ollama response did not contain a formalization"
            )
        try:
            parsed = FormalizationResult.model_validate_json(content)
        except ValidationError as error:
            raise FormalizationAPIError(
                "Ollama returned data that failed geometry validation"
            ) from error

        input_tokens = _usage_value(response, "prompt_eval_count")
        output_tokens = _usage_value(response, "eval_count")
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        self.last_call_metadata = CallMetadata(
            provider=self.provider,
            model=getattr(response, "model", None) or self.model,
            response_id=None,
            latency_seconds=time.perf_counter() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
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


def create_formalizer(config: FormalizerConfig) -> Formalizer:
    """Construct the provider adapter selected by runtime configuration."""

    provider = config.provider.strip().lower()
    if provider == "openai":
        return OpenAIFormalizer(
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )
    if provider == "ollama":
        return OllamaFormalizer(
            model=config.model,
            host=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )
    if provider == "llama-cpp":
        return LlamaCppFormalizer(
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            server_path=config.llama_cpp_server_path,
            model_path=config.llama_cpp_model_path,
            auto_start=config.auto_start_llama_cpp,
            startup_timeout_seconds=config.llama_cpp_startup_timeout_seconds,
        )
    raise UnsupportedProviderError(f"Unsupported provider: {config.provider}")


def formalize_text(
    text: str,
    *,
    formalizer: Formalizer | None = None,
) -> FormalizationResult:
    """Compatibility helper for callers that have not yet injected a formalizer."""

    return (formalizer or create_formalizer(FormalizerConfig())).formalize_text(text)


def formalize_image(
    path: str | Path,
    *,
    formalizer: Formalizer | None = None,
) -> FormalizationResult:
    """Compatibility helper for callers that have not yet injected a formalizer."""

    return (formalizer or create_formalizer(FormalizerConfig())).formalize_image(path)

# General Helpers

def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None

def _validate_text(text: str) -> str:
    clean_text = text.strip()
    if not clean_text:
        raise FormalizationError("geometry text must be non-empty")
    return clean_text
