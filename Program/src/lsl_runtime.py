"""Shared pylsl/liblsl availability detection with actionable diagnostics."""

from __future__ import annotations

from types import ModuleType
from typing import Optional, Tuple


def classify_lsl_import_error(exc: BaseException) -> Tuple[str, str]:
    """Return an availability category and a user-facing explanation."""
    if isinstance(exc, ModuleNotFoundError) and exc.name == "pylsl":
        return (
            "package_missing",
            "The Python package 'pylsl' is not installed. Install project dependencies with `python -m pip install -r requirements.txt`.",
        )
    if isinstance(exc, ModuleNotFoundError):
        return (
            "python_dependency_missing",
            f"The 'pylsl' package is present, but a required Python dependency could not be imported ({exc.name or 'unknown module'}). Details: {exc}",
        )
    if isinstance(exc, (OSError, ImportError, RuntimeError)):
        detail = str(exc).strip() or type(exc).__name__
        return (
            "native_library_unavailable",
            "The Python package 'pylsl' is installed, but its native liblsl library could not be loaded or initialized. "
            f"Check the platform-specific liblsl installation and architecture. Details: {detail}",
        )
    detail = str(exc).strip() or type(exc).__name__
    return "initialization_error", f"pylsl initialization failed: {detail}"


PYLSL: Optional[ModuleType] = None
LSL_AVAILABLE = False
LSL_ERROR_KIND: Optional[str] = None
LSL_ERROR_MESSAGE: Optional[str] = None

try:
    import pylsl as PYLSL  # type: ignore[no-redef]

    # pylsl may be importable while its bundled/installed native library is not.
    PYLSL.library_info()
    LSL_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - platform dependent
    LSL_ERROR_KIND, LSL_ERROR_MESSAGE = classify_lsl_import_error(exc)

if LSL_AVAILABLE:
    StreamInfo = PYLSL.StreamInfo
    StreamOutlet = PYLSL.StreamOutlet
    StreamInlet = PYLSL.StreamInlet
    # pylsl 1.18 exposes resolve_byprop as its supported filtered resolver;
    # older project code used the deprecated resolve_stream alias.
    resolve_stream = getattr(PYLSL, "resolve_stream", PYLSL.resolve_byprop)
    resolve_byprop = PYLSL.resolve_byprop
    resolve_streams = PYLSL.resolve_streams
    local_clock = PYLSL.local_clock
else:
    StreamInfo = None
    StreamOutlet = None
    StreamInlet = None
    resolve_stream = None
    resolve_byprop = None
    resolve_streams = None
    local_clock = None


def lsl_unavailable_message() -> str:
    """Return the diagnosed reason LSL features are unavailable."""
    return LSL_ERROR_MESSAGE or "LSL runtime is unavailable for an unknown reason."
