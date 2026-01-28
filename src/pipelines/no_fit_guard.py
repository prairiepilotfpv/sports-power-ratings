"""Runtime guard that prevents fitting code from running in production schedule mode."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Iterator

_ENABLED = False
_LOG = logging.getLogger(__name__)


def is_no_fit_guard_enabled() -> bool:
    """Return True when fitting is currently forbidden."""
    return _ENABLED


def require_fit_allowed(func_name: str) -> None:
    """Raise when a forbidden fitting function is invoked while the guard is active."""
    if _ENABLED:
        msg = (
            f"Production schedule forbids calling {func_name}. "
            "Run the refresh lane instead."
        )
        _LOG.error(f"Fit guard rejection: {msg} (caller={func_name})")
        raise RuntimeError(msg)


@contextmanager
def enforce_no_fit_guard(enabled: bool) -> Iterator[None]:
    """Context manager that toggles the no-fit guard."""
    global _ENABLED
    previous = _ENABLED
    _ENABLED = enabled
    if enabled:
        _LOG.info("Production schedule mode: fit guard ENABLED - no model fitting allowed")
    try:
        yield
    finally:
        _ENABLED = previous
