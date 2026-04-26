"""sciknow — local-first scientific knowledge system.

The package version is read from ``importlib.metadata`` so it stays in
sync with ``pyproject.toml`` automatically — no manual duplication
between metadata and code. Falls back to "0.0.0+unknown" if the
package isn't installed via uv/pip (e.g. a bare ``PYTHONPATH=. python
-c 'import sciknow'``).
"""
from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("sciknow")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except ImportError:  # pragma: no cover — Python < 3.8
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
