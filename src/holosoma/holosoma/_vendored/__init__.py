"""Holosoma vendored third-party packages.

This namespace contains verbatim copies of upstream projects that have been
re-namespaced so that they can be imported as `holosoma._vendored.<pkg>`. The
goal is byte-equivalent runtime behavior with the original repositories while
keeping a single, auditable Python import path under holosoma's installed tree.

Code under this package is exempt from holosoma's lint rules (see
`pyproject.toml`). Treat it as an internal compatibility layer; do not import
from `holosoma._vendored.*` from holosoma's stable public API.
"""
