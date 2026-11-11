"""The Prove version post 1 is pinned to (spec 20.3, point 9: "Prove's version for post 1 is pinned at 5A and printed in
`RESULTS-P1.md`"). Session 5A pins the version 8A pinned for the read, 0.1.2 (ruling 5A-4)."""
from __future__ import annotations

from importlib import metadata

PROVE_VERSION = "0.1.2"


class PinMismatch(RuntimeError):
    """The installed Prove is not the pinned one."""


def installed_prove_version() -> str:
    return metadata.version("poisson-prove")


def require_prove_pin() -> str:
    """Return the installed Prove version when it is the pinned one; raise PinMismatch otherwise."""
    got = installed_prove_version()
    if got != PROVE_VERSION:
        raise PinMismatch(f"poisson-prove {got} is installed; post 1 is pinned to {PROVE_VERSION} (ruling 5A-4)")
    return got
