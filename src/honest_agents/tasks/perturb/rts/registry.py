"""`RTS_OPERATORS`: every module of this package (other than `base` and `registry`) that defines a module-level `SPEC`.

Discovery is by `pkgutil`, so an operator lane adds its module and never edits a shared file. Two modules declaring one
name raise, with the tiers named when they differ (`transfer` groups cells by operator, so one name in two tiers would
be silently merged).
"""

from __future__ import annotations

import importlib
import pkgutil

from honest_agents.tasks.perturb.rts.base import RTSOperatorSpec

_SKIP = frozenset({"base", "registry"})


def discover(package: str = __package__) -> dict[str, RTSOperatorSpec]:
    """Operator name -> spec for every submodule of `package` with a `SPEC` (a module without one is not an operator)."""
    pkg = importlib.import_module(package)
    found: dict[str, tuple[RTSOperatorSpec, str]] = {}
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        if info.name in _SKIP or info.name.startswith("_") or info.ispkg:
            continue
        module = f"{package}.{info.name}"
        spec = getattr(importlib.import_module(module), "SPEC", None)
        if spec is None:
            continue
        if not isinstance(spec, RTSOperatorSpec):
            raise TypeError(f"{module}.SPEC is a {type(spec).__name__}, not an RTSOperatorSpec")
        if spec.name in found:
            other, where = found[spec.name]
            tiers = f" in two tiers ({other.tier}, {spec.tier})" if other.tier != spec.tier else ""
            raise ValueError(f"operator name {spec.name!r} declared twice{tiers}: {where} and {module}")
        found[spec.name] = (spec, module)
    return {name: spec for name, (spec, _) in sorted(found.items())}


RTS_OPERATORS: dict[str, RTSOperatorSpec] = discover()
