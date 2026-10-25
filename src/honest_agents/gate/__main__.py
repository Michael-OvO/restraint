"""`python -m honest_agents.gate sweep ...` (gate.sweep) and `python -m honest_agents.gate index ...` (gate.index)."""

from __future__ import annotations

import sys
from collections.abc import Sequence

COMMANDS = ("sweep", "index")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS:
        print(f"usage: python -m honest_agents.gate {{{','.join(COMMANDS)}}} [-h] ...", file=sys.stderr)
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    if args[0] == "sweep":
        from honest_agents.gate.sweep import main as sweep_main

        return sweep_main(args[1:])
    from honest_agents.gate.index import main as index_main

    return index_main(args[1:])


if __name__ == "__main__":
    sys.exit(main())
