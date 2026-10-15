"""`python -m honest_agents.train`."""

import sys

from honest_agents.train.cli import main, record_exit_status

status = main()
record_exit_status(status)  # for scripts/train.sh: `accelerate launch` turns every non-zero status into 1
sys.exit(status)
