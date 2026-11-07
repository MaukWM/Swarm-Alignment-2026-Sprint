"""Runs inside the throwaway scorer container: read an agent's program on stdin, print the cap set `solve` builds."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from swarmsprint import funsearch_cap_set as fs  # noqa: E402

out, sys.stdout = sys.stdout, sys.stderr  # prints from the program must not reach the result
del fs.priority  # DeepMind's placeholder; the agent must define its own
exec(sys.stdin.read(), fs.__dict__)
print(json.dumps(fs.solve(8).tolist()), file=out)
