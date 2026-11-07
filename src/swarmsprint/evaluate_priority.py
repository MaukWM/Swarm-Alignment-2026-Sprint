"""Sandbox entry point: read an agent's program on stdin, print the cap set DeepMind's `solve` builds with it."""

import json
import resource
import sys
from pathlib import Path

# Limits for the agent's program: 1 GB memory, 20 s CPU, 1 MB per written file.
resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))
sys.path.insert(0, str(Path(__file__).parents[1]))
from swarmsprint import funsearch_cap_set as fs  # noqa: E402

out, sys.stdout = sys.stdout, sys.stderr  # prints from the program must not reach the result
del fs.priority  # DeepMind's placeholder; the agent must define its own
exec(sys.stdin.read(), fs.__dict__)
print(json.dumps(fs.solve(8).tolist()), file=out)
