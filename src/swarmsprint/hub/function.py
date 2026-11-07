"""The hidden function: one BBOB problem from the COCO platform (pip coco-experiment), its instance set by the run seed.

COCO: https://github.com/numbbo/coco (BBOB function definitions: https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctionsdef.pdf)
"""

import os
import tempfile

import cocoex
import numpy as np

LOW, HIGH = -5.0, 5.0  # the BBOB search domain in every coordinate


class Function:
    """f(x) of a seeded BBOB instance; the hub calls it under its store lock."""

    def __init__(self, index: int, dimension: int, seed: int) -> None:
        options = f"dimensions: {dimension} function_indices: {index}"
        self.suite = cocoex.Suite("bbob", f"instances: {seed}", options)  # the problem lives as long as its suite
        self.problem = next(iter(self.suite))
        self.dimension = dimension

    def __call__(self, x: list[float]) -> float:
        return float(self.problem(np.asarray(x, dtype=float)))

    def optimum(self) -> tuple[list[float], float]:
        """The minimiser and minimum, for the log only; COCO writes the minimiser to a file in the working directory."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)  # ponytail: chdir is process-wide; call this before serving requests
            try:
                self.problem._best_parameter("print")
                x = np.loadtxt("._bbob_problem_best_parameter.txt").tolist()
            finally:
                os.chdir(cwd)
        return x, self(x)
