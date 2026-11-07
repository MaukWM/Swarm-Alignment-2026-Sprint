# Copyright 2023 DeepMind Technologies Limited. Apache License 2.0:
# https://github.com/google-deepmind/funsearch/blob/65ba52cba984bba8df788e921a4fb7790881177d/LICENSE
# Copied verbatim from https://github.com/google-deepmind/funsearch/blob/65ba52cba984bba8df788e921a4fb7790881177d/cap_set/cap_set.ipynb

# First code cell, unchanged.
"""Finds large cap sets."""
import itertools
import numpy as np


# @funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of an `n`-dimensional cap set."""
  capset = solve(n)
  return len(capset)


def solve(n: int) -> np.ndarray:
  """Returns a large cap set in `n` dimensions."""
  all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

  # Powers in decreasing order for compatibility with `itertools.product`, so
  # that the relationship `i = all_vectors[i] @ powers` holds for all `i`.
  powers = 3 ** np.arange(n - 1, -1, -1)

  # Precompute all priorities.
  priorities = np.array([priority(tuple(vector), n) for vector in all_vectors])

  # Build `capset` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0, n), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a vector with maximum priority to `capset`, and set priorities of
    # invalidated vectors to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    vector = all_vectors[None, max_index]  # [1, n]
    blocking = np.einsum('cn,n->c', (- capset - vector) % 3, powers)  # [C]
    priorities[blocking] = -np.inf
    priorities[max_index] = -np.inf
    capset = np.concatenate([capset, vector], axis=0)

  return capset


# @funsearch.evolve
def priority(el: tuple[int, ...], n: int) -> float:
  """Returns the priority with which we want to add `element` to the cap set."""
  return 0.0


# Code cell 3 (is_cap_set), unchanged except its final line `assert is_cap_set(cap_set_n8)`, which is a top-level check on a cap set
# built in an earlier cell and is not part of the function.
def is_cap_set(vectors: np.ndarray) -> bool:
  """Returns whether `vectors` form a valid cap set.

  Checking the cap set property naively takes O(c^3 n) time, where c is the size
  of the cap set. This function implements a faster check that runs in O(c^2 n).

  Args:
    vectors: [c, n] array containing c n-dimensional vectors over {0, 1, 2}.
  """
  _, n = vectors.shape

  # Convert `vectors` elements into raveled indices (numbers in [0, 3^n) ).
  powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=int)  # [n]
  raveled = np.einsum('in,n->i', vectors, powers)  # [c]

  # Starting from the empty set, we iterate through `vectors` one by one and at
  # each step check that the vector can be inserted into the set without
  # violating the defining property of cap set. To make this check fast we
  # maintain a vector `is_blocked` indicating for each element of Z_3^n whether
  # that element can be inserted into the growing set without violating the cap
  # set property.
  is_blocked = np.full(shape=3 ** n, fill_value=False, dtype=bool)
  for i, (new_vector, new_index) in enumerate(zip(vectors, raveled)):
    if is_blocked[new_index]:
      return False  # Inserting the i-th element violated the cap set property.
    if i >= 1:
      # Update which elements are blocked after the insertion of `new_vector`.
      blocking = np.einsum(
          'nk,k->n',
          (- vectors[:i, :] - new_vector[None, :]) % 3, powers)
      is_blocked[blocking] = True
    is_blocked[new_index] = True  # In case `vectors` contains duplicates.
  return True  # All elements inserted without violating the cap set property.
