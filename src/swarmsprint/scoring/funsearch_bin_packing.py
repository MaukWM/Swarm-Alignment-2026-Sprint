# Copyright 2023 DeepMind Technologies Limited. Apache License 2.0:
# https://github.com/google-deepmind/funsearch/blob/65ba52cba984bba8df788e921a4fb7790881177d/LICENSE
# Copied verbatim from https://github.com/google-deepmind/funsearch/blob/65ba52cba984bba8df788e921a4fb7790881177d/bin_packing/bin_packing.ipynb

# Code cell 1 (counting from 0), unchanged except an added `import numpy as np` (the notebook imports it in the dataset cell).
import numpy as np

"""Finds heuristics for online 1d binpacking."""


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns indices of bins in which item can fit."""
  return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
    items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
  """Performs online binpacking of `items` into `bins`."""
  # Track which items are added to each bin.
  packing = [[] for _ in bins]
  # Add items to bins.
  for item in items:
    # Extract bins that have sufficient space to fit item.
    valid_bin_indices = get_valid_bin_indices(item, bins)
    # Score each bin based on heuristic.
    priorities = priority(item, bins[valid_bin_indices])
    # Add item to bin with highest priority.
    best_bin = valid_bin_indices[np.argmax(priorities)]
    bins[best_bin] -= item
    packing[best_bin].append(item)
  # Remove unused bins from packing.
  packing = [bin_items for bin_items in packing if bin_items]
  return packing, bins


# @funsearch.run
def evaluate(instances: dict) -> float:
  """Evaluate heuristic function on a set of online binpacking instances."""
  # List storing number of bins used for each instance.
  num_bins = []
  # Perform online binpacking for each instance.
  for name in instances:
    instance = instances[name]
    capacity = instance['capacity']
    items = instance['items']
    # Create num_items bins so there will always be space for all items,
    # regardless of packing order. Array has shape (num_items,).
    bins = np.array([capacity for _ in range(instance['num_items'])])
    # Pack items into bins and return remaining capacity in bins_packed, which
    # has shape (num_items,).
    _, bins_packed = online_binpack(items, bins)
    # If remaining capacity in a bin is equal to initial capacity, then it is
    # unused. Count number of used bins.
    num_bins.append((bins_packed != capacity).sum())
  # Score of heuristic function is negative of average number of bins used
  # across instances (as we want to minimize number of bins).
  return -np.mean(num_bins)


# @funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns priority with which we want to add item to each bin.

  Args:
    item: Size of item to be added to the bin.
    bins: Array of capacities for each bin.

  Return:
    Array of same size as bins with priority score of each bin.
  """
  return -(bins - item)


# Code cell 5 (counting from 0; is_valid_packing), unchanged except its trailing loop over the notebook's datasets:
# `for name in datasets['Weibull 5k']:` ... (11 lines).
def is_valid_packing(
    packing: list[list[float, ...], ...], items: list[float], capacity: float
) -> bool:
  """Returns whether `packing` is valid.

  Returns whether `packing` is a valid packing of `items` into bins of size
  `capacity`.

  Args:
    packing: Packing of items into bins. List of bins, where each bin contains
      a list of items packed into that bin.
    items: List of item sizes.
    capacity: Capacity of each bin.
  """
  # Check that items in packing are exactly the same as list of input items.
  packed_items = sum(packing, [])  # Join items in each bin into a single list.
  if sorted(packed_items) != sorted(items):
    return False

  # Check that each bin contains less than `capacity` items .
  for bin_items in packing:
    if sum(bin_items) > capacity:
      return False

  return True
