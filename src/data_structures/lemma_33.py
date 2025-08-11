import heapq
from collections.abc import Sequence
from typing import TypeVar

K = TypeVar("K")  # Key type
V = TypeVar("V", bound=int | float)  # Value type (numeric, bounded by B)


class Lemma33DataStructure[K, V: int | float]:
	"""
	Lemma 3.3 (pg. 6): Given at most N key/value pairs to be inserted, an integer parameter M, and an upper bound B on all the values involved, there exists a data structure that supports the following operations:

	- INSERT: Insert a key/value pair in amortized O(max{1, log(N/M)}) time. If the key already exists, update its value.
	- BATCH PREPEND: Insert L key/value pairs such that each value in L is smaller than any value currently in the data structure, in amortized O(L·max{1, log(L/M)}) time. If there are multiple pairs with the same key, keep the one with the smallest value.
	- PULL: Return a subset S' of keys where |S' | ≤ M associated with the smallest |S' | values and an upper bound x that separates S' from the remaining values in the data structure, in amortized O(|S' |) time. Specifically, if there are no remaining values, x should be B. Otherwise, x should satisfy max(S') < x ≤ min(D), where D is the set of elements in the data structure after the pull operation.

	Attributes:
		max_pairs: Maximum number of key-value pairs (N).
		pull_limit: Maximum number of elements to return in PULL operation (M).
		value_upper_bound: Upper bound on all values (B).
		key_value_map: Dictionary storing key-value pairs.
		value_heap: Min-heap for efficient retrieval of smallest values.
	"""  # noqa: D415, E501

	def __init__(self, max_pairs: int, pull_limit: int, value_upper_bound: V) -> None:
		"""Initialize the data structure.

		Args:
			max_pairs: Maximum number of key-value pairs to be inserted (N).
			pull_limit: Maximum number of elements to return in PULL operation (M).
			value_upper_bound: Upper bound on all values involved (B).

		Raises:
			ValueError: If parameters are invalid (negative or M > N).
		"""
		if max_pairs <= 0 or pull_limit <= 0:
			raise ValueError("max_pairs and pull_limit must be positive")
		if pull_limit > max_pairs:
			raise ValueError("pull_limit cannot exceed max_pairs")

		self.max_pairs = max_pairs
		self.pull_limit = pull_limit
		self.value_upper_bound = value_upper_bound
		self.key_value_map: dict[K, V] = {}
		self.value_heap: list[tuple[V, K]] = []
		self._heap_valid: set[K] = set()

	def insert(self, key: K, value: V) -> None:
		"""Insert a key-value pair or update existing key.

		Inserts a new key-value pair into the data structure. If the key
		already exists, updates its value.

		Time Complexity: Amortized O(max{1, log(N/M)})

		Args:
			key: The key to insert or update.
			value: The value associated with the key.

		Raises:
			ValueError: If value exceeds the upper bound.
		"""
		if value > self.value_upper_bound:
			raise ValueError(f"Value {value} exceeds upper bound {self.value_upper_bound}")

		# Update existing key or insert new one
		if key in self.key_value_map:
			# Mark old heap entry as invalid
			self._heap_valid.discard(key)

		self.key_value_map[key] = value
		heapq.heappush(self.value_heap, (value, key))
		self._heap_valid.add(key)

	def batch_prepend(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
		"""Insert multiple key-value pairs with values smaller than existing ones.

		Batch inserts L key-value pairs where each value is smaller than any
		value currently in the data structure. For duplicate keys, keeps the
		one with the smallest value.

		Time Complexity: Amortized O(L·max{1, log(L/M)})

		Args:
			key_value_pairs: Sequence of (key, value) tuples to insert.

		Raises:
			ValueError: If any value exceeds the upper bound or is not smaller than existing values.
		"""
		if not key_value_pairs:
			return

		# Verify all values are smaller than existing ones
		if self.key_value_map:
			min_existing = min(self.key_value_map.values())
			max_new = max(value for _, value in key_value_pairs)
			if max_new >= min_existing:
				raise ValueError(f"New values must be smaller than existing minimum {min_existing}")

		# Process pairs, keeping smallest value for duplicate keys
		new_pairs: dict[K, V] = {}
		for key, value in key_value_pairs:
			if value > self.value_upper_bound:
				raise ValueError(f"Value {value} exceeds upper bound {self.value_upper_bound}")

			if key not in new_pairs or value < new_pairs[key]:
				new_pairs[key] = value

		# Insert or update keys
		for key, value in new_pairs.items():
			if key in self.key_value_map:
				# Only update if new value is smaller
				if value < self.key_value_map[key]:
					self._heap_valid.discard(key)
					self.key_value_map[key] = value
					heapq.heappush(self.value_heap, (value, key))
					self._heap_valid.add(key)
			else:
				self.key_value_map[key] = value
				heapq.heappush(self.value_heap, (value, key))
				self._heap_valid.add(key)

	def pull(self) -> tuple[dict[K, V], V]:
		"""Return keys with smallest values and a separating upper bound.

		Returns a subset S' of at most M keys associated with the smallest
		values, along with an upper bound x that separates S' from remaining
		values. If no values remain after pull, x equals B. Otherwise,
		max(S') < x <= min(D) where D is the remaining set.

		Time Complexity: Amortized O(|S'|)

		Returns:
			A tuple containing:
				- Dictionary of pulled key-value pairs (S')
				- Upper bound separating pulled values from remaining (x)
		"""
		result: dict[K, V] = {}

		# Clean invalid heap entries
		while self.value_heap:
			value, key = self.value_heap[0]
			if key not in self._heap_valid or self.key_value_map.get(key) != value:
				heapq.heappop(self.value_heap)
			else:
				break

		# Pull up to M smallest values
		pulled_keys: set[K] = set()
		max_pulled_value: V | None = None

		while len(result) < self.pull_limit and self.value_heap:
			value, key = heapq.heappop(self.value_heap)

			# Skip invalid entries
			if key not in self._heap_valid or self.key_value_map.get(key) != value:
				continue

			result[key] = value
			pulled_keys.add(key)
			max_pulled_value = value

			# Remove from main storage
			del self.key_value_map[key]
			self._heap_valid.discard(key)

		# Determine upper bound x
		if not self.key_value_map:
			# No remaining values
			upper_bound = self.value_upper_bound
		else:
			# Ensure x is between max(S') and min(D)  # noqa: ERA001
			min_remaining = min(self.key_value_map.values())
			upper_bound = min_remaining if max_pulled_value is not None else min_remaining

		return result, upper_bound

	def size(self) -> int:
		"""Return the current number of key-value pairs.

		Returns:
			Number of key-value pairs currently stored.
		"""
		return len(self.key_value_map)

	def is_empty(self) -> bool:
		"""Check if the data structure is empty.

		Returns:
			True if no key-value pairs are stored, False otherwise.
		"""
		return len(self.key_value_map) == 0
