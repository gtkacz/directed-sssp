import heapq
import math
from collections import defaultdict

from data_structures import Lemma33DataStructure


class DirectedSSSP:
	"""Implementation of the O(m log^(2/3) n) SSSP algorithm."""

	def __init__(self, n: int, edges: list[tuple[int, int, float]]):
		"""
		Initialize with n vertices and edges as (u, v, weight) tuples.
		Vertices are numbered 0 to n-1.
		"""
		self.n = n
		self.edges = edges
		self.m = len(edges)

		# Build adjacency list
		self.adj = defaultdict(list)
		for u, v, w in edges:
			self.adj[u].append((v, w))

		# Parameters
		self.k = max(1, int(math.log(n) ** (1 / 3)))
		self.t = max(1, int(math.log(n) ** (2 / 3)))

		# Distance estimates
		self.db = [float("inf")] * n
		self.pred = [-1] * n

	def find_pivots(self, B: float, S: set[int]) -> tuple[set[int], set[int]]:
		"""Algorithm 1: Find pivots to reduce frontier size."""
		W = S.copy()
		W0 = S.copy()

		# Relax for k steps
		for i in range(self.k):
			Wi = set()
			for u in W0:
				for v, w in self.adj[u]:
					if self.db[u] + w <= self.db[v]:
						self.db[v] = self.db[u] + w
						self.pred[v] = u
						if self.db[u] + w < B:
							Wi.add(v)
			W.update(Wi)

			if len(W) > self.k * len(S):
				return S, W

			W0 = Wi

		# Build forest F
		F = defaultdict(set)  # parent -> children
		roots = set()

		for v in W:
			if self.pred[v] in W:
				F[self.pred[v]].add(v)
			elif v in S:
				roots.add(v)

		# Find roots with large trees
		P = set()
		for root in roots:
			tree_size = self._count_tree_size(root, F)
			if tree_size >= self.k:
				P.add(root)

		return P, W

	def _count_tree_size(self, root: int, F: dict[int, set[int]]) -> int:
		"""Count size of tree rooted at given vertex."""
		size = 1
		if root in F:
			for child in F[root]:
				size += self._count_tree_size(child, F)
		return size

	def base_case(self, B: float, x: int) -> tuple[float, set[int]]:
		"""Algorithm 2: Base case for l=0."""
		U0 = {x}
		heap = [(self.db[x], x)]
		in_heap = {x}

		while heap and len(U0) < self.k + 1:
			dist, u = heapq.heappop(heap)
			if u not in U0:
				U0.add(u)

			for v, w in self.adj[u]:
				new_dist = self.db[u] + w
				if new_dist <= self.db[v] and new_dist < B:
					self.db[v] = new_dist
					self.pred[v] = u

					if v not in in_heap:
						heapq.heappush(heap, (self.db[v], v))
						in_heap.add(v)

		if len(U0) <= self.k:
			return B, U0
		max_dist = max(self.db[v] for v in U0)
		U = {v for v in U0 if self.db[v] < max_dist}
		return max_dist, U

	def bmssp(self, l: int, B: float, S: set[int]) -> tuple[float, set[int]]:
		"""Algorithm 3: Bounded Multi-Source Shortest Path."""
		if l == 0:
			assert len(S) == 1
			x = next(iter(S))
			return self.base_case(B, x)

		# Find pivots
		P, W = self.find_pivots(B, S)

		# Initialize data structure
		M = 2 ** ((l - 1) * self.t)
		D = Lemma33DataStructure(M, B)

		for x in P:
			D.insert(x, self.db[x])

		B_prime_0 = min(self.db[x] for x in P) if P else B
		U = set()
		i = 0

		# Main loop
		while len(U) < self.k * (2 ** (l * self.t)) and not D.is_empty():
			i += 1

			# Pull from D
			Si, Bi = D.pull()
			Si = set(Si)

			# Recursive call
			B_prime_i, Ui = self.bmssp(l - 1, Bi, Si)
			U.update(Ui)

			K = []

			# Relax edges from Ui
			for u in Ui:
				for v, w in self.adj[u]:
					new_dist = self.db[u] + w
					if new_dist <= self.db[v]:
						self.db[v] = new_dist
						self.pred[v] = u

						if Bi <= new_dist < B:
							D.insert(v, new_dist)
						elif B_prime_i <= new_dist < Bi:
							K.append((v, new_dist))

			# Batch prepend
			prepend_list = K.copy()
			for x in Si:
				if B_prime_i <= self.db[x] < Bi:
					prepend_list.append((x, self.db[x]))
			D.batch_prepend(prepend_list)

		# Determine final boundary
		B_prime = B_prime_i if i > 0 else B
		if len(U) >= self.k * (2 ** (l * self.t)):
			B_prime = min(B_prime, B_prime_i)

		# Add vertices from W
		for x in W:
			if self.db[x] < B_prime:
				U.add(x)

		return B_prime, U

	def compute_shortest_paths(self, source: int) -> list[float]:
		"""Main algorithm to compute shortest paths from source."""
		# Initialize
		self.db = [float("inf")] * self.n
		self.pred = [-1] * self.n
		self.db[source] = 0

		# Call BMSSP with top-level parameters
		l = math.ceil(math.log(self.n) / self.t) if self.t > 0 else 1
		B = float("inf")
		S = {source}

		B_prime, U = self.bmssp(l, B, S)

		return self.db
