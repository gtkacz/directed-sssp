from typing import TypedDict


class Proof(TypedDict):
	"""
	Represents a proof with a specific operation and its complexity.

	Attributes:
		operation: The operation being proved.
		complexity: The complexity of the operation.
	"""

	operation: str
	complexity: str
