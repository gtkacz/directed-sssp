import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class ComplexityCheckResult:
	"""Results from complexity checking.

	Attributes:
		is_match: Whether the function matches the expected complexity.
		confidence: Confidence score (0-1) of the match.
		measured_coefficient: The measured constant coefficient.
		r_squared: R-squared value of the curve fit.
		message: Human-readable result message.
	"""

	is_match: bool
	confidence: float
	measured_coefficient: float
	r_squared: float
	message: str


def parse_big_o_expression(big_o_str: str) -> str:
	"""Parse a big-O notation string into a Python-evaluable expression.

	Args:
		big_o_str: Big-O notation string (e.g., "O(n*log(n))").

	Returns:
		Python-evaluable expression string.

	Raises:
		ValueError: If the big-O string format is invalid.
	"""
	# Remove O() wrapper
	match = re.match(r"^O\((.*)\)$", big_o_str.strip(), re.IGNORECASE)
	if not match:
		raise ValueError(f"Invalid big-O format: {big_o_str}")

	expr = match.group(1)

	# Replace common big-O notations with Python equivalents
	replacements = {
		r"\blog\b": "np.log",
		r"\bln\b": "np.log",
		r"\blog2\b": "np.log2",
		r"\blog10\b": "np.log10",
		r"\bsqrt\b": "np.sqrt",
		r"\bmax\b": "np.maximum",
		r"\bmin\b": "np.minimum",
		r"\^": "**",
	}

	for pattern, replacement in replacements.items():
		expr = re.sub(pattern, replacement, expr, flags=re.IGNORECASE)

	return expr


def generate_test_sizes(param_ranges: dict[str, tuple[int, int]], num_samples: int = 10) -> list[dict[str, int]]:
	"""Generate test parameter combinations for complexity testing.

	Args:
		param_ranges: Dictionary mapping parameter names to (min, max) ranges.
		num_samples: Number of different sizes to test per parameter.

	Returns:
		List of parameter dictionaries for testing.
	"""
	# Generate logarithmically spaced values for better coverage
	return [dict(zip(param_ranges.keys(), params, strict=False)) for params in zip(*[
		np.logspace(np.log10(min_val), np.log10(max_val), num_samples, dtype=int)
		for min_val, max_val in param_ranges.values()
	], strict=False)]


def measure_execution_time(func: Callable[..., Any], test_input: Any, num_runs: int = 5) -> float:
	"""Measure average execution time of a function.

	Args:
		func: Function to measure.
		test_input: Input to pass to the function.
		num_runs: Number of runs to average.

	Returns:
		Average execution time in seconds.
	"""
	times = []
	for _ in range(num_runs):
		start = time.perf_counter()
		func(test_input)
		end = time.perf_counter()
		times.append(end - start)

	# Use median to reduce impact of outliers
	return np.median(times)


def check_complexity(
	func: Callable[..., Any],
	big_o_string: str,
	param_dict: dict[str, tuple[int, int]],
	input_generator: Callable[[dict[str, int]], Any] | None = None,
	confidence_threshold: float = 0.85,
	margin: float = 0.2,
	num_samples: int = 10,
) -> ComplexityCheckResult:
	"""Check if a function matches the specified big-O complexity.

	This method empirically validates a function's time complexity by:
	1. Running the function with various input sizes
	2. Measuring execution times
	3. Fitting the data to the expected complexity curve
	4. Evaluating the goodness of fit

	Args:
		func: The function to analyze.
		big_o_string: Big-O notation string (e.g., "O(n*log(n))").
		param_dict: Dictionary mapping parameter names to (min, max) ranges
				   for testing (e.g., {"n": (10, 1000), "m": (5, 500)}).
		input_generator: Optional function that takes a dict of parameters
						and returns the actual input for func. If None,
						assumes func takes the size directly as input.
		confidence_threshold: R-squared threshold for accepting the match (0-1).
		margin: Acceptable deviation margin for the coefficient (0-1).
		num_samples: Number of different input sizes to test.

	Returns:
		ComplexityCheckResult containing match status and statistics.

	Raises:
		ValueError: If the big-O string or parameters are invalid.

	Example:
		>>> def bubble_sort(arr):
		...     n = len(arr)
		...     for i in range(n):
		...         for j in range(n - i - 1):
		...             if arr[j] > arr[j + 1]:
		...                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
		>>> def generate_array(params):
		...     return list(range(params["n"], 0, -1))
		>>> result = check_complexity(
		...     func=bubble_sort, big_o_string="O(n**2)", param_dict={"n": (10, 100)}, input_generator=generate_array
		... )
		>>> print(result.is_match)  # Should be True for bubble sort
	"""
	# Parse the big-O expression
	try:
		complexity_expr = parse_big_o_expression(big_o_string)
	except Exception as e:
		raise ValueError(f"Failed to parse big-O expression: {e}")

	# Generate test cases
	test_cases = generate_test_sizes(param_dict, num_samples)

	# If no input generator provided, assume func takes size directly
	if input_generator is None:

		def default_generator(params: dict[str, int]) -> int:
			# Assume single parameter for simple cases
			return list(params.values())[0]

		input_generator = default_generator

	# Measure execution times
	measured_times = []
	expected_complexities = []

	for params in test_cases:
		try:
			# Generate input
			test_input = input_generator(params)

			# Measure time
			exec_time = measure_execution_time(func, test_input)
			measured_times.append(exec_time)

			# Calculate expected complexity value
			# Create namespace for evaluation
			eval_namespace = {"np": np, **{k.lower(): v for k, v in params.items()}}
			complexity_value = eval(complexity_expr.lower(), eval_namespace)
			expected_complexities.append(complexity_value)

		except Exception as e:
			warnings.warn(f"Failed to test with params {params}: {e}")
			continue

	if len(measured_times) < 3:
		return ComplexityCheckResult(
			is_match=False,
			confidence=0.0,
			measured_coefficient=0.0,
			r_squared=0.0,
			message="Insufficient data points for analysis",
		)

	# Convert to numpy arrays
	measured_times = np.array(measured_times)
	expected_complexities = np.array(expected_complexities)

	# Fit the data: time = coefficient * complexity
	def linear_model(x, coefficient):
		return coefficient * x

	try:
		# Perform curve fitting
		popt, _ = curve_fit(linear_model, expected_complexities, measured_times)
		measured_coefficient = popt[0]

		# Calculate R-squared
		predicted_times = linear_model(expected_complexities, measured_coefficient)
		ss_res = np.sum((measured_times - predicted_times) ** 2)
		ss_tot = np.sum((measured_times - np.mean(measured_times)) ** 2)
		r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

		# Determine if it's a match
		is_match = r_squared >= confidence_threshold

		# Create result message
		if is_match:
			message = (
				f"Function matches {big_o_string} complexity "
				f"(R² = {r_squared:.3f}, coefficient = {measured_coefficient:.2e})"
			)
		else:
			message = (
				f"Function does NOT match {big_o_string} complexity (R² = {r_squared:.3f} < {confidence_threshold})"
			)

		return ComplexityCheckResult(
			is_match=is_match,
			confidence=r_squared,
			measured_coefficient=measured_coefficient,
			r_squared=r_squared,
			message=message,
		)

	except Exception as e:
		return ComplexityCheckResult(
			is_match=False,
			confidence=0.0,
			measured_coefficient=0.0,
			r_squared=0.0,
			message=f"Curve fitting failed: {e}",
		)


# Example usage and test
if __name__ == "__main__":
	# Example 1: Testing a quadratic function
	def quadratic_function(n: int) -> None:
		"""Simulates O(n²) complexity."""
		total = 0
		for i in range(n):
			for j in range(n):
				total += i * j

	# Example 2: Testing a linear function
	def linear_function(n: int) -> None:
		"""Simulates O(n) complexity."""
		total = 0
		for i in range(n):
			total += i

	# Test quadratic function
	result = check_complexity(func=quadratic_function, big_o_string="O(n**2)", param_dict={"n": (10, 100)})
	print(f"Quadratic test: {result.message}")

	# Test linear function against quadratic (should fail)
	result = check_complexity(func=linear_function, big_o_string="O(n**2)", param_dict={"n": (10, 1000)})
	print(f"Linear vs Quadratic: {result.message}")

	# Test linear function against linear (should pass)
	result = check_complexity(func=linear_function, big_o_string="O(n)", param_dict={"n": (10, 1000)})
	print(f"Linear test: {result.message}")
