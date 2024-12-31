import numpy as np
from typing import Tuple, Union, List, Optional

def generate_test_vectors(
    size: int, 
    dtype: np.dtype = np.float32,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test vectors for vector operations.
    
    Args:
        size: Size of the vectors
        dtype: Data type for the vectors
        min_val: Minimum value for random generation
        max_val: Maximum value for random generation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of two vectors (a, b)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return (
        np.random.uniform(min_val, max_val, size).astype(dtype),
        np.random.uniform(min_val, max_val, size).astype(dtype)
    )

def generate_test_matrices(
    shape: Tuple[int, int], 
    dtype: np.dtype = np.float32,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test matrices for matrix operations.
    
    Args:
        shape: Shape of the matrices as (rows, cols)
        dtype: Data type for the matrices
        min_val: Minimum value for random generation
        max_val: Maximum value for random generation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of two matrices (A, B)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return (
        np.random.uniform(min_val, max_val, shape).astype(dtype),
        np.random.uniform(min_val, max_val, shape).astype(dtype)
    )

def assert_allclose(
    actual: np.ndarray, 
    expected: np.ndarray, 
    rtol: float = 1e-5, 
    atol: float = 1e-8,
    err_msg: str = ""
) -> None:
    """Assert that two arrays are element-wise equal within a tolerance.
    
    Args:
        actual: Array to check
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        err_msg: Error message to display on failure
    """
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)

def create_task_test_suite(task_name: str, implementations: List[str] = ["cuda", "numpy", "pytorch"]):
    """Create a test suite for comparing different implementations of a task.
    
    This function returns a callable that runs tests across different implementations.
    
    Args:
        task_name: Name of the task
        implementations: List of implementations to test
        
    Returns:
        Test runner function
    """
    from infinite_ml.tasks import get_task
    
    def run_tests(test_func):
        """Run tests for different implementations.
        
        Args:
            test_func: Function that takes an implementation function and performs tests
        """
        results = {}
        
        for impl in implementations:
            try:
                func = get_task(task_name, impl)
                print(f"\nTesting {impl} implementation...")
                result = test_func(func)
                results[impl] = result
                print(f"✓ {impl} implementation passed")
            except Exception as e:
                print(f"✗ {impl} implementation failed: {str(e)}")
                results[impl] = False
        
        return results
    
    return run_tests
