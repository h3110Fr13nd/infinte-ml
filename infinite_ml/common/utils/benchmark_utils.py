import time
import numpy as np
from typing import Callable, Dict, List, Any, Union, Optional
from infinite_ml.tasks import get_task

def benchmark_function(func: Callable, *args, num_runs: int = 10, **kwargs) -> Dict[str, float]:
    """Run a function multiple times and return timing statistics.
    
    Args:
        func: The function to benchmark
        *args: Positional arguments to pass to the function
        num_runs: Number of times to run the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary containing timing statistics
    """
    times = []
    for _ in range(num_runs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times)
    }

def benchmark_task(
    task_name: str, 
    impls: List[str] = ["cuda", "numpy"], 
    sizes: Optional[List[int]] = None,
    num_runs: int = 10
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Benchmark different implementations of a task.
    
    Args:
        task_name: Name of the task to benchmark
        impls: List of implementation types to benchmark
        sizes: List of input sizes to benchmark. If None, uses sizes from task config
        num_runs: Number of times to run each benchmark
        
    Returns:
        Nested dictionary of benchmark results by implementation and size
    """
    # Try to import task config to get default sizes
    try:
        config = __import__(f"infinite_ml.tasks.{task_name}.config", fromlist=["*"])
        default_sizes = getattr(config, "BENCHMARK_SIZES", [1000, 10000, 100000])
    except (ImportError, AttributeError):
        default_sizes = [1000, 10000, 100000]
    
    sizes = sizes or default_sizes
    results = {}
    
    for impl in impls:
        try:
            # Get the task implementation
            func = get_task(task_name, impl)
            impl_results = {}
            
            for size in sizes:
                # Try to import task-specific benchmark utils
                try:
                    benchmark_module = __import__(
                        f"infinite_ml.tasks.{task_name}.benchmark", 
                        fromlist=["create_benchmark_inputs"]
                    )
                    bench_args, bench_kwargs = benchmark_module.create_benchmark_inputs(size)
                except (ImportError, AttributeError):
                    # Default to generating random arrays if no specific benchmark
                    bench_args = (np.random.random(size).astype(np.float32),)
                    bench_kwargs = {}
                
                # Run benchmark
                stats = benchmark_function(func, *bench_args, num_runs=num_runs, **bench_kwargs)
                impl_results[size] = stats
            
            results[impl] = impl_results
        except Exception as e:
            print(f"Error benchmarking {impl} implementation: {str(e)}")
    
    return results
