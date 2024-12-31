from typing import Callable, Any
import numpy as np

def get_implementation(impl_type: str = "cuda") -> Callable:
    """Get the Hello World implementation based on the specified type.
    
    Args:
        impl_type: Implementation type to use ('cuda', 'numpy', or 'pytorch')
        
    Returns:
        Function that performs Hello World
    """
    if impl_type == "cuda":
        try:
            from infinite_ml.tasks.task_001_hello_world.cuda import hello_world
            print("Using CUDA implementation")
            return hello_world
        except ImportError as e:
            print(f"Error importing CUDA implementation: {e}")
            # Fall back to Python implementation
            impl_type = "python"
    
    if impl_type == "numpy":
        from infinite_ml.tasks.task_001_hello_world.numpy import hello_world
        return hello_world
    
    # Default to pure Python implementation
    from infinite_ml.tasks.task_001_hello_world.python import hello_world
    return hello_world
