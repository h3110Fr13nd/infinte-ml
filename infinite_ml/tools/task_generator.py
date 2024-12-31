#!/usr/bin/env python
"""
Task template generator for Infinite ML.
This script creates the necessary folder structure and starter files for a new task.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

# Templates for different files
TEMPLATES = {
    "init.py": """# Make package importable
from .interface import get_implementation
""",

    "interface.py": '''from typing import Callable, Any
import numpy as np

def get_implementation(impl_type: str = "cuda") -> Callable:
    """Get the {task_name} implementation based on the specified type.
    
    Args:
        impl_type: Implementation type to use ('cuda', 'numpy', or 'pytorch')
        
    Returns:
        Function that performs {task_name}
    """
    if impl_type == "cuda":
        from .cuda.{task_file_name} import {task_function_name}
        return {task_function_name}
    elif impl_type == "numpy":
        from .numpy.{task_file_name} import {task_function_name}
        return {task_function_name}
    elif impl_type == "pytorch":
        from .pytorch.{task_file_name} import {task_function_name}
        return {task_function_name}
    else:
        raise ValueError(f"Unsupported implementation type: {{impl_type}}")
''',

    "config.py": '''# Configuration settings for {task_name} task

# Default implementation to use
DEFAULT_IMPLEMENTATION = "cuda"

# Available implementations
AVAILABLE_IMPLEMENTATIONS = ["cuda", "numpy", "pytorch"]

# Benchmark sizes to use when running benchmarks
BENCHMARK_SIZES = [1000, 10000, 100000, 1000000]

# Customizable parameters
# Add task-specific parameters here
''',

    "benchmark.py": '''import numpy as np
from typing import Tuple, Dict, Any

def create_benchmark_inputs(size: int) -> Tuple[Tuple, Dict]:
    """Create benchmark inputs for {task_name} of the specified size.
    
    Args:
        size: Size of inputs to create
        
    Returns:
        Tuple of (args, kwargs) for the benchmark function
    """
    # TODO: Create appropriate benchmark inputs for this task
    # Example for a vector operation:
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    
    # Return as args and kwargs
    return ((a, b), {{}})
''',

    "readme.md": '''# {task_title}

## Description
Brief description of the {task_name} task and its purpose.

## Theory
Explain the theoretical background behind {task_name}.

## Implementations

### CUDA
The CUDA implementation leverages parallel processing on the GPU for high performance.

### NumPy
The NumPy implementation uses vectorized operations for CPU-based computation.

### PyTorch
The PyTorch implementation demonstrates how to use a deep learning framework for this task.

## Usage

```python
from infinite_ml.tasks import get_task

# Get the implementation
{task_function_name} = get_task("{task_id}", "cuda")  # or "numpy", "pytorch"

# Use the function
# TODO: Add usage example specific to this task
```

## Performance Characteristics
Discuss expected performance characteristics and scaling behavior.
''',

    "cuda_impl.cu": '''#include "{task_file_name}.cuh"

// CUDA kernel for {task_name}
__global__ void {task_function_name}Kernel(/* TODO: Add parameters */) {{
    // TODO: Implement CUDA kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < /* size */) {{
        // Perform operation
    }}
}}

// Host function that launches the kernel
cudaError_t {task_function_name}Cuda(/* TODO: Add parameters */) {{
    // TODO: Implement host function
    
    // Allocate device memory
    
    // Copy input data to device
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (/* size */ + blockSize - 1) / blockSize;
    {task_function_name}Kernel<<<numBlocks, blockSize>>>(/* parameters */);
    
    // Copy results back to host
    
    // Free device memory
    
    return cudaSuccess;
}}
''',

    "cuda_header.cuh": """#pragma once

#include <cuda_runtime.h>

/**
 * CUDA kernel for {task_name}.
 * 
 * @param // TODO: Add parameters and documentation
 */
__global__ void {task_function_name}Kernel(/* TODO: Add parameters */);

/**
 * Host function for {task_name} using CUDA.
 * 
 * @param // TODO: Add parameters and documentation
 * @return cudaError_t status code
 */
cudaError_t {task_function_name}Cuda(/* TODO: Add parameters */);
""",

    "cuda_binding.cpp": '''#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "{task_file_name}.cuh"

namespace py = pybind11;

// Python binding for the CUDA implementation
py::array_t<float> {task_function_name}(py::array_t<float> input1, py::array_t<float> input2) {{
    // Get buffer info
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    
    // Check dimensions
    if (buf1.size != buf2.size) {{
        throw std::runtime_error("Input shapes must match");
    }}
    
    // Allocate output array
    py::array_t<float> result = py::array_t<float>(buf1.size);
    py::buffer_info buf_result = result.request();
    
    // Get pointers to data
    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);
    
    // Call CUDA function
    // TODO: Update function call and parameters
    {task_function_name}Cuda(ptr1, ptr2, ptr_result, buf1.size);
    
    return result;
}}

PYBIND11_MODULE({task_file_name}_py, m) {{
    m.doc() = "CUDA implementation of {task_name}";
    m.def("{task_function_name}", &{task_function_name}, 
          "Perform {task_name} operation using CUDA",
          py::arg("input1"), py::arg("input2"));
}}
''',

    "cuda_cmakelists.txt": """set(TASK_NAME {cmake_task_name})

include(${{
    CMAKE_SOURCE_DIR
}}/cmake/modules/SetupPybind.cmake)

# Build CUDA library
cuda_add_library(${{
    TASK_NAME
}}_cuda SHARED
    {task_file_name}.cu
)

target_include_directories(${{
    TASK_NAME
}}_cuda PUBLIC
    ${{
        CMAKE_SOURCE_DIR
    }}/infinite_ml/common/cuda
)

# Create Python binding module
setup_task_binding(
    TARGET_NAME {task_file_name}_py
    SOURCE {task_file_name}_binding.cpp
    TASK_NAME ${{
        TASK_NAME
    }}
    TASK_PATH {task_id}/cuda
    LINK_LIBS ${{
        TASK_NAME
    }}_cuda
)
""",
    "numpy_impl.py": '''import numpy as np
from typing import Union, Tuple, List

def {task_function_name}(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """Perform {task_name} operation using NumPy.
    
    Args:
        input1: First input array
        input2: Second input array
        
    Returns:
        Result of the operation
    """
    # TODO: Implement NumPy version
    return input1 + input2  # Example implementation
''',

    "pytorch_impl.py": '''import torch
import numpy as np
from typing import Union, Tuple, List

def {task_function_name}(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """Perform {task_name} operation using PyTorch.
    
    Args:
        input1: First input array (numpy)
        input2: Second input array (numpy)
        
    Returns:
        Result of the operation (numpy)
    """
    # Convert numpy arrays to PyTorch tensors
    tensor1 = torch.from_numpy(input1)
    tensor2 = torch.from_numpy(input2)
    
    # TODO: Implement PyTorch version
    result = tensor1 + tensor2  # Example implementation
    
    # Convert back to numpy
    return result.numpy()
'''
}

def snake_case(name: str) -> str:
    """Convert a name to snake_case."""
    # Replace spaces and hyphens with underscores
    s = re.sub(r'[\s-]', '_', name)
    # Insert underscores between camelCase
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    # Insert underscores between lowercase and uppercase
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    # Convert to lowercase
    s = s.lower()
    # Replace multiple underscores with a single underscore
    s = re.sub('_+', '_', s)
    return s

def camel_case(name: str) -> str:
    """Convert a name to CamelCase."""
    # Replace spaces and hyphens with underscores
    s = re.sub(r'[\s-]', '_', name)
    # Insert underscores between camelCase
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    # Insert underscores between lowercase and uppercase
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    # Split into parts and capitalize each part
    parts = s.split('_')
    return ''.join(part.title() for part in parts)

def generate_task_id(task_name: str) -> str:
    """Generate a task ID from the task name.
    
    Args:
        task_name: Human-readable task name
        
    Returns:
        Generated task ID (e.g., "vector_addition" from "Vector Addition")
    """
    # Convert to snake case and ensure it's a valid identifier
    task_id = snake_case(task_name)
    # Remove any non-alphanumeric characters except underscores
    task_id = re.sub(r'[^a-z0-9_]', '', task_id)
    return task_id

def create_task_folder(
    task_name: str,
    task_id: Optional[str] = None,
    task_function_name: Optional[str] = None,
    project_root: str = str(Path.cwd()),
    impls: List[str] = ["cuda", "numpy", "pytorch"]
) -> None:
    """Create a new task folder with all necessary files.
    
    Args:
        task_name: Human-readable task name (e.g., "Vector Addition")
        task_id: Task ID (e.g., "vector_addition"), generated from task_name if not provided
        task_function_name: Function name to use (default derived from task_name)
        project_root: Project root directory, defaults to current directory's parent
        impls: List of implementations to generate
    """
    print(f"Project root: {project_root} {Path.cwd()} {str(Path.cwd())}")
    # Generate task_id if not provided
    if task_id is None:
        task_id = generate_task_id(task_name)
    
    # Derive names if not provided
    if task_function_name is None:
        task_function_name = snake_case(task_name.replace(" ", "_"))
    
    task_file_name = snake_case(task_name)
    task_title = task_name.title()
    cmake_task_name = snake_case(task_name).replace("_", "")
    
    # Setup paths
    if project_root is None:
        # Try to find the project root by looking for the infinite_ml directory
        current_dir = Path.cwd()
        possible_roots = [current_dir]
        for parent in current_dir.parents:
            possible_roots.append(parent)
            if (parent / "infinite_ml").exists():
                project_root = str(parent)
                break
        else:
            # If not found, default to current directory's parent
            project_root = str(current_dir.parent)
    print(f"Project root: {project_root}")
    tasks_dir = os.path.join(project_root, "infinite_ml", "tasks")
    task_dir = os.path.join(tasks_dir, task_id)
    
    # Make sure task directory doesn't exist
    if os.path.exists(task_dir):
        print(f"Error: Task directory already exists: {task_dir}")
        return
    
    # Create base task directory
    os.makedirs(task_dir, exist_ok=True)
    
    # Create template vars
    template_vars = {
        "task_id": task_id,
        "task_name": task_name,
        "task_function_name": task_function_name,
        "task_file_name": task_file_name,
        "task_title": task_title,
        "cmake_task_name": cmake_task_name
    }
    
    # Create __init__.py
    with open(os.path.join(task_dir, "__init__.py"), "w") as f:
        f.write(TEMPLATES["init.py"])
    
    # Create interface.py
    with open(os.path.join(task_dir, "interface.py"), "w") as f:
        f.write(TEMPLATES["interface.py"].format(**template_vars))
    
    # Create config.py
    with open(os.path.join(task_dir, "config.py"), "w") as f:
        f.write(TEMPLATES["config.py"].format(**template_vars))
    
    # Create benchmark.py
    with open(os.path.join(task_dir, "benchmark.py"), "w") as f:
        f.write(TEMPLATES["benchmark.py"].format(**template_vars))
    
    # Create README.md
    with open(os.path.join(task_dir, "README.md"), "w") as f:
        f.write(TEMPLATES["readme.md"].format(**template_vars))
    
    # Create implementation directories and files
    if "cuda" in impls:
        cuda_dir = os.path.join(task_dir, "cuda")
        os.makedirs(cuda_dir, exist_ok=True)
        
        # Create CUDA implementation files
        with open(os.path.join(cuda_dir, f"{task_file_name}.cu"), "w") as f:
            f.write(TEMPLATES["cuda_impl.cu"].format(**template_vars))
            
        with open(os.path.join(cuda_dir, f"{task_file_name}.cuh"), "w") as f:
            f.write(TEMPLATES["cuda_header.cuh"].format(**template_vars))
            
        with open(os.path.join(cuda_dir, f"{task_file_name}_binding.cpp"), "w") as f:
            f.write(TEMPLATES["cuda_binding.cpp"].format(**template_vars))
            
        with open(os.path.join(cuda_dir, "CMakeLists.txt"), "w") as f:
            f.write(TEMPLATES["cuda_cmakelists.txt"].format(**template_vars))
            
        # Create CUDA __init__.py
        with open(os.path.join(cuda_dir, "__init__.py"), "w") as f:
            f.write("# Make package importable")
    
    if "numpy" in impls:
        numpy_dir = os.path.join(task_dir, "numpy")
        os.makedirs(numpy_dir, exist_ok=True)
        
        # Create NumPy implementation
        with open(os.path.join(numpy_dir, f"{task_file_name}.py"), "w") as f:
            f.write(TEMPLATES["numpy_impl.py"].format(**template_vars))
            
        # Create NumPy __init__.py
        with open(os.path.join(numpy_dir, "__init__.py"), "w") as f:
            f.write("# Make package importable")
    
    if "pytorch" in impls:
        pytorch_dir = os.path.join(task_dir, "pytorch")
        os.makedirs(pytorch_dir, exist_ok=True)
        
        # Create PyTorch implementation
        with open(os.path.join(pytorch_dir, f"{task_file_name}.py"), "w") as f:
            f.write(TEMPLATES["pytorch_impl.py"].format(**template_vars))
            
        # Create PyTorch __init__.py
        with open(os.path.join(pytorch_dir, "__init__.py"), "w") as f:
            f.write("# Make package importable")
    
    print(f"Successfully created task template: {task_name} in {task_dir}")
    print(f"Next steps:")
    print(f"1. Implement the task-specific code in the generated files")
    print(f"2. Update the CMakeLists.txt in the tasks directory to include your new task")
    print(f"3. Test your implementation using the CLI: python -m infinite_ml.cli list")

def main():
    parser = argparse.ArgumentParser(description='Generate a new task template for Infinite ML')
    parser.add_argument('task_name', help='Human-readable task name (e.g., "Vector Addition")')
    parser.add_argument('--task-id', help='Task ID (e.g., "vector_addition", generated from task_name if not provided)')
    parser.add_argument('--function-name', help='Function name for the task (default derived from task_name)')
    parser.add_argument('--project-root', help='Project root directory', default=Path.cwd())
    parser.add_argument('--impls', nargs='+', default=['cuda', 'numpy', 'pytorch'],
                        help='Implementations to generate (default: cuda numpy pytorch)')
    
    args = parser.parse_args()
    
    create_task_folder(
        args.task_name,
        args.task_id,
        args.function_name,
        args.project_root,
        args.impls
    )

if __name__ == "__main__":
    main()
