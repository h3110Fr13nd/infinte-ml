# Vector Addition: A CUDA vs CPU Benchmark Task

## Introduction

This task implements a simple vector addition operation (C = A + B) to demonstrate the performance differences between CPU and GPU implementations. While vector addition is conceptually simple, it serves as an excellent starting point for understanding parallel computation fundamentals and the CUDA programming model.

## Implementation Overview

The task provides three separate implementations:

1. **CUDA Implementation**: Uses NVIDIA's CUDA platform to perform vector addition in parallel on the GPU
2. **NumPy Implementation**: Uses NumPy's optimized vector operations on the CPU
3. **PyTorch Implementation**: Leverages PyTorch's tensor operations (which can use either CPU or GPU)

## CUDA Implementation Details

The CUDA implementation consists of several key components:

### CUDA Kernel

The heart of the GPU implementation is the CUDA kernel function:

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
```

This kernel:
- Calculates a unique index for each thread
- Performs a single addition operation per thread
- Includes boundary checking to handle cases where thread count exceeds array size

### Performance Measurement

The implementation includes comprehensive performance measurement:

- Uses CUDA events to precisely time kernel execution
- Calculates throughput in terms of operations per second
- Reports efficiency metrics including additions per thread
- Compares performance across different input sizes and thread configurations

### Thread and Block Configuration

The code explores various configurations to find optimal performance:

- **Threads per block**: Tests values from 32 to 1024 (powers of 2)
- **Number of blocks**: Calculated based on input size and threads per block
- **Input sizes**: Tests from small arrays (1K elements) to large arrays (67M+ elements)

## Logging Infrastructure

A custom logging system helps track and analyze performance results:

- **C++ Logger Class**: Provides different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **File and Console Output**: Records results to both log files and standard output
- **Formatted Output**: Includes timestamps and structured performance data

## Build System

The task uses a sophisticated build system to handle compilation across different platforms:

### CMake Configuration

The `CMakeLists.txt` file:
- Detects CUDA capabilities of the local hardware
- Sets appropriate architecture flags
- Configures position-independent code for library building
- Sets up proper directory structures for output files

### Python Bindings

Python bindings are created using pybind11:

- Exposes C++/CUDA functions to Python
- Handles NumPy array to CUDA pointer conversions
- Provides proper error handling for Python exceptions

## Performance Analysis

The benchmark generates comprehensive performance data:

### Metrics Collected

- **Execution Time**: Raw execution time in milliseconds
- **Throughput (GB/s)**: Memory bandwidth utilization
- **Additions Per Second**: Raw computational throughput
- **Efficiency Metrics**: Performance per thread and block

### Comparative Analysis

The code allows comparing:
- CUDA vs. CPU performance
- Effects of different block sizes
- Scaling with input data size
- Memory transfer overhead vs. computation time

## Example Usage

```python
import numpy as np
import infinite_ml.tasks.task_001_addition.cuda.addition as ca

# Initialize arrays
size = 1_000_000
a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
result = np.zeros_like(a)

# Use CPU implementation
cpu_perf = ca.vector_add_cpu(a, b, result)
print(f"CPU time: {cpu_perf.time_ms} ms")

# Use CUDA implementation with 256 threads per block
cuda_perf = ca.vector_add_cuda(a, b, result, 256)
print(f"CUDA time: {cuda_perf.time_ms} ms")
print(f"Speedup: {cpu_perf.time_ms / cuda_perf.time_ms}x")
```

## Key Insights

From experimenting with this task, several key insights emerge:

1. **Optimal Thread Count**: For vector addition, block sizes of 256-512 threads often provide the best performance balance

2. **Overhead Dominance**: For small arrays, the overhead of transferring data to/from the GPU often exceeds the computational benefit

3. **Memory Bandwidth Limitation**: Vector addition is memory-bound rather than compute-bound, so performance is primarily limited by memory bandwidth

4. **Scaling Efficiency**: As input size increases, the GPU performance advantage grows, demonstrating better scaling characteristics

## Future Improvements

Potential enhancements to this task:

- Implement streams for concurrent kernel execution and memory transfers
- Add memory coalescing optimizations
- Implement shared memory usage examples
- Provide visualizations of performance results
- Add multi-GPU support for larger datasets

## Conclusion

This vector addition task serves as a foundational example for the Infinite ML project, demonstrating the basic principles of CUDA programming and parallel computation while providing a robust performance comparison framework.
