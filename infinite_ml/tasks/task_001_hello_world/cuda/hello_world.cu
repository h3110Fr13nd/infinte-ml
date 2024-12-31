#include <stdio.h>
#include <cuda_runtime.h>
#include "hello_world.cuh"

// CUDA kernel that prints a Hello World message
__global__ void helloWorldKernel() {
    printf("Hello World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

void helloWorldCuda() {
    // Print message from host (CPU)
    printf("Hello World from the host (CPU)!\n");
    
    // Launch kernel with 2 blocks and 4 threads per block
    helloWorldKernel<<<2, 4>>>();
    
    // Synchronize to ensure all printf statements from the kernel are displayed
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return;
    }
    
    printf("CUDA kernel execution completed successfully!\n");
    return;
}


int main() {
    // Print message from host (CPU)
    printf("Hello World from the host (CPU)!\n");
    
    // Launch kernel with 2 blocks and 4 threads per block
    helloWorldKernel<<<2, 4>>>();
    
    // Synchronize to ensure all printf statements from the kernel are displayed
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("CUDA kernel execution completed successfully!\n");
    return 0;
}