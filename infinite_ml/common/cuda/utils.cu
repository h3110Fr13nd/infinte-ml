#include "utils.cuh"
#include <cuda_runtime.h>

// Simple utility to check if CUDA is available
bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// Get total device memory in bytes - remove default argument here
size_t getDeviceMemory(int deviceId) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
        return 0;
    }
    return prop.totalGlobalMem;
}

// Synchronize device (wrapper for cudaDeviceSynchronize)
void syncDevice() {
    cudaDeviceSynchronize();
}
