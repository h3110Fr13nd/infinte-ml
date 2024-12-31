#pragma once

#include <cuda_runtime.h>
#include <string>

// Device information functions
std::string getCudaDeviceInfo();
int getDeviceCount();

// Utility functions
bool isCudaAvailable();
size_t getDeviceMemory(int deviceId = 0);  // Default argument defined here only
void syncDevice();

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Grid and block dimension calculation helpers
inline dim3 getGridDim(int n, int blockSize) {
    return dim3((n + blockSize - 1) / blockSize);
}

inline dim3 getBlockDim(int blockSize) {
    return dim3(blockSize);
}
