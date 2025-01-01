#ifndef ADDITION_CUH
#define ADDITION_CUH

#include <string>
#include <vector>
#include <cuda_runtime.h>

// Forward declaration of Logger
class Logger;

// Structure to store performance results
struct PerformanceResult {
    std::string name;
    int blocks;
    int threadsPerBlock;
    double timeMs;
    int numAdditionsPerformed;
    std::string deviceName;
};

// Function prototypes for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);
PerformanceResult vectorAddCuda(const float *A, const float *B, float *C, int numElements, int threadsPerBlock);
PerformanceResult vectorAddCPU(const float *A, const float *B, float *C, int numElements);

// Utility functions
void verifyResult(const float *A, const float *B, const float *C, int numElements);
void logPerformanceResultsCuda(const PerformanceResult &result);
void logPerformanceResultsCPU(const PerformanceResult &result);

// Logger initialization function
void initializeLogger(const std::string& logFilename);

// Function to clean up logger
void cleanupLogger();

// Function to print performance results to console
void printPerformanceResults(const PerformanceResult &result);

#endif // ADDITION_CUH
