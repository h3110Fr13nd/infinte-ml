#include <stdio.h>
#include "addition.cuh"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "logger.h"

// Global logger instance
Logger* logger = nullptr;

// Initialize logger
void initializeLogger(const std::string& logFilename) {
    logger = new Logger(logFilename);
}

// Clean up logger
void cleanupLogger() {
    delete logger;
    logger = nullptr;
}

// Function to print performance results
void printPerformanceResults(const PerformanceResult &result) {
    std::cout << "Name: " << result.name << "\n";
    std::cout << "Blocks: " << result.blocks << "\n";
    std::cout << "Threads per Block: " << result.threadsPerBlock << "\n";
    std::cout << "Total Threads: " << result.blocks * result.threadsPerBlock << "\n";
    std::cout << "Total Size (bytes): " << result.blocks * result.threadsPerBlock * sizeof(float) << "\n";
    std::cout << "Total Additions Performed: " << result.numAdditionsPerformed << "\n";
    std::cout << "Time (ms): " << result.timeMs << "\n";
    std::cout << "Throughput (GB/s): " << (result.blocks * result.threadsPerBlock * sizeof(float) / (result.timeMs * 1e6)) << "\n";
    std::cout << "Throughput (Additions/s): " << (result.blocks * result.threadsPerBlock / (result.timeMs * 1e-3)) << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Throughput (Additions/Thread/ms): " << (result.blocks * result.threadsPerBlock / (result.timeMs * 1e-3)) << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Throughput (Additions/Thread/s): " << (result.blocks * result.threadsPerBlock / result.timeMs) << "\n";
    std::cout << "----------------------------------------\n";
}

// Log performance results
void logPerformanceResultsCuda(const PerformanceResult &result) {
    if (!logger) return;
    
    logger->info("Name: " + result.name);
    logger->info("Blocks: " + std::to_string(result.blocks));
    logger->info("Threads per Block: " + std::to_string(result.threadsPerBlock));
    logger->info("Total Threads: " + std::to_string(result.blocks * result.threadsPerBlock));
    logger->info("Total Size (bytes): " + std::to_string(result.blocks * result.threadsPerBlock * sizeof(float)));
    logger->info("Total Additions Performed: " + std::to_string(result.numAdditionsPerformed));
    logger->info("Time (ms): " + std::to_string(result.timeMs));
    logger->info("Throughput (GB/s): " + std::to_string((result.blocks * result.threadsPerBlock * sizeof(float) / (result.timeMs * 1e6))));
    logger->info("Throughput (Additions/s): " + std::to_string((result.blocks * result.threadsPerBlock / (result.timeMs * 1e-3))));
    logger->info("Throughput (Additions/Thread/ms): " + std::to_string((result.blocks * result.threadsPerBlock / (result.timeMs * 1e-3))));
    logger->info("Throughput (Additions/Thread/s): " + std::to_string((result.blocks * result.threadsPerBlock / result.timeMs)));
    logger->info("========================================");
    logger->info("");
    logger->info("");
    logger->info("");

}

void logPerformanceResultsCPU(const PerformanceResult &result) {
    if (!logger) return;
    
    logger->info("Name: " + result.name);
    logger->info("Blocks: " + std::to_string(result.blocks));
    logger->info("Threads per Block: " + std::to_string(result.threadsPerBlock));
    logger->info("Total Threads: " + std::to_string(result.blocks * result.threadsPerBlock));
    logger->info("Total Size (bytes): " + std::to_string(result.blocks * result.threadsPerBlock * sizeof(float)));
    logger->info("Total Additions Performed: " + std::to_string(result.numAdditionsPerformed));
    logger->info("Time (ms): " + std::to_string(result.timeMs));
    logger->info("Throughput (GB/s): " + std::to_string((result.blocks * result.threadsPerBlock * sizeof(float) / (result.timeMs * 1e6))));
    logger->info("Throughput (Additions/s): " + std::to_string((result.blocks * result.threadsPerBlock / (result.timeMs * 1e-3))));
    logger->info("========================================");
    logger->info("");
}


// Verify results
void verifyResult(const float *A, const float *B, const float *C, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "!\n";
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Test PASSED\n";
}

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}


PerformanceResult vectorAddCuda(const float *A, const float *B, float *C, int numElements, int threadsPerBlock) {
    size_t size = numElements * sizeof(float);
    std::cout << "Vector size: " << numElements << std::endl;
    std::cout << "Size in bytes: " << size << std::endl;
    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print performance results
    PerformanceResult result;
    result.name = "Vector Addition";
    result.blocks = blocksPerGrid;
    result.threadsPerBlock = threadsPerBlock;
    result.timeMs = milliseconds;
    result.numAdditionsPerformed = numElements;
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Synchronize device
    cudaDeviceSynchronize();

    return result;
}


// Host function to perform vector addition on the CPU
PerformanceResult vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    // Perform vector addition on the CPU
    for (int i = 0; i < numElements; ++i) {
        C[i] = A[i] + B[i];
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = cpuEnd - cpuStart;
    
    PerformanceResult cpuResult;
    cpuResult.name = "CPU Vector Addition";
    cpuResult.blocks = 1;
    cpuResult.threadsPerBlock = 1;
    cpuResult.timeMs = cpuDuration.count();
    cpuResult.numAdditionsPerformed = numElements;
    
    return cpuResult;
}


int main() {
    // Initialize the logger
    initializeLogger("performance.log");
    logger->debug("Starting vector addition performance test...");

    // Initialize data
    int numElementsOptions[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864};
    // int numElementsOptions[] = {1024, 2048, 4096, 8192, 16384};
    // Loop through different sizes
    int threadsPerBlockOptions[] = {32, 64, 128, 256, 512, 1024};

    std::vector<PerformanceResult> results;

    for (int numElements: numElementsOptions) {
        for (int threadsPerBlock: threadsPerBlockOptions){
            // Allocate host memory
            size_t size = numElements * sizeof(float);
            float *h_A = (float *)malloc(size);
            float *h_B = (float *)malloc(size);
            float *h_C = (float *)malloc(size);

            // Initialize input vectors Probably Randomly
            for (int i = 0; i < numElements; ++i) {
                h_A[i] = static_cast<float>(rand()) / RAND_MAX;
                h_B[i] = static_cast<float>(rand()) / RAND_MAX;
            }
            // Perform vector addition on CPU
            PerformanceResult cpuPerformance = vectorAddCPU(h_A, h_B, h_C, numElements);
            results.push_back(cpuPerformance);
            logPerformanceResultsCPU(cpuPerformance);

            // Perform vector addition on GPU
            PerformanceResult cudaPerformance = vectorAddCuda(h_A, h_B, h_C, numElements, threadsPerBlock);
            results.push_back(cudaPerformance);
            logPerformanceResultsCuda(cudaPerformance);
            
            // Verify results
            verifyResult(h_A, h_B, h_C, numElements);
            
            // Free host memory
            free(h_A);
            free(h_B);
            free(h_C);
            
            // Reset device
            cudaDeviceReset();

            // Check for errors
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << "\n";
            } else {
                std::cout << "CUDA operation completed successfully.\n";
            }
        }
    }

    // Clean up logger
    cleanupLogger();
    return 0;
}