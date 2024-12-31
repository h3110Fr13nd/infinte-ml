#pragma once

// #include <string>
#include <cuda_runtime.h>



// A struct to hold result information 
// struct HelloWorldResult {
//     bool success;
//     std::string message;
//     std::string errorMessage;
    
//     // Add a constructor to initialize members
//     HelloWorldResult() : success(false), message(""), errorMessage("") {}
// };

// CUDA kernel declaration
__global__ void helloWorldKernel();

void helloWorldCuda();
// Host function to launch the kernel and process data
// Modify to pass result by reference to ensure it's modifiable
// void helloWorldCuda(HelloWorldResult& result, int* data, int size);

// Simple function to demonstrate CUDA functionality
// Return by value but ensure implementation uses a local variable that's returned
// HelloWorldResult sayHelloFromCuda();
