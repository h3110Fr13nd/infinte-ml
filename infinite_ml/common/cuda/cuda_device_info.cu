#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <sstream>

std::string getCudaDeviceInfo() {
    int dev_count;
    cudaDeviceProp prop;
    std::stringstream ss;

    cudaGetDeviceCount(&dev_count);
    
    if (dev_count == 0) {
        return "No CUDA devices found.";
    }
    
    cudaGetDeviceProperties(&prop, 0);

    ss << ">> CUDA enabled devices in the system: " << dev_count << "\n\n";

    ss << ">> Max grid size: (" << prop.maxGridSize[0] << ", " 
       << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
    ss << ">> Max block size: " << prop.maxThreadsPerBlock << "\n\n";

    ss << ">> Number of SMs: " << prop.multiProcessorCount << "\n";
    ss << ">> Clock rate of the SMs (in kHz): " << prop.clockRate << "\n";

    ss << ">> Max threads dimension: (" << prop.maxThreadsDim[0] << ", " 
       << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    ss << ">> Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n\n";

    ss << ">> Registers available per block: " << prop.regsPerBlock << "\n";
    ss << ">> Registers available per SM: " << prop.regsPerMultiprocessor << "\n\n";

    ss << ">> Warp size (threads per warp): " << prop.warpSize << "\n\n";
    ss << ">> Shared memory size per block: " << prop.sharedMemPerBlock << " bytes\n";
    ss << ">> Shared memory size per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n\n";

    ss << ">> L2 cache size: " << prop.l2CacheSize << " bytes\n";

    return ss.str();
}

int getDeviceCount() {
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    return dev_count;
}
